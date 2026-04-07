"""
Full VAE marker gene pipeline

Steps
-----
1. Select dataset
2. Filter and log-normalize (filter_anndata)
3. Identify highly variable genes (identify_hvg)
4. Train VAE
5. Embed all cells into the latent space
6. Extract marker genes per latent dimension via decoder probing
7. Save all outputs to an experiment directory

Usage
-----
Run with defaults (interactive dataset selection):
    python scripts/run_pipeline.py

Specify dataset and output directory directly:
    python scripts/run_pipeline.py --dataset fetal --out experiments/fetal_run1

Override key hyperparameters:
    python scripts/run_pipeline.py --dataset fetal --n_hvg 3000 --latent_dim 32 --epochs 150 --beta 2.0
"""

import argparse
import os
import sys
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR, EXPERIMENTS_DIR, FIGURES_DIR
from src.preprocessing.preprocessing import filter_anndata, identify_hvg, regress_confounders
from src.training.train_vae import train_vae, embed_adata
from src.analysis.marker_genes import decoder_probing, markers_to_dataframe, summarise_latent_dims
from src.analysis.embedding_analysis import compare_vae_pca_clustering
from src.analysis.dge import run_cluster_dge
from src.preprocessing.gene_lists import GUT_CELL_TYPE_MARKERS, PBMC_CELL_TYPE_MARKERS


# ---------------------------------------------------------------------------
# Available datasets
# ---------------------------------------------------------------------------

DATASETS = {
    "pbmc":      "pbmc_3k/pbmc3k_raw.h5ad",
    "fetal":     "elmentiate_2021/fetal_RAWCOUNTS_cellxgene.h5ad",
    "pediatric": "elmentiate_2021/pediatric_RAWCOUNTS_cellxgene_c.h5ad",
    "full":      "elmentiate_2021/Full_obj_raw_counts_nosoupx_v2.h5ad",
}

# Default tissue type per dataset (used to select cell-type marker set)
DATASET_TISSUE = {
    "pbmc":      "pbmc",
    "fetal":     "gut",
    "pediatric": "gut",
    "full":      "gut",
}

TISSUE_MARKERS = {
    "gut":  GUT_CELL_TYPE_MARKERS,
    "pbmc": PBMC_CELL_TYPE_MARKERS,
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full VAE marker gene pipeline on a gut scRNA-seq dataset"
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default=None,
        help="Dataset to run on.  If not provided, an interactive prompt is shown.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for the experiment.  Defaults to experiments/<dataset>_run.",
    )

    # Filtering
    parser.add_argument("--min_cells",  type=int,   default=3,    help="Minimum cells a gene must appear in (default: 3)")
    parser.add_argument("--min_genes",  type=int,   default=200,  help="Minimum genes per cell (default: 200)")
    parser.add_argument("--max_genes",  type=int,   default=6000, help="Maximum genes per cell (default: 6000)")
    parser.add_argument("--max_mito",   type=float, default=20.0, help="Maximum percent mitochondrial counts (default: 20)")
    parser.add_argument("--no_filter_noncoding", action="store_true",
                        help="Disable filtering of non-coding genes (genes with '.' in name)")

    # HVG
    parser.add_argument("--n_hvg",      type=int,   default=3000, help="Number of highly variable genes to use (default: 3000)")

    # VAE architecture
    parser.add_argument("--latent_dim", type=int,   default=32,   help="Latent space dimensionality (default: 32)")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[512, 256],
                        help="Encoder/decoder hidden layer sizes (default: 512 256)")
    parser.add_argument("--dropout",    type=float, default=0.1,  help="Dropout rate (default: 0.1)")

    # Training
    parser.add_argument("--epochs",     type=int,   default=200,  help="Maximum training epochs (default: 200)")
    parser.add_argument("--batch_size", type=int,   default=512,  help="Mini-batch size (default: 512)")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--beta",       type=float, default=1.0,  help="KL weight beta (default: 1.0)")
    parser.add_argument("--patience",   type=int,   default=20,   help="Early stopping patience (default: 20)")

    # Confounder regression
    parser.add_argument("--no_regress_sex", action="store_true",
                        help="Disable sex confounder regression (default: enabled)")
    parser.add_argument("--regress_cell_cycle", action="store_true",
                        help="Enable cell cycle confounder regression (default: disabled)")

    # Marker gene extraction
    parser.add_argument("--top_k",      type=int,   default=50,   help="Number of marker genes per latent dim (default: 50)")
    parser.add_argument("--probe_scale", type=float, default=3.0, help="Unit vector scale for decoder probing (default: 3.0)")

    # Clustering comparison
    parser.add_argument("--n_pcs",           type=int,   default=50,  help="Number of PCs for PCA clustering comparison (default: 50)")
    parser.add_argument("--n_neighbors",     type=int,   default=15,  help="k for kNN graphs in clustering comparison (default: 15)")
    parser.add_argument("--leiden_res",      type=float, default=0.5, help="Leiden resolution for clustering comparison (default: 0.5)")
    parser.add_argument("--color_by",        nargs="+",  default=None,
                        help="Additional obs column(s) to colour UMAP panels by (e.g. cell_type sample)")
    parser.add_argument("--skip_clustering", action="store_true",
                        help="Skip the VAE vs PCA clustering comparison step")

    # Tissue / cell-type marker set
    parser.add_argument(
        "--tissue",
        choices=list(TISSUE_MARKERS.keys()),
        default=None,
        help="Cell-type marker set to use for cluster annotation. "
             "Auto-selected from dataset if omitted (gut datasets → 'gut', pbmc → 'pbmc').",
    )

    # DGE + cell-type annotation
    parser.add_argument("--n_dge_genes",   type=int, default=50,
                        help="Genes per cluster retained by DGE (default: 50)")
    parser.add_argument("--n_dot_genes",   type=int, default=5,
                        help="Genes per cluster shown in the dot plot (default: 5)")
    parser.add_argument("--skip_dge",      action="store_true",
                        help="Skip DGE and cell-type annotation step")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Interactive prompt (used when script is run with no CLI arguments)
# ---------------------------------------------------------------------------

def _ask(prompt: str, default, cast=str):
    """Prompt with a default value; return default on empty input."""
    raw = input(f"  {prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except (ValueError, TypeError):
        print(f"    Invalid input, using default ({default})")
        return default


def _ask_bool(prompt: str, default: bool) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input(f"  {prompt} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def prompt_interactive() -> argparse.Namespace:
    """Prompt the user for key pipeline parameters interactively."""
    print("\n=== VAE Marker Gene Pipeline ===\n")

    # --- Dataset ---
    print("Available datasets:")
    keys = list(DATASETS.keys())
    for i, k in enumerate(keys, 1):
        print(f"  [{i}] {k}")
    while True:
        raw = input("  Select dataset (name or number): ").strip().lower()
        if raw in DATASETS:
            dataset = raw
            break
        if raw.isdigit() and 1 <= int(raw) <= len(keys):
            dataset = keys[int(raw) - 1]
            break
        print(f"    Invalid — choose from: {', '.join(keys)}")

    # --- Tissue (auto-default from dataset) ---
    default_tissue = DATASET_TISSUE.get(dataset, "gut")
    tissue_keys = list(TISSUE_MARKERS.keys())
    raw = input(f"  Tissue {tissue_keys} [{default_tissue}]: ").strip().lower()
    tissue = raw if raw in TISSUE_MARKERS else default_tissue

    # --- Key training params ---
    epochs     = _ask("Max epochs",       200, int)
    latent_dim = _ask("Latent dimensions", 32, int)

    # --- Steps ---
    skip_clustering = _ask_bool("Skip clustering + DGE?", False)
    skip_dge        = _ask_bool("Skip DGE only?", False) if not skip_clustering else False

    # All other params use defaults
    n_hvg       = 3000
    beta        = 1.0
    patience    = 20
    leiden_res  = 0.5
    n_neighbors = 15
    regress_sex = True
    regress_cc  = False

    return argparse.Namespace(
        dataset=dataset,
        out=None,
        tissue=tissue,
        # filtering (use defaults)
        min_cells=3,
        min_genes=200,
        max_genes=6000,
        max_mito=20.0,
        no_filter_noncoding=False,
        # HVG
        n_hvg=n_hvg,
        # VAE
        latent_dim=latent_dim,
        hidden_dims=[512, 256],
        dropout=0.1,
        # training
        epochs=epochs,
        batch_size=512,
        lr=1e-3,
        beta=beta,
        patience=patience,
        # confounders
        no_regress_sex=not regress_sex,
        regress_cell_cycle=regress_cc,
        # marker genes
        top_k=50,
        probe_scale=3.0,
        # clustering
        n_pcs=50,
        n_neighbors=n_neighbors,
        leiden_res=leiden_res,
        color_by=None,
        skip_clustering=skip_clustering,
        # DGE
        n_dge_genes=50,
        n_dot_genes=5,
        skip_dge=skip_dge,
    )


# ---------------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------------

def select_dataset(choice: str | None) -> tuple[str, str]:
    """
    Return the dataset key and resolved file path, prompting interactively if needed

    :param choice: dataset key string from argparse, or None to prompt the user
    :return: tuple of (dataset_key, absolute_file_path)
    """
    if choice is None:
        print("\nAvailable datasets:")
        for i, (key, rel_path) in enumerate(DATASETS.items(), start=1):
            print(f"  [{i}] {key:12s}  {rel_path}")
        while True:
            raw = input("\nSelect dataset (name or number): ").strip().lower()
            if raw in DATASETS:
                choice = raw
                break
            if raw.isdigit() and 1 <= int(raw) <= len(DATASETS):
                choice = list(DATASETS.keys())[int(raw) - 1]
                break
            print(f"  Invalid selection '{raw}'. Please enter one of: {', '.join(DATASETS)}")

    file_path = os.path.join(DATA_DIR, DATASETS[choice])
    if not os.path.exists(file_path):
        sys.exit(f"ERROR: Dataset file not found: {file_path}")

    return choice, file_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    # If run directly with no CLI flags, use interactive prompts
    if len(sys.argv) == 1:
        args = prompt_interactive()
    else:
        args = parse_args()

    # 1. Dataset selection
    print("=" * 60)
    print("  Gut Marker Analysis — VAE Pipeline")
    print("=" * 60)

    dataset_key, file_path = select_dataset(args.dataset)
    tissue = args.tissue or DATASET_TISSUE.get(dataset_key, "gut")
    cell_type_markers = TISSUE_MARKERS[tissue]
    print(f"\n[1/9] Dataset : {dataset_key}  ({file_path})")
    print(f"      Tissue  : {tissue}  ({len(cell_type_markers)} cell types)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(EXPERIMENTS_DIR, f"{dataset_key}_{timestamp}")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    print(f"      Output  : {out_dir}")

    # 2. Filter and normalize
    print(f"\n[2/9] Filtering and normalizing ...")
    adata = filter_anndata(
        file_path=file_path,
        min_cells=args.min_cells,
        min_genes=args.min_genes,
        max_genes=args.max_genes,
        max_mito=args.max_mito,
        filter_noncoding=not args.no_filter_noncoding,
    )
    print(f"      Cells after filtering : {adata.n_obs:,}")
    print(f"      Genes after filtering : {adata.n_vars:,}")

    # 3. Identify highly variable genes
    print(f"\n[3/9] Identifying {args.n_hvg} highly variable genes ...")
    adata = identify_hvg(adata, n_top_genes=args.n_hvg, filter_nonHVG=True)
    print(f"      Genes after HVG filter : {adata.n_vars:,}")

    # 4. Regress out confounders
    regress_sex = not args.no_regress_sex
    regress_cc  = args.regress_cell_cycle
    regression_desc = []
    if regress_sex:
        regression_desc.append("sex")
    if regress_cc:
        regression_desc.append("cell cycle")
    desc_str = ", ".join(regression_desc) if regression_desc else "none"
    print(f"\n[4/9] Regressing out confounders ({desc_str}) → adata.layers['regressed'] ...")
    adata = regress_confounders(adata, regress_sex=regress_sex, regress_cell_cycle=regress_cc)
    print(f"      adata.layers['regressed'] shape : {adata.layers['regressed'].shape}")

    # 5. Train VAE
    print(f"\n[5/9] Training VAE on 'regressed' layer  (latent_dim={args.latent_dim}, beta={args.beta}) ...")
    vae_config = {
        "latent_dim":              args.latent_dim,
        "hidden_dims":             args.hidden_dims,
        "dropout":                 args.dropout,
        "n_epochs":                args.epochs,
        "batch_size":              args.batch_size,
        "learning_rate":           args.lr,
        "beta":                    args.beta,
        "early_stopping_patience": args.patience,
        "use_hvg":                 False,   # HVGs already filtered above
    }
    model, history = train_vae(adata, config=vae_config, save_dir=out_dir, layer='regressed')

    # 6. Embed all cells
    print(f"\n[6/9] Embedding {adata.n_obs:,} cells into latent space ...")
    Z = embed_adata(model, adata, use_hvg=False, layer='regressed')
    adata.obsm["X_vae"] = Z

    embeddings_path = os.path.join(out_dir, "latent_embeddings.npy")
    np.save(embeddings_path, Z)
    print(f"      Saved embeddings → {embeddings_path}  shape={Z.shape}")

    adata_path = os.path.join(out_dir, "adata.h5ad")
    adata.write_h5ad(adata_path)
    print(f"      Saved AnnData    → {adata_path}")

    # 7. Extract marker genes
    print(f"\n[7/9] Extracting top {args.top_k} marker genes per latent dimension ...")
    gene_names = adata.var_names.tolist()
    markers = decoder_probing(model, gene_names, top_k=args.top_k, scale=args.probe_scale)

    markers_df = markers_to_dataframe(markers)
    markers_path = os.path.join(out_dir, "marker_genes.csv")
    markers_df.to_csv(markers_path, index=False)
    print(f"      Saved marker genes → {markers_path}")

    summary_df = summarise_latent_dims(model, gene_names, top_k=10, scale=args.probe_scale)
    summary_path = os.path.join(out_dir, "marker_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # Print top 5 genes for first 8 dims as a quick sanity check
    print(f"\n      Top 5 marker genes for first 8 latent dimensions:")
    print(summary_df.iloc[:5, :8].to_string(index=False))

    # 8. Clustering comparison: VAE latent space vs PCA
    if not args.skip_clustering:
        print(f"\n[8/9] Comparing VAE vs PCA Leiden clustering ...")
        compare_vae_pca_clustering(
            adata,
            out_dir=fig_dir,
            n_pcs=args.n_pcs,
            n_neighbors=args.n_neighbors,
            leiden_resolution=args.leiden_res,
            color_by=args.color_by,
        )
    else:
        print(f"\n[8/9] Skipping clustering comparison (--skip_clustering)")

    # 9. DGE + cell-type annotation
    if not args.skip_dge and not args.skip_clustering:
        print(f"\n[9/9] Running DGE and cell-type annotation ...")
        run_cluster_dge(
            adata,
            out_dir=fig_dir,
            n_dge_genes=args.n_dge_genes,
            n_dot_genes=args.n_dot_genes,
            cell_type_markers=cell_type_markers,
        )
    else:
        print(f"\n[9/9] Skipping DGE (--skip_dge or --skip_clustering)")

    # Re-save adata now that it has DGE results, cluster annotations, and UMAPs
    adata.write_h5ad(adata_path)
    print(f"      Updated AnnData → {adata_path}")

    # Save run config
    run_config = vars(args)
    run_config["dataset_file"] = file_path
    run_config["n_cells_filtered"] = int(adata.n_obs)
    run_config["n_genes_hvg"] = int(adata.n_vars)
    run_config["regress_sex"] = regress_sex
    run_config["regress_cell_cycle"] = regress_cc
    run_config["layer"] = "regressed"
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    figure_files = sorted(f for f in os.listdir(fig_dir) if f.endswith(".png"))
    table_files  = sorted(f for f in os.listdir(out_dir) if f.endswith(".csv"))

    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete.")
    print(f"  Experiment dir : {out_dir}")
    print(f"  Figures        : {fig_dir}")
    if figure_files:
        print(f"\n  Figures:")
        for fname in figure_files:
            print(f"    {fname}")
    if table_files:
        print(f"\n  Tables:")
        for fname in table_files:
            print(f"    {fname}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
