"""
Classical differential gene expression (DGE) on Leiden clusters,
cell-type scoring against curated gut marker gene sets, and
cluster visualisation.

Main entry point
----------------
    from src.analysis.dge import run_cluster_dge

    run_cluster_dge(adata, out_dir="experiments/my_run/")

This runs DGE, scores + annotates clusters, and saves four figures:

    dge_dotplot_vae.png         — top DGE markers per VAE cluster
    dge_dotplot_pca.png         — top DGE markers per PCA cluster
    cell_type_scores_vae.png    — cell-type score heatmap per VAE cluster
    cell_type_scores_pca.png    — cell-type score heatmap per PCA cluster
    annotated_umap.png          — 1×2 UMAP coloured by assigned cell type
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


# ---------------------------------------------------------------------------
# Curated gut cell-type marker genes (Elmentaite et al. 2021 + literature)
# ---------------------------------------------------------------------------

GUT_CELL_TYPE_MARKERS: dict[str, list[str]] = {
    # Epithelial
    "Stem/TA":          ["LGR5", "OLFM4", "ASCL2", "AXIN2", "MKI67", "TOP2A", "PCNA", "SOX9"],
    "Enterocyte":       ["FABP1", "FABP2", "ALPI", "SLC5A1", "APOA4", "APOB", "CYP3A4", "ALDOB"],
    "Goblet":           ["MUC2", "TFF3", "CLCA1", "SPDEF", "AGR2", "FCGBP"],
    "Enteroendocrine":  ["CHGA", "CHGB", "NEUROD1", "PAX4", "NEUROG3", "SST", "GCG"],
    "Paneth":           ["DEFA5", "DEFA6", "LYZ", "MMP7", "ITLN2"],
    "Tuft":             ["DCLK1", "POU2F3", "TRPM5", "PTGS1", "IL25"],
    "M cell":           ["GP2", "SPIB", "CCL20", "MARCKS"],
    # Stromal
    "Fibroblast":       ["COL1A1", "COL3A1", "DCN", "LUM", "PDGFRA", "VIM"],
    "Endothelial":      ["PECAM1", "CDH5", "VWF", "CLDN5", "ENG", "PLVAP"],
    "Smooth muscle":    ["ACTA2", "MYH11", "TAGLN", "CNN1", "MYLK"],
    "Pericyte":         ["RGS5", "PDGFRB", "NOTCH3", "MCAM", "ABCC9"],
    # Immune
    "Macrophage":       ["CD68", "LYZ", "CSF1R", "MRC1", "MARCO", "MSR1"],
    "Dendritic cell":   ["CLEC9A", "XCR1", "ITGAX", "FCER1A", "CLEC10A"],
    "T cell":           ["CD3D", "CD3E", "TRAC", "CD4", "CD8A", "IL7R"],
    "B cell":           ["MS4A1", "CD79A", "CD79B", "IGHM", "BANK1"],
    "Plasma cell":      ["MZB1", "SDC1", "IGHG1", "IGHA1", "JCHAIN", "DERL3"],
    "NK/ILC":           ["GNLY", "NKG7", "KLRD1", "RORC", "GATA3", "EOMES"],
    "Mast cell":        ["TPSAB1", "CPA3", "KIT", "HPGDS", "MS4A2"],
}


# ---------------------------------------------------------------------------
# DGE
# ---------------------------------------------------------------------------


def run_dge(
    adata,
    groupby_key: str,
    method: str = "wilcoxon",
    n_genes: int = 50,
    key_added: str | None = None,
) -> str:
    """
    Run one-vs-rest Wilcoxon rank-sum DGE for each Leiden cluster.

    Uses ``adata.X`` (log-normalised expression) so that raw fold-changes
    and expression fractions are on an interpretable scale regardless of
    which layer was used for VAE training.

    :param adata: AnnData object with clustering labels in ``adata.obs``
    :param groupby_key: obs column to group by (e.g. ``'leiden_vae'``)
    :param method: statistical test passed to ``sc.tl.rank_genes_groups``.
        Default is ``'wilcoxon'``
    :param n_genes: number of top genes to retain per cluster.  Default is 50
    :param key_added: key under which results are stored in ``adata.uns``.
        Defaults to ``'dge_<groupby_key>'``
    :return: the key used to store results in ``adata.uns``
    """
    if key_added is None:
        key_added = f"dge_{groupby_key}"

    print(f"[dge] Running {method} DGE on '{groupby_key}' ({n_genes} genes/cluster) ...")
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby_key,
        method=method,
        n_genes=n_genes,
        key_added=key_added,
        pts=True,           # store fraction of cells expressing each gene
    )
    n_clusters = len(adata.obs[groupby_key].cat.categories)
    print(f"[dge] Done — {n_clusters} clusters, results in adata.uns['{key_added}']")
    return key_added


# ---------------------------------------------------------------------------
# Cell-type scoring and cluster annotation
# ---------------------------------------------------------------------------


def score_and_annotate_clusters(
    adata,
    groupby_key: str,
    cell_type_markers: dict[str, list[str]] | None = None,
    score_key_prefix: str = "ct_score",
) -> tuple[str, pd.DataFrame]:
    """
    Score every cell against curated gut cell-type gene sets, then label each
    Leiden cluster with its highest mean-scoring cell type.

    :param adata: AnnData object
    :param groupby_key: obs column containing cluster labels
    :param cell_type_markers: dict mapping cell-type name → list of marker genes.
        Defaults to :data:`GUT_CELL_TYPE_MARKERS`
    :param score_key_prefix: prefix for per-cell score columns added to
        ``adata.obs``.  Default is ``'ct_score'``
    :return: tuple of (annotation_obs_key, score_matrix_DataFrame) where
        *annotation_obs_key* is the new ``adata.obs`` column holding the
        assigned cell-type label for each cell, and *score_matrix_DataFrame*
        is a ``(n_clusters × n_cell_types)`` DataFrame of mean scores
    """
    if cell_type_markers is None:
        cell_type_markers = GUT_CELL_TYPE_MARKERS

    score_cols: list[tuple[str, str]] = []  # (cell_type, obs_key)

    for cell_type, genes in cell_type_markers.items():
        genes_present = [g for g in genes if g in adata.var_names]
        if not genes_present:
            print(f"[dge] Warning: no genes found in adata for '{cell_type}', skipping")
            continue

        obs_key = f"{score_key_prefix}_{cell_type.replace('/', '_').replace(' ', '_')}"
        sc.tl.score_genes(adata, gene_list=genes_present, score_name=obs_key)
        score_cols.append((cell_type, obs_key))

    # Build cluster × cell-type score matrix
    clusters = adata.obs[groupby_key].cat.categories
    score_matrix = pd.DataFrame(
        index=clusters,
        columns=[ct for ct, _ in score_cols],
        dtype=float,
    )
    for cell_type, obs_key in score_cols:
        score_matrix[cell_type] = adata.obs.groupby(groupby_key)[obs_key].mean()

    # Assign each cluster the cell type with the highest mean score
    annotation_key = f"{groupby_key}_cell_type"
    cluster_labels = score_matrix.idxmax(axis=1)
    adata.obs[annotation_key] = (
        adata.obs[groupby_key].map(cluster_labels).astype("category")
    )

    print(f"[dge] Cell-type assignments for '{groupby_key}':")
    for cluster, label in cluster_labels.items():
        print(f"       cluster {cluster:>3} → {label}")

    return annotation_key, score_matrix


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def save_dge_table(
    adata,
    dge_key: str,
    out_dir: str,
    label: str = "",
) -> str:
    """
    Export the full DGE results table to a CSV file.

    Columns: ``cluster``, ``gene``, ``score``, ``log2fc``, ``pval``, ``pval_adj``,
    ``pct_expressing`` (fraction of cells in the cluster expressing the gene).

    :param adata: AnnData object
    :param dge_key: key in ``adata.uns`` produced by :func:`run_dge`
    :param out_dir: directory to save the CSV
    :param label: short string appended to the filename
        (e.g. ``'vae'`` → ``dge_results_vae.csv``)
    :return: path to the saved CSV
    """
    df = sc.get.rank_genes_groups_df(adata, group=None, key=dge_key)
    df = df.rename(columns={
        "group":          "cluster",
        "names":          "gene",
        "scores":         "score",
        "logfoldchanges": "log2fc",
        "pvals":          "pval",
        "pvals_adj":      "pval_adj",
        "pct_nz_group":   "pct_expressing",
    })
    # Sort by cluster then descending score
    df = df.sort_values(["cluster", "score"], ascending=[True, False])

    fname = f"dge_results_{label}.csv" if label else "dge_results.csv"
    fpath = os.path.join(out_dir, fname)
    df.to_csv(fpath, index=False)
    print(f"[dge] Saved DGE table ({len(df):,} rows) → '{fpath}'")
    return fpath


def plot_dge_dotplot(
    adata,
    dge_key: str,
    groupby_key: str,
    out_dir: str,
    n_genes: int = 5,
    label: str = "",
) -> None:
    """
    Dot plot of the top *n_genes* DGE markers per cluster.

    Dot colour encodes mean expression (z-scored across clusters);
    dot size encodes the fraction of cells expressing the gene.

    :param adata: AnnData object
    :param dge_key: key in ``adata.uns`` produced by :func:`run_dge`
    :param groupby_key: obs column used as the y-axis grouping
    :param out_dir: directory to save the figure
    :param n_genes: top genes per cluster to display.  Default is 5
    :param label: short string appended to the filename
        (e.g. ``'vae'`` → ``dge_dotplot_vae.png``)
    """
    top_genes = _extract_top_genes(adata, dge_key, n_genes)

    dp = sc.pl.dotplot(
        adata,
        var_names=top_genes,
        groupby=groupby_key,
        standard_scale="var",
        return_fig=True,
    )
    fname = f"dge_dotplot_{label}.png" if label else "dge_dotplot.png"
    fpath = os.path.join(out_dir, fname)
    dp.savefig(fpath, bbox_inches="tight", dpi=150)
    print(f"[dge] Saved dot plot → '{fpath}'")


def plot_cell_type_scores(
    score_matrix: pd.DataFrame,
    out_dir: str,
    label: str = "",
) -> None:
    """
    Heatmap of mean cell-type scores per cluster.

    Scores are z-scored column-wise (per cell type) so that all cell types
    are on a comparable scale.

    :param score_matrix: ``(n_clusters × n_cell_types)`` DataFrame from
        :func:`score_and_annotate_clusters`
    :param out_dir: directory to save the figure
    :param label: short string appended to the filename
    """
    # z-score each cell type column so scales are comparable
    normed = (score_matrix - score_matrix.mean()) / (score_matrix.std() + 1e-9)

    n_clusters, n_types = normed.shape
    fig, ax = plt.subplots(figsize=(max(8, n_types * 0.6), max(4, n_clusters * 0.5)))

    im = ax.imshow(normed.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    plt.colorbar(im, ax=ax, label="z-scored mean score", shrink=0.6)

    ax.set_xticks(range(n_types))
    ax.set_xticklabels(normed.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f"Cluster {c}" for c in normed.index], fontsize=8)
    ax.set_title(
        f"Cell-type scores per cluster — {label.upper() if label else 'clusters'}",
        fontsize=11,
        pad=10,
    )

    # Annotate each cell with the z-score
    for i in range(n_clusters):
        for j in range(n_types):
            val = normed.values[i, j]
            ax.text(
                j, i, f"{val:.1f}",
                ha="center", va="center",
                fontsize=6,
                color="white" if abs(val) > 1.2 else "black",
            )

    plt.tight_layout()
    fname = f"cell_type_scores_{label}.png" if label else "cell_type_scores.png"
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[dge] Saved cell-type score heatmap → '{fpath}'")


def plot_annotated_umap(
    adata,
    out_dir: str,
    vae_annotation_key: str = "leiden_vae_cell_type",
    pca_annotation_key: str = "leiden_pca_cell_type",
) -> None:
    """
    Side-by-side UMAP coloured by the assigned cell-type labels from the VAE
    and PCA clusterings.

    :param adata: AnnData object with ``X_umap_vae``, ``X_umap_pca``, and the
        two annotation obs columns already populated
    :param out_dir: directory to save the figure
    :param vae_annotation_key: obs column with VAE cluster cell-type labels
    :param pca_annotation_key: obs column with PCA cluster cell-type labels
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), squeeze=False)

    _scatter_annotated(adata, axes[0, 0], "X_umap_vae", vae_annotation_key, "VAE — cell type")
    _scatter_annotated(adata, axes[0, 1], "X_umap_pca", pca_annotation_key, "PCA — cell type")

    fig.suptitle("Predicted cell-type annotations", fontsize=13, y=1.01)
    plt.tight_layout()

    fpath = os.path.join(out_dir, "annotated_umap.png")
    fig.savefig(fpath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[dge] Saved annotated UMAP → '{fpath}'")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_cluster_dge(
    adata,
    out_dir: str,
    n_dge_genes: int = 50,
    n_dot_genes: int = 5,
    cell_type_markers: dict[str, list[str]] | None = None,
) -> dict:
    """
    Run the full DGE + cell-type annotation pipeline for both VAE and PCA
    Leiden clusterings and save all figures.

    Assumes :func:`~src.analysis.embedding_analysis.compare_vae_pca_clustering`
    has already been called so that ``leiden_vae``, ``leiden_pca``,
    ``X_umap_vae``, and ``X_umap_pca`` exist in *adata*.

    :param adata: AnnData object
    :param out_dir: experiment directory; figures are saved here
    :param n_dge_genes: genes per cluster retained by DGE.  Default is 50
    :param n_dot_genes: genes per cluster shown in the dot plot.  Default is 5
    :param cell_type_markers: custom marker dict; defaults to
        :data:`GUT_CELL_TYPE_MARKERS`
    :return: dict with keys ``'dge_vae_key'``, ``'dge_pca_key'``,
        ``'vae_annotation_key'``, ``'pca_annotation_key'``,
        ``'vae_score_matrix'``, ``'pca_score_matrix'``
    """
    os.makedirs(out_dir, exist_ok=True)

    results: dict = {}

    for rep, groupby_key in [("vae", "leiden_vae"), ("pca", "leiden_pca")]:
        if groupby_key not in adata.obs.columns:
            print(f"[dge] '{groupby_key}' not found in adata.obs — skipping {rep.upper()}")
            continue

        print(f"\n[dge] ── {rep.upper()} clustering ──────────────────────────────")

        # 1. DGE
        dge_key = run_dge(adata, groupby_key, n_genes=n_dge_genes)
        results[f"dge_{rep}_key"] = dge_key

        # 2. Save full DGE results table
        save_dge_table(adata, dge_key, out_dir, label=rep)

        # 3. Dot plot of top markers
        plot_dge_dotplot(adata, dge_key, groupby_key, out_dir,
                         n_genes=n_dot_genes, label=rep)

        # 4. Cell-type scoring + cluster annotation
        annotation_key, score_matrix = score_and_annotate_clusters(
            adata, groupby_key,
            cell_type_markers=cell_type_markers,
            score_key_prefix=f"ct_score_{rep}",
        )
        results[f"{rep}_annotation_key"] = annotation_key
        results[f"{rep}_score_matrix"] = score_matrix

        # 4. Cell-type score heatmap
        plot_cell_type_scores(score_matrix, out_dir, label=rep)

    # 5. Combined annotated UMAP (both representations side by side)
    if "vae_annotation_key" in results and "pca_annotation_key" in results:
        plot_annotated_umap(
            adata,
            out_dir,
            vae_annotation_key=results["vae_annotation_key"],
            pca_annotation_key=results["pca_annotation_key"],
        )

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_top_genes(adata, dge_key: str, n_genes: int) -> dict[str, list[str]]:
    """Pull top n genes per group from rank_genes_groups results."""
    result = adata.uns[dge_key]
    groups = result["names"].dtype.names
    top_genes: dict[str, list[str]] = {}
    for grp in groups:
        genes = [g for g in result["names"][grp][:n_genes] if g in adata.var_names]
        if genes:
            top_genes[grp] = genes
    return top_genes


def _scatter_annotated(adata, ax, umap_key: str, color_key: str, title: str) -> None:
    """Scatter one UMAP panel coloured by a categorical obs column."""
    if umap_key not in adata.obsm:
        ax.set_visible(False)
        return

    coords = adata.obsm[umap_key]
    values = adata.obs[color_key]

    categories = list(values.cat.categories) if hasattr(values, "cat") \
        else sorted(values.unique())

    # Build a palette — prefer scanpy's default_102, fall back to tab20
    try:
        palette = list(sc.pl.palettes.default_102)
    except AttributeError:
        palette = list(sc.pl.palettes.zeileis_28)
    palette = [palette[i % len(palette)] for i in range(len(categories))]

    for i, cat in enumerate(categories):
        mask = (values == cat).values
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=2, alpha=0.5, linewidths=0,
            color=palette[i], label=str(cat),
            rasterized=True,
        )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)

    if len(categories) <= 25:
        ax.legend(
            markerscale=5, fontsize=7, loc="best",
            framealpha=0.4, title=color_key,
            title_fontsize=8, handletextpad=0.3,
        )
