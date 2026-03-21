# Gut Marker Analysis

A pipeline for extracting marker genes from single-cell RNA-seq (scRNA-seq) data using a Variational Autoencoder (VAE). Developed for analyzing human gut tissue datasets (fetal, pediatric, and adult).  Using data from Elmentiate et al., 2021 Cells of the human intestinal tract mapped across space and time

## Overview

The core idea is to train a VAE on gene expression data, then probe the decoder to identify which genes each latent dimension encodes. This "decoder probing" approach accounts for non-linearities in the network, yielding more accurate marker gene rankings than simple weight matrix inspection.

## Project Structure

```
Gut_Marker_Analysis/
├── config.py                    # Project-wide path configuration
├── scripts/
│   └── run_pipeline.py          # Main CLI entry point
├── src/
│   ├── models/vae.py            # VAE architecture
│   ├── training/train_vae.py    # Training loop and cell embedding
│   ├── preprocessing/
│   │   ├── preprocessing.py     # QC filtering, normalization, HVG selection
│   │   └── gene_lists.py        # Sex and cell cycle gene lists
│   ├── analysis/marker_genes.py # Decoder probing for marker extraction
│   ├── data/anndata_dataset.py  # PyTorch Dataset wrapper for AnnData
│   └── utils/utility_functions.py
├── data/
│   └── elmentiate_2021/         # Input H5AD datasets
├── experiments/                 # Per-run outputs (timestamped)
└── figures/
```

## Pipeline Steps

1. **Load dataset** — fetal, pediatric, or full adult gut
2. **Filter & normalize** — QC filtering by gene count and mitochondrial %, log-normalization
3. **HVG selection** — Select highly variable genes (default: 3000)
4. **Confounder regression** — Remove sex and optionally cell cycle effects
5. **Train VAE** — Fit model on regressed expression layer
6. **Embed cells** — Encode cells into latent space
7. **Extract markers** — Probe decoder to rank genes per latent dimension

## Usage

```bash
python scripts/run_pipeline.py --dataset fetal
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `fetal` | Dataset: `fetal`, `pediatric`, or `full` |
| `--n_hvg` | `3000` | Number of highly variable genes |
| `--latent_dim` | `32` | VAE latent space dimensionality |
| `--hidden_dims` | `512 256` | Encoder/decoder hidden layer sizes |
| `--epochs` | `200` | Max training epochs |
| `--batch_size` | `512` | Training batch size |
| `--lr` | `1e-3` | Learning rate |
| `--beta` | `1.0` | KL divergence weight (>1 for beta-VAE) |
| `--top_k` | `50` | Top genes to extract per latent dimension |
| `--probe_scale` | `3.0` | Scale factor for decoder probing |
| `--min_genes` | `200` | Min genes per cell (QC filter) |
| `--max_genes` | `6000` | Max genes per cell (QC filter) |
| `--max_mito` | `20` | Max mitochondrial % (QC filter) |

## Architecture

**VAE**: `n_genes → 512 → 256 → [μ, σ²]` (latent dim 32) `→ 256 → 512 → n_genes`

- Reconstruction loss: MSE
- Regularization: KL divergence (beta-weighted)
- Training: Adam optimizer with ReduceLROnPlateau scheduler and early stopping
- Device: auto-detects MPS (Apple Silicon), CUDA, or CPU

## Outputs

Each run is saved to `experiments/<dataset>_<timestamp>/`:

| File | Description |
|---|---|
| `vae_best.pt` | Best model weights |
| `latent_embeddings.npy` | Cell embeddings (n_cells × latent_dim) |
| `marker_genes.csv` | Full ranked marker list (latent_dim, rank, gene, score) |
| `marker_summary.csv` | Top 10 genes per latent dimension (wide format) |
| `history.pkl` | Training loss curves |
| `run_config.json` | All pipeline parameters |
| `metadata.json` | Model architecture details |

## Dependencies

- `torch` — VAE training
- `scanpy` — Single-cell analysis and AnnData I/O
- `numpy`, `pandas` — Numerical computing
- `scipy` — Sparse matrix handling

## Data

Input data should be in H5AD format and placed in `data/elmentiate_2021/`. The pipeline expects raw counts.

Dataset files (from Elmentaite et al. 2021):
- `fetal_RAWCOUNTS_cellxgene.h5ad`
- `pediatric_RAWCOUNTS_cellxgene_c.h5ad`
- `Full_obj_raw_counts_nosoupx_v2.h5ad`
