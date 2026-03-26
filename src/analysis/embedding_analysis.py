"""
Post-training embedding analysis.

Provides two main entry points:

1. ``load_vae_embeddings`` — load a saved .npy latent embedding array into
   ``adata.obsm['X_vae']`` so downstream analysis can reuse a completed run
   without re-running training.

2. ``compare_vae_pca_clustering`` — run Leiden clustering independently on the
   VAE latent space and on PCA, compute a UMAP layout for each, and produce a
   side-by-side comparison figure.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_vae_embeddings(adata, embeddings_path: str) -> None:
    """
    Load pre-computed VAE latent embeddings from a ``.npy`` file and store
    them in ``adata.obsm['X_vae']`` in-place.

    :param adata: AnnData object to update — must have the same number of
        cells as rows in the embedding array
    :param embeddings_path: path to a ``.npy`` file containing an array of
        shape ``(n_cells, latent_dim)``
    """
    Z = np.load(embeddings_path)
    if Z.shape[0] != adata.n_obs:
        raise ValueError(
            f"Embedding array has {Z.shape[0]} rows but adata has {adata.n_obs} cells."
        )
    adata.obsm["X_vae"] = Z
    print(f"Loaded VAE embeddings from '{embeddings_path}'  shape={Z.shape}")


def compare_vae_pca_clustering(
    adata,
    out_dir: str | None = None,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    leiden_resolution: float = 0.5,
    umap_min_dist: float = 0.3,
    color_by: str | list[str] | None = None,
    random_state: int = 42,
) -> dict:
    """
    Compare Leiden clustering on the VAE latent space vs PCA, visualised
    with UMAP.

    For each representation the function:

    1. Builds a k-nearest-neighbour graph.
    2. Runs Leiden community detection.
    3. Computes a 2-D UMAP layout.

    Results are written back into *adata* and a comparison figure is saved
    when *out_dir* is provided.

    New adata keys created
    ----------------------
    ``obsm['X_umap_vae']``      — UMAP coordinates from the VAE graph
    ``obsm['X_umap_pca']``      — UMAP coordinates from the PCA graph
    ``obs['leiden_vae']``       — Leiden labels from the VAE graph
    ``obs['leiden_pca']``       — Leiden labels from the PCA graph
    ``obsp['vae_neighbors_*']`` — sparse connectivity / distance matrices

    :param adata: AnnData object with ``adata.obsm['X_vae']`` already set
        (via :func:`load_vae_embeddings` or :func:`~src.training.train_vae.embed_adata`)
    :param out_dir: directory to save the figure.  If *None* the figure is
        displayed but not saved.  Default is *None*
    :param n_pcs: number of principal components for PCA.  Default is 50
    :param n_neighbors: k for the kNN graphs.  Default is 15
    :param leiden_resolution: Leiden resolution parameter — higher values
        produce more clusters.  Default is 0.5
    :param umap_min_dist: UMAP ``min_dist`` parameter.  Default is 0.3
    :param color_by: additional ``obs`` column name(s) to include as extra
        panels in the figure (e.g. ``'cell_type'`` or
        ``['cell_type', 'sample']``).  Default is *None*
    :param random_state: random seed for PCA, UMAP, and Leiden.  Default is 42
    :return: dict with keys ``'vae_clusters'`` and ``'pca_clusters'`` holding
        the Leiden label arrays for each representation
    """
    if "X_vae" not in adata.obsm:
        raise ValueError(
            "adata.obsm['X_vae'] not found.  "
            "Call embed_adata() or load_vae_embeddings() first."
        )

    # --- PCA (recompute so n_pcs is controlled here) ---
    print(f"[embedding_analysis] Computing PCA ({n_pcs} PCs) ...")
    sc.pp.pca(adata, n_comps=n_pcs, random_state=random_state)

    # ------------------------------------------------------------------ VAE --
    print(f"[embedding_analysis] Building kNN graph on VAE embeddings (k={n_neighbors}) ...")
    sc.pp.neighbors(
        adata,
        use_rep="X_vae",
        n_neighbors=n_neighbors,
        key_added="vae_neighbors",
        random_state=random_state,
    )
    print(f"[embedding_analysis] Leiden clustering on VAE graph "
          f"(resolution={leiden_resolution}) ...")
    sc.tl.leiden(
        adata,
        resolution=leiden_resolution,
        neighbors_key="vae_neighbors",
        key_added="leiden_vae",
        random_state=random_state,
    )
    print("[embedding_analysis] Computing UMAP on VAE graph ...")
    sc.tl.umap(
        adata,
        neighbors_key="vae_neighbors",
        min_dist=umap_min_dist,
        random_state=random_state,
    )
    adata.obsm["X_umap_vae"] = adata.obsm["X_umap"].copy()

    # ------------------------------------------------------------------ PCA --
    print(f"[embedding_analysis] Building kNN graph on PCA embeddings (k={n_neighbors}) ...")
    sc.pp.neighbors(
        adata,
        use_rep="X_pca",
        n_neighbors=n_neighbors,
        key_added="pca_neighbors",
        random_state=random_state,
    )
    print(f"[embedding_analysis] Leiden clustering on PCA graph "
          f"(resolution={leiden_resolution}) ...")
    sc.tl.leiden(
        adata,
        resolution=leiden_resolution,
        neighbors_key="pca_neighbors",
        key_added="leiden_pca",
        random_state=random_state,
    )
    print("[embedding_analysis] Computing UMAP on PCA graph ...")
    sc.tl.umap(
        adata,
        neighbors_key="pca_neighbors",
        min_dist=umap_min_dist,
        random_state=random_state,
    )
    adata.obsm["X_umap_pca"] = adata.obsm["X_umap"].copy()

    n_vae = adata.obs["leiden_vae"].nunique()
    n_pca = adata.obs["leiden_pca"].nunique()
    print(f"[embedding_analysis] VAE → {n_vae} Leiden clusters")
    print(f"[embedding_analysis] PCA → {n_pca} Leiden clusters")

    # ---------------------------------------------------------------- Figure --
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), squeeze=False
    )

    _draw_panel(adata, axes[0, 0], "X_umap_vae", "leiden_vae", row_label="VAE")
    _draw_panel(adata, axes[0, 1], "X_umap_pca", "leiden_pca", row_label="PCA")

    fig.suptitle(
        f"Leiden clustering: VAE ({n_vae} clusters) vs PCA ({n_pca} clusters)\n"
        f"k={n_neighbors}  resolution={leiden_resolution}  n_pcs={n_pcs}",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(out_dir, "clustering_comparison.png")
        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        print(f"[embedding_analysis] Saved figure → '{fig_path}'")

    plt.show()

    return {
        "vae_clusters": adata.obs["leiden_vae"].values,
        "pca_clusters": adata.obs["leiden_pca"].values,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _draw_panel(adata, ax, umap_key: str, color_key: str, row_label: str) -> None:
    """Render a single UMAP panel coloured by one obs key."""
    coords = adata.obsm[umap_key]
    values = adata.obs[color_key]

    if hasattr(values, "cat"):
        categories = list(values.cat.categories)
    else:
        categories = sorted(values.unique(), key=lambda x: (isinstance(x, str), x))

    palette = _get_palette(len(categories))

    for i, cat in enumerate(categories):
        mask = (values == cat).values
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=2,
            alpha=0.5,
            label=str(cat),
            color=palette[i],
            linewidths=0,
            rasterized=True,
        )

    ax.set_title(f"{row_label} — {color_key}", fontsize=11)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)

    if len(categories) <= 30:
        legend = ax.legend(
            markerscale=5,
            fontsize=7,
            loc="best",
            framealpha=0.4,
            title=color_key,
            title_fontsize=8,
            handletextpad=0.3,
            borderpad=0.4,
        )
        legend.get_frame().set_linewidth(0.5)


def _get_palette(n: int) -> list[str]:
    """Return a list of *n* hex colour strings."""
    try:
        # scanpy's extended palette covers up to 102 colours
        base = list(sc.pl.palettes.default_102)
    except AttributeError:
        base = list(sc.pl.palettes.zeileis_28)

    if n <= len(base):
        return base[:n]

    # Cycle if we need more colours than the palette provides
    return [base[i % len(base)] for i in range(n)]
