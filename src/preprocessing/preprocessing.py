import scanpy as sc
import scanpy.external as sce
import numpy as np
import pandas as pd
import scipy.sparse as sp
from src.utils.utility_functions import load_data
from src.preprocessing.gene_lists import SEX_GENES, S_GENES, G2M_GENES


file_path = ('data/elmentiate_2021/fetal_RAWCOUNTS_cellxgene.h5ad')

def filter_anndata(file_path, min_cells=3, min_genes=200, max_genes=6000, max_mito=20,
                             normalization_target=1e6, filter_noncoding=True, save_qc_metrics=False):
    """
    Filter and normalize anndata object

    :param file_path: filtered_feature_bc_matrix/.h5ad file path of the scRNA-seq sample.
    :param min_cells: filteration parameter specifying the minimum number of cells a gene must appear in to be
    included in the analysis.  Default parameter is 3
    :param min_genes: filteration parameter specifying the minimum number of genes a cell must include to be included
    in the analysis.  Default parameter is 200
    :param max_genes: filteration parameter specifying the maximum number of genes a cell can include to be included
    in the analysis (helps to remove potential doublets).  Default parameter is 6000
    :param max_mito: filteration parameter specifying the maximum percentage of mitochondrial genes a cell can have
    expressed to be included in the analysis (helps to remove dead cells).  Default parameter is 20
    :param normalization_target: parameter specifying the rna expression count to normalize all cells to.  Default is
    1e6
    :param save_qc_metrics: boolean value to save violin plot of quality control metrics ['n_genes_by_counts',
    'total_counts','pct_counts_mt'].  Default is False
    :return: a filtered and log normalized anndata object
    """

    adata = load_data(file_path)

    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # save basic violin plot of pre-filtered qc metrics including n_genes_by_counts, total_Counts, and pct_counts_mt
    if save_qc_metrics:
        sc.pl.violin(adata=adata, keys=['n_genes_by_counts','total_counts','pct_counts_mt'], jitter=0.4,
                     multi_panel=True,size=1, show=False,
                     save=f"_violin_qc_metrics.png")

    # filter out genes that appear in less than 3 cells
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Only include cells where the number of genes is less than 6,000 to exclude potential doublets
    adata = adata[adata.obs.n_genes_by_counts < max_genes, :]

    # Only include cells where the number of genes is more than 200
    adata = adata[adata.obs.n_genes_by_counts > min_genes, :]

    # Only include cells with less than 20% mitochondrial genes to exclude potential dead cells
    adata = adata[adata.obs.pct_counts_mt < max_mito, :]

    if filter_noncoding:
        mask = ~adata.var_names.str.contains(r'\.')
        adata = adata[:, mask].copy()

    # Save raw anndata before doing normalization
    adata.raw = adata.copy()

    # Normalize the data to 1,000,000 reads per cell and then log normalize the data
    sc.pp.normalize_total(adata, target_sum=normalization_target)
    sc.pp.log1p(adata)

    return adata

def identify_hvg(adata, n_top_genes=None, min_mean=0.0125, max_mean=3, min_disp=0.5, filter_nonHVG=False, save_metrics=False):
    """
    Take a filtered and log normalized anndata object and select only the highly variable genes (HVGs)


    :param adata: anndata object of a gene count x cell matrix
    :param n_top_genes: number of the top highly variable genes to keep.  Default is None
    :param min_mean: min
    :param max_mean:
    :param min_disp:
    :param filter_nonHVG: Boolean statement to filter anndata object to only include HVGs
    :param save_metrics: Boolean value on whether to save a dispersion vs expression graph. Default is False
    :return: anndata object of gene count x cell matrix of highly variable genes
    """

    # identify highly variable genes
    sc.pp.highly_variable_genes(adata=adata, n_top_genes=n_top_genes, min_mean=min_mean, max_mean=max_mean,
                                min_disp=min_disp)  ## This function expects logarithmized data
    #
    # plot dispersion vs expression
    if save_metrics:
        sc.pl.highly_variable_genes(adata, log=False, show=False,
                                    save=f"_dispersion_vs_expression.png")

    # filter to only include HVGs
    if filter_nonHVG:
        adata = adata[:, adata.var.highly_variable]

    return adata

def regress_confounders(adata, regress_sex: bool = True, regress_cell_cycle: bool = False):
    """
    Regress biological confounders out of the expression matrix and store
    residuals in ``adata.layers['regressed']``.

    :param adata: AnnData object with log-normalized expression in ``adata.X``
    :param regress_sex: if True, compute a sex score from Y-chromosome /
        X-inactivation genes and regress it out.  Default is True
    :param regress_cell_cycle: if True, score S-phase and G2/M-phase genes and
        regress both scores out.  Default is False
    :return: adata with ``adata.layers['regressed']`` populated
    """
    # Copy expression into the new layer (dense float32)
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    adata.layers['regressed'] = X.astype(np.float32)

    keys = []

    if regress_sex:
        present = [g for g in SEX_GENES if g in adata.var_names]
        if present:
            sc.tl.score_genes(adata, gene_list=present, score_name='sex_score')
            keys.append('sex_score')
        else:
            print("      [regress_confounders] Warning: no sex genes found in var_names; skipping sex regression")

    if regress_cell_cycle:
        s_present   = [g for g in S_GENES   if g in adata.var_names]
        g2m_present = [g for g in G2M_GENES if g in adata.var_names]
        if s_present and g2m_present:
            sc.tl.score_genes_cell_cycle(adata, s_genes=s_present, g2m_genes=g2m_present)
            keys.extend(['S_score', 'G2M_score'])
        else:
            print("      [regress_confounders] Warning: insufficient cell cycle genes found; skipping cell cycle regression")

    if keys:
        sc.pp.regress_out(adata, keys=keys, layer='regressed')

    return adata


def compute_pca_and_clustering(adata, regression=True, harmony=True, max_value_clipping=10, pca_solver='arpack',
                n_neighbors=10, n_pcs=30, leiden_clustering=True, clustering_resolution=0.25, save_plots=False):
    """
    Compute the principal component analysis, umap, and perform leiden clustering
    :param adata: anndata object with log normalized data for gene counts x cells
    :param regression: Boolean value on whether the batch effects should be regressed out.  Default value is True
    :param harmony: Boolean value on whether harmony batch correction should be run. Defualt value is True
    :param max_value_clipping: Scale the unit variance and clip values exceding the threshold.  Default is 10
    :param pca_solver: the pca solver to use. Default is 'arpack'
    :param n_neighbors: number of neighbors to calculate the neighborhood graph for. Default is 10
    :param n_pcs: number of principal components to compute. Default is 30
    :param leiden_clustering: Boolean on whether to perform leiden clustering. Default is True
    :param clustering_resolution: resolution for leiden clustering.  Higher values find more clusters
    :param save_plots: Boolean value on whether to save a umap plot of the batches and leiden clusters
    :return: anndata object of gene count x cells matrix with umap coordinates and leiden cluster lablels
    """

    # Create layer in adata object for scaling
    adata.layers['scaled'] = adata.X.copy() #.copy() because not a sparse matrix

    # regress out effects of total counts per cell and percent mitochondrial counts
    # NOTE THIS IS A CONTROVERSIAL STEP AND NOT ALWAYS ACCURATE TO DO
    if regression:
        sc.pp.regress_out(adata=adata, keys=['total_counts', 'pct_counts_mt'], layer='scaled')

    # scale to unit variance for PCA, clip values exceeding var 10
    sc.pp.scale(adata, max_value=max_value_clipping, layer='scaled') #have to save in separate layer so as not to disrupt the log-normalized data

    # reduce the dimensionality of the data using PCA
    # sc.tl.pca(adata, svd_solver=pca_solver)
    sc.pp.pca(adata, layer='scaled', svd_solver=pca_solver) #allows for specifying the layer

    # run harmony to adjust PCs
    if harmony:
        sce.pp.harmony_integrate(adata=adata, key='batch')
        adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']

        # computing the neighborhood graph
    sc.pp.neighbors(adata=adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    # compute UMAP coordinates
    sc.tl.umap(adata=adata)

    if leiden_clustering:
        # clustering the nieghborhood graph using leiden clustering
        sc.tl.leiden(adata=adata, resolution=clustering_resolution)

    if save_plots:
        sc.pl.umap(adata=adata, color=['batch', 'leiden'], show=False,
                   save=f"_umap.png")

    return adata

