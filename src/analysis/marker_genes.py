"""
Extract marker genes from a trained VAE via decoder analysis.

Two methods are provided:

1. decoder_probing  (recommended)
   Passes unit vectors through the decoder to see which genes are activated
   by each latent dimension.  Exact — accounts for all non-linearities and
   batch norm, regardless of network depth.

2. effective_weight_matrix
   Computes W_out @ W_mid @ W_in  (linear approximation of the decoder).
   Fast and intuitive but ignores batch norm and ReLU.  Best used as a quick
   sanity-check or for shallow decoders.

Typical usage
-------------
    from src.analysis.marker_genes import decoder_probing, markers_to_dataframe

    markers = decoder_probing(model, gene_names, top_k=50)
    df = markers_to_dataframe(markers)
    df.to_csv('results/latent_markers.csv', index=False)
"""

import numpy as np
import pandas as pd
import torch

from src.models.vae import VAE


def decoder_probing(model: VAE, gene_names: list[str],
                    top_k: int = 50,
                    scale: float = 3.0) -> dict[str, list[tuple[str, float]]]:
    """
    Identify marker genes for each latent dimension by probing the decoder with
    scaled unit vectors

    For latent dimension j, sets z = scale * e_j (where e_j is the j-th standard
    basis vector) and measures the reconstructed gene expression above the zero-vector
    baseline.  Genes with the highest scores are the markers for that dimension.
    This method is exact and accounts for all non-linearities and batch normalization.

    :param model: trained VAE in eval mode
    :param gene_names: list of gene names of length n_genes, in the same column order
    as the expression matrix used for training
    :param top_k: number of top marker genes to return per latent dimension.  Default is 50
    :param scale: magnitude of the unit vector probe, roughly equivalent to the number
    of standard deviations from the prior.  Default is 3.0
    :return: dictionary mapping 'latent_N' to a list of (gene_name, score) tuples sorted
    in descending order by score, of length top_k
    """
    device = next(model.parameters()).device
    model.eval()

    latent_dim = model.latent_dim
    z_probe = torch.zeros(latent_dim, latent_dim, device=device)
    for i in range(latent_dim):
        z_probe[i, i] = scale

    with torch.no_grad():
        recon = model.decode(z_probe).cpu().numpy()  # (latent_dim, n_genes)

    with torch.no_grad():
        baseline = model.decode(torch.zeros(1, latent_dim, device=device)).cpu().numpy()[0]

    scores = recon - baseline  # (latent_dim, n_genes)

    markers = {}
    for dim in range(latent_dim):
        s = scores[dim]
        top_idx = np.argsort(s)[::-1][:top_k]
        markers[f"latent_{dim}"] = [(gene_names[i], float(s[i])) for i in top_idx]

    return markers


def effective_weight_matrix(model: VAE) -> np.ndarray:
    """
    Compute the linear approximation of the decoder as a matrix by chaining the
    weight matrices of all Linear layers in the decoder

    This ignores batch normalization and ReLU non-linearities so it is an
    approximation, but is fast and gives an interpretable picture of gene-latent
    associations.  For a more accurate result use decoder_probing instead.

    :param model: trained VAE
    :return: numpy array of shape (n_genes, latent_dim) representing the approximate
    linear mapping from the latent space to gene expression
    """
    linear_layers = []

    for module in model.decoder.modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append(module)

    with torch.no_grad():
        W = linear_layers[0].weight.cpu().numpy()  # (h1, latent_dim)
        for layer in linear_layers[1:]:
            W = layer.weight.cpu().numpy() @ W      # (h_{i+1}, latent_dim)

    return W


def top_genes_from_weight_matrix(W: np.ndarray, gene_names: list[str],
                                 top_k: int = 50,
                                 use_abs: bool = True) -> dict[str, list[tuple[str, float]]]:
    """
    Extract top marker genes per latent dimension from an effective weight matrix

    :param W: numpy array of shape (n_genes, latent_dim) as returned by
    effective_weight_matrix()
    :param gene_names: list of gene names of length n_genes
    :param top_k: number of top genes to return per latent dimension.  Default is 50
    :param use_abs: if True, rank genes by absolute weight value to capture both positive
    and negative associations.  If False, rank by raw weight value.  Default is True
    :return: dictionary mapping 'latent_N' to a list of (gene_name, weight) tuples sorted
    in descending order by score, of length top_k
    """
    scores = np.abs(W) if use_abs else W
    markers = {}
    for dim in range(W.shape[1]):
        s = scores[:, dim]
        top_idx = np.argsort(s)[::-1][:top_k]
        markers[f"latent_{dim}"] = [(gene_names[i], float(W[i, dim])) for i in top_idx]
    return markers


def markers_to_dataframe(markers: dict) -> pd.DataFrame:
    """
    Convert a markers dictionary to a tidy DataFrame

    :param markers: dictionary mapping latent dimension keys to lists of
    (gene_name, score) tuples as returned by decoder_probing() or
    top_genes_from_weight_matrix()
    :return: DataFrame with columns ['latent_dim', 'rank', 'gene', 'score']
    """
    rows = []
    for dim_key, gene_scores in markers.items():
        for rank, (gene, score) in enumerate(gene_scores, start=1):
            rows.append({"latent_dim": dim_key, "rank": rank,
                         "gene": gene, "score": score})
    return pd.DataFrame(rows)


def summarise_latent_dims(model: VAE, gene_names: list[str],
                          top_k: int = 10, scale: float = 3.0) -> pd.DataFrame:
    """
    Build a summary table showing the top marker genes for each latent dimension
    side-by-side

    :param model: trained VAE in eval mode
    :param gene_names: list of gene names of length n_genes, in the same column order
    as the expression matrix used for training
    :param top_k: number of top marker genes to show per latent dimension.  Default is 10
    :param scale: magnitude of the unit vector probe passed to decoder_probing().  Default is 3.0
    :return: DataFrame with one column per latent dimension (latent_0, latent_1, ...)
    containing the top_k gene names, useful for printing to a notebook
    """
    markers = decoder_probing(model, gene_names, top_k=top_k, scale=scale)
    data = {}
    for dim_key, gene_scores in markers.items():
        data[dim_key] = [g for g, _ in gene_scores]
    return pd.DataFrame(data)
