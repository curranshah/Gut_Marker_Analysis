"""
Train a VAE on a preprocessed AnnData object.

Typical usage
-------------
    from src.training.train_vae import train_vae
    from src.preprocessing.preprocessing import filter_anndata, identify_hvg

    adata = filter_anndata('data/fetal_RAWCOUNTS_cellxgene.h5ad')
    adata = identify_hvg(adata, n_top_genes=3000, filter_nonHVG=True)

    model, history = train_vae(adata, use_hvg=True)
"""

import os
import json
import time
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.vae import VAE, vae_loss
from src.data.anndata_dataset import make_dataloaders
from src.utils.utility_functions import specify_device


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Architecture
    "latent_dim": 32,
    "hidden_dims": [512, 256],
    "dropout": 0.1,

    # Training
    "batch_size": 512,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "n_epochs": 200,
    "beta": 1.0,            # KL weight (1.0 = standard VAE; >1 = beta-VAE)

    # Scheduler (ReduceLROnPlateau)
    "scheduler_mode": "min",
    "scheduler_factor": 0.5,
    "scheduler_patience": 10,
    "min_lr": 1e-6,

    # Early stopping
    "early_stopping_patience": 20,

    # Data
    "use_hvg": True,
    "val_fraction": 0.1,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_vae(adata, config: dict | None = None, save_dir: str | None = None,
              layer: str | None = None):
    """
    Train a VAE on log-normalised scRNA-seq data from an AnnData object

    :param adata: AnnData object after filter_anndata() and optionally identify_hvg(),
    containing log-normalized expression values
    :param config: dictionary of hyperparameters to override DEFAULT_CONFIG values.
    Any keys not provided fall back to the defaults.  Default is None
    :param save_dir: directory path to save the best model weights, metadata.json, and
    training history.  If None, nothing is saved to disk.  Default is None
    :param layer: if set, read expression from ``adata.layers[layer]`` instead of
    ``adata.X``.  Default is None (use ``adata.X``)
    :return: tuple of (model, history) where model is the trained VAE in eval mode on CPU
    and history is a dictionary with lists 'train_loss', 'val_loss', 'recon_loss', 'kl_loss'
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    device = specify_device()
    print(f"Training on device: {device}")

    # --- Data ---
    train_loader, val_loader, dataset = make_dataloaders(
        adata,
        use_hvg=cfg["use_hvg"],
        batch_size=cfg["batch_size"],
        val_fraction=cfg["val_fraction"],
        seed=cfg["seed"],
        layer=layer,
    )
    n_genes = dataset.n_genes
    print(f"Input dimensions: {len(dataset)} cells × {n_genes} genes")

    # --- Model ---
    model = VAE(
        n_genes=n_genes,
        latent_dim=cfg["latent_dim"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=cfg["scheduler_mode"],
        factor=cfg["scheduler_factor"],
        patience=cfg["scheduler_patience"],
        min_lr=cfg["min_lr"],
    )

    # --- Training loop ---
    history = {"train_loss": [], "val_loss": [], "recon_loss": [], "kl_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, cfg["n_epochs"] + 1):
        t0 = time.time()

        # Train
        model.train()
        train_total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, _, _ = vae_loss(recon, batch, mu, logvar, beta=cfg["beta"])
            loss.backward()
            optimizer.step()
            train_total += loss.item() * batch.size(0)

        # Validate
        model.eval()
        val_total = recon_total = kl_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, logvar, beta=cfg["beta"])
                n = batch.size(0)
                val_total += loss.item() * n
                recon_total += recon_loss.item() * n
                kl_total += kl_loss.item() * n

        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        train_loss = train_total / n_train
        val_loss = val_total / n_val
        recon_loss_avg = recon_total / n_val
        kl_loss_avg = kl_total / n_val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["recon_loss"].append(recon_loss_avg)
        history["kl_loss"].append(kl_loss_avg)

        scheduler.step(val_loss)

        elapsed = time.time() - t0
        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:4d}/{cfg['n_epochs']} | "
                  f"train={train_loss:.4f} val={val_loss:.4f} "
                  f"recon={recon_loss_avg:.4f} kl={kl_loss_avg:.4f} "
                  f"lr={lr:.2e} [{elapsed:.1f}s]")

        # Early stopping + checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch} (no improvement for "
                      f"{cfg['early_stopping_patience']} epochs)")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu").eval()

    # --- Save ---
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "vae_best.pt")
        torch.save(model.state_dict(), model_path)

        metadata = {
            "model_class": "VAE",
            "model_file": "vae_best.pt",
            "n_genes": n_genes,
            "latent_dim": cfg["latent_dim"],
            "hidden_dims": cfg["hidden_dims"],
            "gene_names": dataset.gene_names,
            "best_val_loss": best_val_loss,
            "layer": layer,
            "config": cfg,
        }
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        import pickle
        with open(os.path.join(save_dir, "history.pkl"), "wb") as f:
            pickle.dump(history, f)

        print(f"Saved model and metadata to {save_dir}")

    print(f"Best validation loss: {best_val_loss:.4f}")
    return model, history


def embed_adata(model: VAE, adata, use_hvg: bool = True,
                batch_size: int = 1024, device: str | None = None,
                layer: str | None = None) -> np.ndarray:
    """
    Encode all cells in an AnnData object into the VAE latent space

    :param model: trained VAE model
    :param adata: AnnData object containing the same genes used during training
    :param use_hvg: if True, subset to highly variable genes before encoding.
    Must match the setting used during training.  Default is True
    :param batch_size: number of cells to encode per forward pass.  Default is 1024
    :param device: device string to run encoding on.  If None, uses specify_device()
    to select MPS, CUDA, or CPU automatically.  Default is None
    :param layer: if set, read expression from ``adata.layers[layer]`` instead of
    ``adata.X``.  Default is None (use ``adata.X``)
    :return: numpy array of shape (n_cells, latent_dim) containing the mean latent
    embeddings for all cells
    """
    import scipy.sparse as sp
    from torch.utils.data import DataLoader, TensorDataset

    if device is None:
        device = str(specify_device())

    if use_hvg and 'highly_variable' in adata.var.columns:
        adata_sub = adata[:, adata.var.highly_variable]
    else:
        adata_sub = adata

    X = adata_sub.layers[layer] if layer is not None else adata_sub.X
    if sp.issparse(X):
        X = X.toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
    model = model.to(device).eval()

    embeddings = []
    with torch.no_grad():
        for (batch,) in loader:
            mu, _ = model.encode(batch.to(device))
            embeddings.append(mu.cpu().numpy())

    return np.concatenate(embeddings, axis=0)
