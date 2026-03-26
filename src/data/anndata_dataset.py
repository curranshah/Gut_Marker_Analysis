import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class AnnDataDataset(Dataset):
    """
    PyTorch Dataset wrapping an AnnData expression matrix

    Converts adata.X (dense or sparse, float32/64) to a float32 tensor.
    Optionally subsets to highly variable genes before exposing data.
    """

    def __init__(self, adata, use_hvg: bool = True, layer: str | None = None):
        """
        Initialize the dataset from a preprocessed AnnData object

        :param adata: AnnData object (cells x genes) with log-normalized expression values
        :param use_hvg: if True and adata.var['highly_variable'] exists, subset columns to
        highly variable genes only.  Default is True
        :param layer: if set, read expression from ``adata.layers[layer]`` instead of
        ``adata.X``.  Default is None (use ``adata.X``)
        """
        if use_hvg and 'highly_variable' in adata.var.columns:
            adata = adata[:, adata.var.highly_variable]

        X = adata.layers[layer] if layer is not None else adata.X
        if sp.issparse(X):
            X = X.toarray()

        self.data = torch.tensor(X, dtype=torch.float32)
        self.gene_names: list[str] = adata.var_names.tolist()
        self.n_genes: int = self.data.shape[1]

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def make_dataloaders(adata, use_hvg: bool = True, batch_size: int = 512,
                     val_fraction: float = 0.1, num_workers: int = 0,
                     seed: int = 42, layer: str | None = None):
    """
    Build train and validation DataLoaders from a preprocessed AnnData object

    :param adata: AnnData object (cells x genes) with log-normalized expression values
    :param use_hvg: if True, subset to highly variable genes before creating the dataset.
    Default is True
    :param batch_size: number of cells per mini-batch.  Default is 512
    :param val_fraction: fraction of cells held out for validation.  Default is 0.1
    :param num_workers: number of DataLoader worker processes.  Keep at 0 when training
    on MPS to avoid multiprocessing issues.  Default is 0
    :param seed: random seed for reproducible train/val split.  Default is 42
    :param layer: if set, read expression from ``adata.layers[layer]`` instead of
    ``adata.X``.  Default is None (use ``adata.X``)
    :return: tuple of (train_loader, val_loader, dataset) where dataset is the full
    AnnDataDataset containing gene_names and n_genes attributes
    """
    dataset = AnnDataDataset(adata, use_hvg=use_hvg, layer=layer)

    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, dataset
