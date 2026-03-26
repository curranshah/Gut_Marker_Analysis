import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Variational Autoencoder for single-cell RNA-seq expression data

    Encoder: n_genes → hidden_dims → (mu_32, logvar_32)
    Decoder: 32 → reversed(hidden_dims) → n_genes

    Marker genes are extracted post-training via decoder probing or
    effective weight decomposition (see src/analysis/marker_genes.py).
    """

    def __init__(self, n_genes: int, latent_dim: int = 32, hidden_dims: tuple = (512, 256),
                 dropout: float = 0.1):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim

        # --- Encoder ---
        encoder_layers = []
        in_dim = n_genes
        for h in hidden_dims:
            encoder_layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

        # --- Decoder ---
        decoder_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        # Softplus keeps output non-negative, matching log-normalized expression
        decoder_layers += [nn.Linear(in_dim, n_genes), nn.Softplus()]
        self.decoder = nn.Sequential(*decoder_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor):
        """
        Pass input through the encoder and return the latent mean and log-variance

        :param x: log-normalized gene expression tensor of shape (batch, n_genes)
        :return: tuple of (mu, logvar), each of shape (batch, latent_dim)
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample a latent vector using the reparameterization trick during training,
        or return the mean directly during evaluation

        :param mu: latent mean tensor of shape (batch, latent_dim)
        :param logvar: latent log-variance tensor of shape (batch, latent_dim)
        :return: sampled latent vector of shape (batch, latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Pass a latent vector through the decoder to reconstruct gene expression

        :param z: latent vector of shape (batch, latent_dim)
        :return: reconstructed gene expression tensor of shape (batch, n_genes)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """
        Full forward pass: encode, sample, decode

        :param x: log-normalized gene expression tensor of shape (batch, n_genes)
        :return: tuple of (recon, mu, logvar) where recon is the reconstructed
                 expression of shape (batch, n_genes) and mu, logvar are each
                 of shape (batch, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the mean latent embedding for a batch without sampling

        :param x: log-normalized gene expression tensor of shape (batch, n_genes)
        :return: latent mean tensor of shape (batch, latent_dim)
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu


def vae_loss(recon: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0):
    """
    Compute the VAE ELBO loss as reconstruction loss plus a weighted KL divergence term

    :param recon: reconstructed gene expression tensor of shape (batch, n_genes)
    :param x: input gene expression tensor of shape (batch, n_genes)
    :param mu: latent mean tensor of shape (batch, latent_dim)
    :param logvar: latent log-variance tensor of shape (batch, latent_dim)
    :param beta: weight applied to the KL divergence term.  beta=1 is the standard VAE;
    values greater than 1 encourage more disentangled latent representations.  Default is 1.0
    :return: tuple of (total_loss, recon_loss, kl_loss) as scalar tensors
    """
    recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss
