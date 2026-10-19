"""Shared encoder, dataset, and latent alignment terms used by scLAMP-SR."""

import scipy
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kld


class encoder(nn.Module):
    def __init__(self, rna_dim, atac_dim, rna_pro_dim, atac_pro_dim, hidden_dim=512, latent_dim=128, dropout_rate=0.2):
        super().__init__()
        self.rna_encoder = nn.Sequential(
            nn.Linear(rna_pro_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.atac_encoder = nn.Sequential(
            nn.Linear(atac_pro_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.rna_mu = nn.Linear(hidden_dim // 4, latent_dim)
        self.rna_var = nn.Linear(hidden_dim // 4, latent_dim)
        self.atac_mu = nn.Linear(hidden_dim // 4, latent_dim)
        self.atac_var = nn.Linear(hidden_dim // 4, latent_dim)

        self.shared_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.shared_var = nn.Linear(hidden_dim // 2, latent_dim)

    def reparameterize(self, mu, var):
        z = Normal(mu, var.sqrt()).rsample()
        return z

    def forward(self, rna_data, atac_data):
        rna_h = self.rna_encoder(rna_data)
        atac_h = self.atac_encoder(atac_data)

        rna_shared = self.shared_layer(rna_h)
        atac_shared = self.shared_layer(atac_h)

        rna_mu = torch.clamp(self.rna_mu(rna_shared), min=-10, max=10)
        rna_var = torch.clamp(torch.exp(self.rna_var(rna_shared)), min=1e-6, max=1e8)
        z_rna = self.reparameterize(rna_mu, rna_var)

        atac_mu = torch.clamp(self.atac_mu(atac_shared), min=-10, max=10)
        atac_var = torch.clamp(torch.exp(self.atac_var(atac_shared)), min=1e-6, max=1e8)
        z_atac = self.reparameterize(atac_mu, atac_var)

        fused_h = (rna_h + atac_h) / 2
        shared_mu = torch.clamp(self.shared_mu(fused_h), min=-10, max=10)
        shared_var = torch.clamp(torch.exp(self.shared_var(fused_h)), min=1e-6, max=1e8)
        z_shared = self.reparameterize(shared_mu, shared_var)

        return z_rna, z_atac, z_shared, rna_mu, rna_var, atac_mu, atac_var, shared_mu, shared_var


def compute_mmd(x, y, kernel="rbf", sigma=5.0):
    def gaussian_kernel(a, b, sigma):
        dist = ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(2)
        return torch.exp(-dist / (2 * sigma ** 2))

    K_xx = gaussian_kernel(x, x, sigma)
    K_yy = gaussian_kernel(y, y, sigma)
    K_xy = gaussian_kernel(x, y, sigma)
    return (K_xx + K_yy - 2 * K_xy).sum()


def compute_kld(z_mu, z_var, rna_mu, rna_var, atac_mu, atac_var):
    kl_div_z = kld(
        Normal(z_mu, torch.sqrt(z_var)),
        Normal(0, 1),
    ).sum(dim=1)

    kl_div_rna = kld(
        Normal(rna_mu, torch.sqrt(rna_var)),
        Normal(0, 1),
    ).sum(dim=1)

    kl_div_atac = kld(
        Normal(atac_mu, torch.sqrt(atac_var)),
        Normal(0, 1),
    ).sum(dim=1)

    return kl_div_z.mean()


class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, rna_adata, atac_adata):
        self.rna_adata = rna_adata
        self.atac_adata = atac_adata
        self.rna_data = self.rna_adata.X
        self.atac_data = self.atac_adata.X
        self.rna_raw = self.rna_adata.raw.X
        self.atac_raw = self.atac_adata.raw.X

        if scipy.sparse.issparse(self.rna_data):
            self.rna_data = self.rna_data.toarray()
        if scipy.sparse.issparse(self.atac_data):
            self.atac_data = self.atac_data.toarray()
        if scipy.sparse.issparse(self.rna_raw):
            self.rna_raw = self.rna_raw.toarray()
        if scipy.sparse.issparse(self.atac_raw):
            self.atac_raw = self.atac_raw.toarray()

    def __len__(self):
        return len(self.rna_adata)

    def __getitem__(self, idx):
        rna_sample = self.rna_data[idx]
        atac_sample = self.atac_data[idx]
        rna_raw_sample = self.rna_raw[idx]
        atac_raw_sample = self.atac_raw[idx]
        return (
            torch.FloatTensor(rna_sample),
            torch.FloatTensor(atac_sample),
            torch.FloatTensor(rna_raw_sample),
            torch.FloatTensor(atac_raw_sample),
        )
