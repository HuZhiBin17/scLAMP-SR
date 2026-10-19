"""Held-out count-likelihood ablation and frozen-latent decoder calibration.

The default ``train_fitted`` mode fits every transformation on training cells.
The ``recommended_checkpoint`` mode exactly reproduces the preprocessing used
by the current full-data integration checkpoint so that its latent space can be
loaded and frozen while new decoder terms are calibrated on the training split.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

from sclamp_sr import model as state_sr  # noqa: E402
from sclamp_sr import zi_decoder as gentle_zi  # noqa: E402
from sclamp_sr.losses import true_mixture_loss_function  # noqa: E402
from sclamp_sr.utils import setup_seed  # noqa: E402


METHODS = ("fixed", "global", "feature", "continuous_prior", "state_resolved")
METHOD_OUTPUT_NAMES = {
    "fixed": "固定权重",
    "global": "全局权重",
    "feature": "特征级权重",
    "continuous_prior": "连续先验",
    "state_resolved": "状态分辨模型",
}


def dense_float(matrix):
    if sp.issparse(matrix):
        return matrix.toarray().astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32)


def vector(value):
    return np.asarray(value).reshape(-1)


def normalize_log1p(matrix, target_sum=1e4):
    matrix = sp.csr_matrix(matrix, dtype=np.float32).copy()
    totals = vector(matrix.sum(axis=1))
    scale = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
    matrix = sp.diags(scale).dot(matrix).tocsr()
    matrix.data = np.log1p(matrix.data)
    return matrix


def stratified_split(labels, seed, train_fraction=0.70, validation_fraction=0.15):
    """Deterministic within-label split that also works for rare labels."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=str)
    train, valid, test = [], [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        n = len(indices)
        n_train = max(1, int(round(n * train_fraction)))
        n_valid = int(round(n * validation_fraction))
        if n >= 3:
            n_valid = max(1, n_valid)
        n_valid = min(n_valid, max(0, n - n_train - 1))
        n_train = min(n_train, n - n_valid)
        train.extend(indices[:n_train])
        valid.extend(indices[n_train : n_train + n_valid])
        test.extend(indices[n_train + n_valid :])
    return tuple(np.sort(np.asarray(x, dtype=np.int64)) for x in (train, valid, test))


def load_or_create_split(rna, split_path, seed):
    split_path.parent.mkdir(parents=True, exist_ok=True)
    if split_path.exists():
        saved = np.load(split_path, allow_pickle=True)
        return saved["train"], saved["valid"], saved["test"]
    labels = rna.obs["cell_type"].astype(str).to_numpy() if "cell_type" in rna.obs else np.repeat("all", rna.n_obs)
    train, valid, test = stratified_split(labels, seed)
    np.savez_compressed(split_path, train=train, valid=valid, test=test, labels=labels)
    return train, valid, test


def train_nonzero_mask(matrix):
    return vector(matrix.sum(axis=0)) > 0


def fit_rna(train_raw, all_raw, n_top_genes):
    train_log = normalize_log1p(train_raw)
    means = vector(train_log.mean(axis=0))
    squared = vector(train_log.power(2).mean(axis=0))
    variance = squared - means**2
    n_keep = min(n_top_genes, train_log.shape[1])
    hvg = np.argsort(variance)[-n_keep:]
    all_log = normalize_log1p(all_raw)
    return all_log[:, hvg], hvg


def fit_modality2(train_raw, all_raw, kind, tfidf_norm="l1"):
    if kind == "tfidf":
        transformer = TfidfTransformer(
            norm=tfidf_norm,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
        )
        transformer.fit(train_raw)
        return transformer.transform(all_raw).astype(np.float32)
    if kind == "log1p":
        return normalize_log1p(all_raw)
    raise ValueError(f"Unknown modality-2 transform: {kind}")


def fit_recommended_checkpoint_preprocessing(rna, mod2, n_top_genes):
    """Reproduce the preprocessing used by the current recommended checkpoint."""
    import episcanpy.api as epi
    import scanpy as sc

    rna_mask = train_nonzero_mask(rna.X)
    mod2_mask = train_nonzero_mask(mod2.X)
    rna_work = rna[:, rna_mask].copy()
    mod2_work = mod2[:, mod2_mask].copy()
    rna_raw = rna_work.X.tocsr()
    mod2_raw = mod2_work.X.tocsr()

    sc.pp.normalize_total(rna_work)
    sc.pp.log1p(rna_work)
    sc.pp.highly_variable_genes(rna_work, n_top_genes=n_top_genes)
    hvg = np.flatnonzero(rna_work.var["highly_variable"].to_numpy())
    rna_input = rna_work[:, hvg].X.copy()

    epi.pp.filter_features(
        mod2_work,
        min_cells=np.ceil(mod2_work.shape[0] * 0.10),
    )
    epi.pp.tfidf(mod2_work)
    mod2_input = mod2_work.layers["tf-idf"].copy()
    return (
        rna_raw,
        mod2_raw,
        rna_input,
        mod2_input,
        rna_mask,
        mod2_mask,
        hvg,
    )


class PairTensorDataset(Dataset):
    def __init__(self, rna_input, mod2_input, rna_raw, mod2_raw, indices, state_labels=None, state_targets=None):
        self.rna_input = torch.from_numpy(dense_float(rna_input[indices]))
        self.mod2_input = torch.from_numpy(dense_float(mod2_input[indices]))
        self.rna_raw = torch.from_numpy(dense_float(rna_raw[indices]))
        self.mod2_raw = torch.from_numpy(dense_float(mod2_raw[indices]))
        self.state_labels = None if state_labels is None else torch.from_numpy(np.asarray(state_labels, dtype=np.int64))
        self.state_targets = None if state_targets is None else torch.from_numpy(np.asarray(state_targets, dtype=np.float32))

    def __len__(self):
        return self.rna_input.shape[0]

    def __getitem__(self, index):
        result = (self.rna_input[index], self.mod2_input[index], self.rna_raw[index], self.mod2_raw[index])
        if self.state_labels is not None:
            result += (self.state_labels[index],)
        if self.state_targets is not None:
            result += (self.state_targets[index],)
        return result


class ControlledDecoder(state_sr.FeatureWiseStateContrastResidualZIDecoder):
    """One decoder family with an explicit, auditable mixture-weight ablation."""

    def __init__(
        self,
        *args,
        method,
        modality2_prototype_scale=0.0,
        modality2_prototype_penalty=0.08,
        rna_state_mean_scale=0.0,
        rna_state_mean_penalty=0.08,
        auxiliary_state_gate_gradient=False,
        **kwargs,
    ):
        super().__init__(
            *args,
            modality2_prototype_scale=modality2_prototype_scale,
            modality2_prototype_penalty=modality2_prototype_penalty,
            rna_state_mean_scale=rna_state_mean_scale,
            rna_state_mean_penalty=rna_state_mean_penalty,
            auxiliary_state_gate_gradient=auxiliary_state_gate_gradient,
            **kwargs,
        )
        self.method = method
        rna_dim, mod2_dim = args[0], args[1]
        self.rna_dim, self.atac_dim = rna_dim, mod2_dim
        self.rna_global_logits = nn.Parameter(torch.zeros(2))
        self.mod2_global_logits = nn.Parameter(torch.zeros(2))
        self.rna_feature_logits = nn.Parameter(torch.zeros(2, rna_dim))
        self.mod2_feature_logits = nn.Parameter(torch.zeros(2, mod2_dim))
        self._freeze_unused_parameters()

    def _freeze_unused_parameters(self):
        parameters = [
            self.rna_global_logits,
            self.mod2_global_logits,
            self.rna_feature_logits,
            self.mod2_feature_logits,
            self.rna_prior_bias,
            self.rna_prior_feature_weights,
            self.rna_gene_residual,
            self.atac_logits,
            self.prototype_gene_factors,
            self.prototype_modality2_factors,
            self.prototype_rna_mean_factors,
        ]
        for parameter in parameters:
            if parameter is None:
                continue
            parameter.requires_grad_(False)
        for parameter in self.prototype_gate.parameters():
            parameter.requires_grad_(False)
        if self.method == "global":
            self.rna_global_logits.requires_grad_(True)
            self.mod2_global_logits.requires_grad_(True)
        elif self.method == "feature":
            self.rna_feature_logits.requires_grad_(True)
            self.mod2_feature_logits.requires_grad_(True)
        elif self.method == "continuous_prior":
            self.rna_prior_bias.requires_grad_(True)
            self.rna_prior_feature_weights.requires_grad_(True)
            self.rna_gene_residual.requires_grad_(True)
            self.atac_logits.requires_grad_(True)
        elif self.method == "state_resolved":
            self.rna_prior_bias.requires_grad_(True)
            self.rna_prior_feature_weights.requires_grad_(True)
            self.rna_gene_residual.requires_grad_(True)
            self.atac_logits.requires_grad_(True)
            self.prototype_gene_factors.requires_grad_(True)
            if self.prototype_modality2_factors is not None:
                self.prototype_modality2_factors.requires_grad_(True)
            if self.prototype_rna_mean_factors is not None:
                self.prototype_rna_mean_factors.requires_grad_(True)
            for parameter in self.prototype_gate.parameters():
                parameter.requires_grad_(True)

    def _constant_logits(self, dim, device):
        return torch.zeros(2, dim, device=device)

    def get_rna_logits(self, z=None):
        if self.method == "fixed":
            return self._constant_logits(self.rna_dim, self.rna_global_logits.device)
        if self.method == "global":
            return self.rna_global_logits[:, None].expand(-1, self.rna_dim)
        if self.method == "feature":
            return self.rna_feature_logits
        if self.method == "continuous_prior":
            base = self.get_rna_base_logit()
            return torch.stack([base, torch.zeros_like(base)], dim=0)
        return super().get_rna_logits(z)

    def get_mod2_logits(self, z=None):
        if self.method == "fixed":
            return self._constant_logits(self.atac_dim, self.mod2_global_logits.device)
        if self.method == "global":
            return self.mod2_global_logits[:, None].expand(-1, self.atac_dim)
        if self.method == "feature":
            return self.mod2_feature_logits
        if self.method == "state_resolved":
            return self.get_modality2_logits(z)
        return self.atac_logits

    def prior_regularization_for_method(self, z):
        if self.method == "continuous_prior":
            return gentle_zi.FeatureWiseMixtureDecoder.prior_regularization(self)
        if self.method == "state_resolved":
            return super().prior_regularization(z)
        return z.new_zeros(())

    def forward(self, z, sample=None, state_context=None):
        state_z = z if state_context is None else state_context
        rna_mean = self.rna_decoder_mean(z) + 1e-6
        rna_dispersion = self.rna_decoder_dispersion(z) + 1e-6
        rna_poisson = self.rna_decoder_poisson(z) + 1e-6
        rna_state_mean_residual = (
            self.get_rna_state_mean_residual(state_z)
            if self.method == "state_resolved"
            else None
        )
        if rna_state_mean_residual is not None:
            state_mean_multiplier = torch.exp(rna_state_mean_residual)
            rna_mean = rna_mean * state_mean_multiplier
            rna_poisson = rna_poisson * state_mean_multiplier
        rna_nb_zero_logits = self.rna_decoder_nb_zero_logits(z)
        rna_ps_zero_logits = self.rna_decoder_ps_zero_logits(z)
        mod2_mean = self.atac_decoder_mean(z) + 1e-6
        mod2_dispersion = self.atac_decoder_dispersion(z) + 1e-6
        mod2_poisson = self.atac_decoder_poisson(z) + 1e-6
        mod2_nb_zero_logits = self.atac_decoder_nb_zero_logits(z)
        mod2_ps_zero_logits = self.atac_decoder_ps_zero_logits(z)

        rna_weights = torch.softmax(self.get_rna_logits(state_z), dim=0)
        mod2_weights = torch.softmax(self.get_mod2_logits(state_z), dim=0)
        rna_nb = gentle_zi.negative_binomial_from_mean_dispersion(
            rna_mean,
            rna_dispersion,
        )
        rna_ps = gentle_zi.Poisson(rate=rna_poisson)
        mod2_nb = gentle_zi.negative_binomial_from_mean_dispersion(
            mod2_mean,
            mod2_dispersion,
        )
        mod2_ps = gentle_zi.Poisson(rate=mod2_poisson)
        residual = self.get_cell_residual(state_z) if self.method == "state_resolved" else torch.zeros_like(rna_mean)
        modality2_residual = (
            self.get_modality2_cell_residual(state_z)
            if self.method == "state_resolved"
            else None
        )
        prototype_probs = self.get_prototype_probs(state_z) if self.method == "state_resolved" else None
        outputs = {
            "rna_mean": rna_mean, "rna_dispersion": rna_dispersion, "rna_poisson": rna_poisson,
            "rna_nb_dist": rna_nb, "rna_ps_dist": rna_ps,
            "rna_nb_zero_logits": rna_nb_zero_logits, "rna_ps_zero_logits": rna_ps_zero_logits,
            "atac_mean": mod2_mean, "atac_dispersion": mod2_dispersion, "atac_bnl": mod2_poisson,
            "atac_nb_dist": mod2_nb, "atac_bnl_dist": mod2_ps,
            "atac_nb_zero_logits": mod2_nb_zero_logits, "atac_ps_zero_logits": mod2_ps_zero_logits,
            "rna_lambda_1": rna_weights[0], "rna_lambda_2": rna_weights[1],
            "atac_lambda_1": mod2_weights[0], "atac_lambda_2": mod2_weights[1],
            "mixture_prior_reg": self.prior_regularization_for_method(state_z),
            "rna_cell_residual": residual, "rna_prototype_probs": prototype_probs,
            "atac_cell_residual": modality2_residual,
            "rna_state_mean_residual": rna_state_mean_residual,
        }
        return None, None, outputs


class ControlledModel(state_sr.scAPIFeatureLambda):
    def __init__(
        self,
        *args,
        method,
        state_head_source="classifier",
        state_gate_input="decoder",
        state_count=12,
        modality2_prototype_scale=0.0,
        modality2_prototype_penalty=0.08,
        rna_state_mean_scale=0.0,
        rna_state_mean_penalty=0.08,
        auxiliary_state_gate_gradient=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.decoder = ControlledDecoder(
            *args[:2],
            *args[4:6],
            hidden_dim=kwargs.get("hidden_dim", 512),
            latent_dim=kwargs.get("latent_dim", 32),
            dropout_rate=kwargs.get("dropout_rate", 0.2),
            method=method,
            prototype_count=state_count,
            modality2_prototype_scale=modality2_prototype_scale,
            modality2_prototype_penalty=modality2_prototype_penalty,
            rna_state_mean_scale=rna_state_mean_scale,
            rna_state_mean_penalty=rna_state_mean_penalty,
            auxiliary_state_gate_gradient=auxiliary_state_gate_gradient,
        )
        self.method = method
        self.state_count = int(state_count)
        self.state_head_source = state_head_source
        self.state_gate_input = state_gate_input
        with torch.random.fork_rng():
            self.state_classifier = nn.Sequential(
                nn.Linear(kwargs.get("latent_dim", 32), max(32, kwargs.get("latent_dim", 32))),
                nn.ReLU(),
                nn.Linear(max(32, kwargs.get("latent_dim", 32)), self.state_count),
            )
        if method != "state_resolved" or state_head_source == "prototype":
            for parameter in self.state_classifier.parameters():
                parameter.requires_grad_(False)

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        if self.method == "state_resolved" and self.state_head_source == "prototype":
            dist_params = outputs[-1]
            dist_params["state_logits"] = (
                self.decoder.prototype_gate(dist_params["shared_mu"])
                / max(self.decoder.prototype_temperature, 1e-6)
            )
        return outputs


def log_weight(weight):
    return torch.log(weight.unsqueeze(0) + 1e-8) if weight.dim() == 1 else torch.log(weight + 1e-8)


def mixture_log_probability(target, params, prefix):
    if prefix == "rna":
        nb, ps = params["rna_nb_dist"], params["rna_ps_dist"]
        nb_zi, ps_zi = params["rna_nb_zero_logits"], params["rna_ps_zero_logits"]
        w1, w2 = params["rna_lambda_1"], params["rna_lambda_2"]
    else:
        nb, ps = params["atac_nb_dist"], params["atac_bnl_dist"]
        nb_zi, ps_zi = params["atac_nb_zero_logits"], params["atac_ps_zero_logits"]
        w1, w2 = params["atac_lambda_1"], params["atac_lambda_2"]
    # NegativeBinomial.log_prob is defined for real-valued observations.  Keep
    # integer counts in floating storage here: CUDA's integer lgamma path
    # invokes NVRTC, which is unavailable in the local PyTorch installation.
    target = target.round().float()
    nb_lp = gentle_zi.zero_inflated_log_prob(nb, target, nb_zi)
    ps_lp = gentle_zi.zero_inflated_log_prob(ps, target, ps_zi)
    return torch.logsumexp(torch.stack([log_weight(w1) + nb_lp, log_weight(w2) + ps_lp]), dim=0)


def predictive_moments(params, prefix):
    if prefix == "rna":
        mean, disp, rate = params["rna_mean"], params["rna_dispersion"], params["rna_poisson"]
        zi1, zi2, w1, w2 = params["rna_nb_zero_logits"], params["rna_ps_zero_logits"], params["rna_lambda_1"], params["rna_lambda_2"]
    else:
        mean, disp, rate = params["atac_mean"], params["atac_dispersion"], params["atac_bnl"]
        zi1, zi2, w1, w2 = params["atac_nb_zero_logits"], params["atac_ps_zero_logits"], params["atac_lambda_1"], params["atac_lambda_2"]
    del disp
    nb_dist = params["rna_nb_dist"] if prefix == "rna" else params["atac_nb_dist"]
    nb_mean, nb_variance = nb_dist.mean, nb_dist.variance
    q1, q2 = 1.0 - torch.sigmoid(zi1), 1.0 - torch.sigmoid(zi2)
    first1, first2 = q1 * nb_mean, q2 * rate
    second1 = q1 * (nb_variance + nb_mean.square())
    second2 = q2 * (rate + rate.square())
    if w1.dim() == 1:
        w1, w2 = w1.unsqueeze(0), w2.unsqueeze(0)
    expected = w1 * first1 + w2 * first2
    second = w1 * second1 + w2 * second2
    p0_nb = torch.exp(params["rna_nb_dist"].log_prob(torch.zeros_like(mean)) if prefix == "rna" else params["atac_nb_dist"].log_prob(torch.zeros_like(mean)))
    p0_ps = torch.exp(-rate)
    zero = w1 * ((1.0 - q1) + q1 * p0_nb) + w2 * ((1.0 - q2) + q2 * p0_ps)
    return expected, torch.clamp(second - expected.square(), min=0.0), zero


def deterministic_parameters(model, rna_x, mod2_x):
    """Decode from mu_shared so repeated test evaluation is deterministic."""
    encoded = model.encoder(rna_x, mod2_x)
    z_rna, z_mod2 = encoded[0], encoded[1]
    rna_mu, rna_var = encoded[3], encoded[4]
    mod2_mu, mod2_var = encoded[5], encoded[6]
    shared_mu, shared_var = encoded[7], encoded[8]
    params = model.decoder(shared_mu, sample=None)[2]
    params.update({
        "rna_mu": rna_mu,
        "rna_var": rna_var,
        "atac_mu": mod2_mu,
        "atac_var": mod2_var,
        "shared_mu": shared_mu,
        "shared_var": shared_var,
        "state_logits": model.state_classifier(shared_mu),
    })
    return z_rna, z_mod2, shared_mu, params


def selected_weight(weight, feature_indices):
    if weight.dim() == 1:
        return weight[feature_indices].unsqueeze(0)
    return weight[:, feature_indices]


def sample_selected_counts(params, prefix, feature_indices):
    if prefix == "rna":
        mean = params["rna_mean"][:, feature_indices]
        dispersion = params["rna_dispersion"][:, feature_indices]
        rate = params["rna_poisson"][:, feature_indices]
        nb_zi = params["rna_nb_zero_logits"][:, feature_indices]
        ps_zi = params["rna_ps_zero_logits"][:, feature_indices]
        weight1 = selected_weight(params["rna_lambda_1"], feature_indices)
    else:
        mean = params["atac_mean"][:, feature_indices]
        dispersion = params["atac_dispersion"][:, feature_indices]
        rate = params["atac_bnl"][:, feature_indices]
        nb_zi = params["atac_nb_zero_logits"][:, feature_indices]
        ps_zi = params["atac_ps_zero_logits"][:, feature_indices]
        weight1 = selected_weight(params["atac_lambda_1"], feature_indices)
    nb_sample = gentle_zi.negative_binomial_from_mean_dispersion(
        mean,
        dispersion,
    ).sample()
    ps_sample = torch.distributions.Poisson(rate=rate).sample()
    nb_sample = torch.where(torch.rand_like(nb_sample) < torch.sigmoid(nb_zi), 0.0, nb_sample)
    ps_sample = torch.where(torch.rand_like(ps_sample) < torch.sigmoid(ps_zi), 0.0, ps_sample)
    choose_nb = torch.rand_like(nb_sample) < weight1
    return torch.where(choose_nb, nb_sample, ps_sample)


def posterior_predictive_coverage(
    model,
    loader,
    device,
    seed,
    draws=20,
    max_cells=512,
    max_features=256,
):
    """Estimate 90% predictive coverage including latent posterior sampling."""
    model.eval()
    rng = np.random.default_rng(seed + 2903)
    rna_dim = loader.dataset.rna_raw.shape[1]
    mod2_dim = loader.dataset.mod2_raw.shape[1]
    rna_indices = torch.as_tensor(
        np.sort(rng.choice(rna_dim, size=min(max_features, rna_dim), replace=False)),
        dtype=torch.long,
        device=device,
    )
    mod2_indices = torch.as_tensor(
        np.sort(rng.choice(mod2_dim, size=min(max_features, mod2_dim), replace=False)),
        dtype=torch.long,
        device=device,
    )
    totals = {
        "rna": {"covered": 0.0, "entries": 0, "width": 0.0},
        "mod2": {"covered": 0.0, "entries": 0, "width": 0.0},
    }
    seen = 0
    fork_devices = [device.index] if device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed + 3911)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + 3911)
        for batch in loader:
            remaining = max_cells - seen
            if remaining <= 0:
                break
            rna_x, mod2_x, rna_y, mod2_y = [value[:remaining].to(device) for value in batch[:4]]
            encoded = model.encoder(rna_x, mod2_x)
            shared_mu, shared_var = encoded[7], encoded[8]
            samples = {"rna": [], "mod2": []}
            for _ in range(draws):
                latent = shared_mu + torch.randn_like(shared_mu) * torch.sqrt(shared_var)
                params = model.decoder(latent, sample=None)[2]
                samples["rna"].append(sample_selected_counts(params, "rna", rna_indices))
                samples["mod2"].append(sample_selected_counts(params, "mod2", mod2_indices))
            for key, target, indices in (
                ("rna", rna_y, rna_indices),
                ("mod2", mod2_y, mod2_indices),
            ):
                sampled = torch.stack(samples[key], dim=0)
                lower = torch.quantile(sampled, 0.05, dim=0)
                upper = torch.quantile(sampled, 0.95, dim=0)
                observed = target[:, indices]
                covered = (observed >= lower) & (observed <= upper)
                totals[key]["covered"] += float(covered.sum().item())
                totals[key]["entries"] += int(covered.numel())
                totals[key]["width"] += float((upper - lower).sum().item())
            seen += rna_y.shape[0]
    result = {}
    for key, values in totals.items():
        result[key] = {
            "coverage_90": values["covered"] / max(values["entries"], 1),
            "interval_width_90": values["width"] / max(values["entries"], 1),
            "coverage_entries": values["entries"],
            "posterior_draws": draws,
        }
    return result


def evaluate(model, loader, device, modality2_name, seed, coverage_draws=20):
    model.eval()
    totals = {key: None for key in ("rna", "mod2")}
    count = 0
    with torch.no_grad():
        for rna_x, mod2_x, rna_y, mod2_y in loader:
            rna_x, mod2_x = rna_x.to(device), mod2_x.to(device)
            rna_y, mod2_y = rna_y.to(device), mod2_y.to(device)
            _, _, _, params = deterministic_parameters(model, rna_x, mod2_x)
            for key, target, prefix in (("rna", rna_y, "rna"), ("mod2", mod2_y, "mod2")):
                lp = mixture_log_probability(target, params, prefix)
                expected, variance, zero = predictive_moments(params, prefix)
                item = {
                    "nll": -lp.sum(dim=0), "obs_sum": target.sum(dim=0), "obs_sq": target.square().sum(dim=0),
                    "obs_zero": (target == 0).sum(dim=0), "pred_sum": expected.sum(dim=0),
                    "pred_second": (variance + expected.square()).sum(dim=0), "pred_zero": zero.sum(dim=0),
                    "entries": torch.tensor(target.shape[0], device=device),
                }
                if totals[key] is None:
                    totals[key] = {name: value.detach().cpu().double() for name, value in item.items()}
                else:
                    for name, value in item.items():
                        totals[key][name] += value.detach().cpu().double()
            count += rna_y.shape[0]
    coverage = posterior_predictive_coverage(
        model,
        loader,
        device,
        seed,
        draws=coverage_draws,
    )
    rows = []
    for key, display in (("rna", "RNA"), ("mod2", modality2_name)):
        item = totals[key]
        n = float(item["entries"].item())
        obs_mean = item["obs_sum"] / n
        pred_mean = item["pred_sum"] / n
        obs_var = torch.clamp(item["obs_sq"] / n - obs_mean.square(), min=0.0)
        pred_var = torch.clamp(item["pred_second"] / n - pred_mean.square(), min=0.0)
        obs_zero = item["obs_zero"] / n
        pred_zero = item["pred_zero"] / n
        rows.append({
            "modality": display, "test_cells": count, "features": len(obs_mean),
            "nll_per_entry": float(item["nll"].sum().item() / (n * len(obs_mean))),
            "zero_rate_mae": float(torch.mean(torch.abs(obs_zero - pred_zero)).item()),
            "mean_mae": float(torch.mean(torch.abs(obs_mean - pred_mean)).item()),
            "variance_mae": float(torch.mean(torch.abs(obs_var - pred_var)).item()),
            "coverage_90": coverage[key]["coverage_90"],
            "interval_width_90": coverage[key]["interval_width_90"],
            "coverage_entries": coverage[key]["coverage_entries"],
            "posterior_draws": coverage[key]["posterior_draws"],
        })
    joint = float(np.mean([row["nll_per_entry"] for row in rows]))
    return rows, joint


def state_feature_statistics(
    raw_counts,
    labels,
    n_states,
    top_features,
    excluded_features=None,
):
    means = np.zeros((n_states, raw_counts.shape[1]), dtype=np.float32)
    detections = np.zeros_like(means)
    state_sizes = np.bincount(labels, minlength=n_states).astype(np.int64)
    total_sum = vector(raw_counts.sum(axis=0)).astype(np.float64)
    total_cells = raw_counts.shape[0]
    for state in range(n_states):
        mask = labels == state
        if not np.any(mask):
            continue
        state_counts = raw_counts[mask]
        means[state] = vector(state_counts.mean(axis=0))
        detections[state] = vector((state_counts > 0).mean(axis=0))
    top_state = means.argmax(axis=0).astype(np.int64)
    sorted_means = np.sort(means, axis=0)
    sorted_detections = np.sort(detections, axis=0)
    top_mean = means[top_state, np.arange(means.shape[1])]
    top_detection = detections[top_state, np.arange(detections.shape[1])]
    second_mean = sorted_means[-2]
    second_detection = sorted_detections[-2]
    specificity = (top_mean - second_mean) + 0.25 * (top_detection - second_detection)
    eligible_mask = (top_mean > 0) & (specificity > 0)
    if excluded_features is not None:
        excluded_features = np.asarray(excluded_features, dtype=np.int64)
        excluded_features = excluded_features[
            (excluded_features >= 0) & (excluded_features < raw_counts.shape[1])
        ]
        eligible_mask[excluded_features] = False
    eligible = np.flatnonzero(eligible_mask)
    ranked = eligible[np.argsort(specificity[eligible])[::-1]]
    selected = ranked[: min(top_features, len(ranked))].astype(np.int64)
    selected_states = top_state[selected]
    other_means = np.zeros(len(selected), dtype=np.float32)
    for state in range(n_states):
        state_mask = selected_states == state
        if not np.any(state_mask):
            continue
        feature_indices = selected[state_mask]
        other_cells = max(total_cells - state_sizes[state], 1)
        state_sum = means[state, feature_indices] * state_sizes[state]
        other_means[state_mask] = (total_sum[feature_indices] - state_sum) / other_cells
    observed_log2fc = np.log2(
        (top_mean[selected] + 0.05) / (other_means + 0.05)
    ).astype(np.float32)
    return (
        selected,
        selected_states,
        specificity[selected],
        observed_log2fc,
        means,
    )


def compute_pseudostates(
    rna_input,
    mod2_input,
    rna_raw,
    mod2_raw,
    train_indices,
    n_states,
    seed,
    top_genes,
    top_modality2_features,
    fixed_labels=None,
    excluded_rna_features=None,
):
    rna = dense_float(rna_input[train_indices])
    mod2 = dense_float(mod2_input[train_indices])
    rna_embed = TruncatedSVD(n_components=min(30, rna.shape[1] - 1), random_state=seed).fit_transform(rna)
    mod2_embed = TruncatedSVD(n_components=min(30, mod2.shape[1] - 1), random_state=seed).fit_transform(mod2)
    combined = StandardScaler().fit_transform(np.concatenate([rna_embed, mod2_embed], axis=1))
    if fixed_labels is None:
        kmeans = KMeans(n_clusters=n_states, random_state=seed, n_init=10).fit(combined)
        labels = kmeans.labels_.astype(np.int64)
        targets, temperature = state_sr.calibrated_kmeans_soft_targets(
            np.square(kmeans.transform(combined)),
            0.90,
        )
    else:
        labels = np.asarray(fixed_labels, dtype=np.int64)
        if labels.shape != (len(train_indices),):
            raise ValueError(
                "fixed_labels must contain one state for every training cell."
            )
        if labels.min(initial=0) < 0 or labels.max(initial=0) >= n_states:
            raise ValueError("fixed_labels fall outside the configured state range.")
        if np.unique(labels).size != n_states:
            raise ValueError("Every configured fixed state must occur in training cells.")
        off_probability = (1.0 - 0.90) / max(n_states - 1, 1)
        targets = np.full(
            (len(labels), n_states),
            off_probability,
            dtype=np.float32,
        )
        targets[np.arange(len(labels)), labels] = 0.90
        temperature = float("nan")
    train_rna_counts = rna_raw[train_indices]
    train_mod2_counts = mod2_raw[train_indices]
    (
        selected,
        selected_states,
        selected_specificity,
        selected_log2fc,
        _,
    ) = state_feature_statistics(
        train_rna_counts,
        labels,
        n_states,
        top_genes,
        excluded_features=excluded_rna_features,
    )
    (
        selected_mod2,
        selected_mod2_states,
        selected_mod2_specificity,
        _,
        _,
    ) = state_feature_statistics(
        train_mod2_counts,
        labels,
        n_states,
        top_modality2_features,
    )
    if len(selected) == 0:
        raise ValueError("No state-specific RNA genes were found for contrast loss.")
    if len(selected_mod2) == 0:
        raise ValueError("No state-specific second-modality features were found.")

    global_rna_mean = vector(train_rna_counts.mean(axis=0))
    positive_mean = global_rna_mean[global_rna_mean > 0]
    low_cutoff, high_cutoff = np.quantile(positive_mean, [0.20, 0.80])
    low_expression = np.flatnonzero(
        (global_rna_mean > 0) & (global_rna_mean <= low_cutoff)
    ).astype(np.int64)
    high_expression = np.flatnonzero(global_rna_mean >= high_cutoff).astype(np.int64)
    return {
        "labels": labels,
        "targets": targets,
        "temperature": temperature,
        "rna_selected": selected,
        "rna_top_states": selected_states,
        "rna_specificity": selected_specificity,
        "rna_observed_log2fc": selected_log2fc,
        "mod2_selected": selected_mod2,
        "mod2_top_states": selected_mod2_states,
        "mod2_specificity": selected_mod2_specificity,
        "low_expression": low_expression,
        "high_expression": high_expression,
        "low_expression_cutoff": float(low_cutoff),
        "high_expression_cutoff": float(high_cutoff),
    }


def build_optimizer(model, args):
    special = []
    if args.method == "global":
        special = [
            ([model.decoder.rna_global_logits, model.decoder.mod2_global_logits], args.mixture_lr, 1e-5),
        ]
    elif args.method == "feature":
        special = [
            ([model.decoder.rna_feature_logits, model.decoder.mod2_feature_logits], args.mixture_lr, 1e-5),
        ]
    elif args.method in ("continuous_prior", "state_resolved"):
        special = [
            ([model.decoder.atac_logits], args.mixture_lr, 1e-5),
            ([model.decoder.rna_prior_bias, model.decoder.rna_prior_feature_weights], args.prior_lr, 1e-5),
            ([model.decoder.rna_gene_residual], args.gene_residual_lr, 0.0),
        ]
    special_ids = {id(parameter) for group, _, _ in special for parameter in group}
    main_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in special_ids
    ]
    groups = [{"params": main_parameters, "lr": args.main_lr}]
    groups.extend(
        {"params": group, "lr": learning_rate, "weight_decay": weight_decay}
        for group, learning_rate, weight_decay in special
    )
    return torch.optim.Adam(groups)


def weight_contrast_loss(weights, batch_states, selected_features, selected_top_states, margin):
    if selected_features is None or selected_features.numel() == 0:
        zero = weights.sum() * 0.0
        return zero, zero.detach()
    if weights.dim() == 1:
        weights = weights.unsqueeze(0).expand(len(batch_states), -1)
    selected_weights = weights.index_select(1, selected_features)
    losses, contrasts = [], []
    for state in torch.unique(batch_states):
        top_mask = batch_states == state
        other_mask = ~top_mask
        feature_mask = selected_top_states == state
        if not bool(top_mask.any()) or not bool(other_mask.any()) or not bool(feature_mask.any()):
            continue
        contrast = (
            selected_weights[top_mask][:, feature_mask].mean(dim=0)
            - selected_weights[other_mask][:, feature_mask].mean(dim=0)
        )
        losses.append(torch.relu(margin - contrast))
        contrasts.append(contrast)
    if not losses:
        zero = selected_weights.sum() * 0.0
        return zero, zero.detach()
    return torch.cat(losses).mean(), torch.cat(contrasts).mean().detach()


def rna_expected_mean(params):
    first = (1.0 - torch.sigmoid(params["rna_nb_zero_logits"])) * params["rna_mean"]
    second = (1.0 - torch.sigmoid(params["rna_ps_zero_logits"])) * params["rna_poisson"]
    return params["rna_lambda_1"] * first + params["rna_lambda_2"] * second


def state_mean_contrast_loss(
    expected,
    state_targets,
    selected_genes,
    selected_top_states,
    desired_log2fc,
):
    selected_expected = expected.index_select(1, selected_genes)
    losses, contrasts = [], []
    for state in torch.unique(selected_top_states):
        gene_mask = selected_top_states == state
        state_weight = state_targets[:, state].unsqueeze(1)
        other_weight = 1.0 - state_weight
        target_mean = (
            (selected_expected[:, gene_mask] * state_weight).sum(dim=0)
            / state_weight.sum().clamp_min(1e-6)
        )
        other_mean = (
            (selected_expected[:, gene_mask] * other_weight).sum(dim=0)
            / other_weight.sum().clamp_min(1e-6)
        )
        contrast = torch.log2((target_mean + 0.05) / (other_mean + 0.05))
        desired = desired_log2fc[gene_mask]
        losses.append(torch.nn.functional.smooth_l1_loss(contrast, desired, reduction="none"))
        contrasts.append(contrast)
    if not losses:
        zero = expected.sum() * 0.0
        return zero, zero.detach()
    return torch.cat(losses).mean(), torch.cat(contrasts).mean().detach()


def expression_order_loss(rna_weights, low_expression, high_expression, margin):
    if rna_weights.dim() == 1:
        low_mean = rna_weights.index_select(0, low_expression).mean()
        high_mean = rna_weights.index_select(0, high_expression).mean()
    else:
        low_mean = rna_weights.index_select(1, low_expression).mean()
        high_mean = rna_weights.index_select(1, high_expression).mean()
    gap = high_mean - low_mean
    return torch.relu(gap.new_tensor(margin) - gap), gap.detach()


def decoder_input_mode(epoch, args):
    if args.decoder_training_schedule == "shared_only":
        return "shared"
    if args.freeze_latent or epoch < args.warmup_epochs:
        return "shared"
    post_warmup_step = epoch - args.warmup_epochs
    if post_warmup_step % 2 == 0:
        return "shared"
    if (post_warmup_step // 2) % 2 == 0:
        return "rna"
    return "atac"


@torch.no_grad()
def validation_joint_nll(model, loader, device):
    model.eval()
    totals = {"rna": 0.0, "mod2": 0.0}
    entries = {"rna": 0, "mod2": 0}
    for batch in loader:
        rna_x, mod2_x, rna_y, mod2_y = [
            value.to(device, non_blocking=True) for value in batch[:4]
        ]
        _, _, _, _, _, params = model(
            rna_x,
            mod2_x,
            decoder_input_mode="shared",
            materialize_decoder_samples=False,
        )
        for key, target, prefix in (
            ("rna", rna_y, "rna"),
            ("mod2", mod2_y, "atac"),
        ):
            totals[key] += float(
                -mixture_log_probability(target, params, prefix).sum().item()
            )
            entries[key] += target.numel()
    return float(np.mean([totals[key] / entries[key] for key in totals]))


@torch.no_grad()
def extract_encoder_means(model, loader, device):
    model.eval()
    integrated, rna_latent, mod2_latent = [], [], []
    for batch in loader:
        rna_x, mod2_x = [
            value.to(device, non_blocking=True) for value in batch[:2]
        ]
        encoder_outputs = model.encoder(rna_x, mod2_x)
        rna_latent.append(encoder_outputs[3].detach().cpu().numpy())
        mod2_latent.append(encoder_outputs[5].detach().cpu().numpy())
        integrated.append(encoder_outputs[7].detach().cpu().numpy())
    return {
        "integrated": np.concatenate(integrated, axis=0),
        "rna": np.concatenate(rna_latent, axis=0),
        "atac": np.concatenate(mod2_latent, axis=0),
    }


def freeze_latent_modules(model):
    for module in (model.encoder, model.adversarial_module, model.state_classifier):
        for parameter in module.parameters():
            parameter.requires_grad_(False)


def keep_frozen_latent_modules_in_eval_mode(model):
    model.encoder.eval()
    model.adversarial_module.eval()
    model.state_classifier.eval()


def train(
    model,
    train_loader,
    valid_loader,
    args,
    device,
    selected_genes=None,
    selected_top_states=None,
    desired_rna_log2fc=None,
    selected_mod2_features=None,
    selected_mod2_top_states=None,
    low_expression=None,
    high_expression=None,
):
    optimizer = build_optimizer(model, args)
    tensor_names = (
        "selected_genes",
        "selected_top_states",
        "selected_mod2_features",
        "selected_mod2_top_states",
        "low_expression",
        "high_expression",
    )
    local_values = locals()
    tensors = {
        name: (
            None
            if local_values[name] is None
            else torch.as_tensor(local_values[name], dtype=torch.long, device=device)
        )
        for name in tensor_names
    }
    selected_genes = tensors["selected_genes"]
    selected_top_states = tensors["selected_top_states"]
    selected_mod2_features = tensors["selected_mod2_features"]
    selected_mod2_top_states = tensors["selected_mod2_top_states"]
    low_expression = tensors["low_expression"]
    high_expression = tensors["high_expression"]
    desired_rna_log2fc = (
        None
        if desired_rna_log2fc is None
        else torch.as_tensor(desired_rna_log2fc, dtype=torch.float32, device=device)
    )
    started = time.time()
    history = []
    best_validation = math.inf
    best_state = None
    stale_validations = 0
    for epoch in range(args.epochs):
        model.train()
        if args.freeze_latent:
            # Frozen BatchNorm buffers must not drift when the decoder is calibrated.
            keep_frozen_latent_modules_in_eval_mode(model)
        epoch_totals = {
            "loss": 0.0,
            "rna_recon_loss": 0.0,
            "modality2_recon_loss": 0.0,
            "kl_loss": 0.0,
            "shared_reg_loss": 0.0,
            "adv_loss": 0.0,
            "prior_reg_loss": 0.0,
            "rna_weight_contrast": 0.0,
            "mod2_weight_contrast": 0.0,
            "rna_mean_contrast": 0.0,
            "expression_gap": 0.0,
            "state_target_loss": 0.0,
            "state_gate_max_probability": 0.0,
        }
        batches = 0
        mode = decoder_input_mode(epoch, args)
        for batch_idx, batch in enumerate(train_loader):
            rna_x, mod2_x, rna_y, mod2_y = batch[:4]
            state_labels = batch[4].to(device, non_blocking=True) if len(batch) >= 5 else None
            state_targets = batch[5].to(device, non_blocking=True) if len(batch) >= 6 else None
            rna_x, mod2_x = rna_x.to(device, non_blocking=True), mod2_x.to(device, non_blocking=True)
            rna_y, mod2_y = rna_y.to(device, non_blocking=True), mod2_y.to(device, non_blocking=True)
            if not args.freeze_latent and batch_idx % 5 == 0:
                with torch.no_grad():
                    z_rna, z_mod2, z_shared, *_ = model.encoder(rna_x, mod2_x)
                model.adversarial_module.train_discriminator_step(
                    z_shared,
                    z_rna,
                    z_mod2,
                )
            optimizer.zero_grad(set_to_none=True)
            _, _, z, zr, zm, params_out = model(rna_x, mod2_x, decoder_input_mode=mode, materialize_decoder_samples=False)
            losses = true_mixture_loss_function(
                rna_y,
                mod2_y,
                params_out,
                zr,
                zm,
                z_shared=z,
                gamma=args.adversarial_weight,
                alpha=args.kl_weight,
                beta=args.mmd_weight,
                alignment_type="mmd",
                prior_reg_weight=args.prior_reg_weight,
            )
            loss = losses["total_loss"]
            rna_weight_contrast = loss.detach().new_zeros(())
            mod2_weight_contrast = loss.detach().new_zeros(())
            rna_mean_contrast = loss.detach().new_zeros(())
            expression_gap = loss.detach().new_zeros(())
            state_target_loss = loss.detach().new_zeros(())
            state_gate_max_probability = loss.detach().new_zeros(())
            if args.method == "state_resolved" and state_labels is not None:
                contrast_loss, rna_weight_contrast = weight_contrast_loss(
                    params_out["rna_lambda_1"],
                    state_labels,
                    selected_genes,
                    selected_top_states,
                    args.state_contrast_margin,
                )
                loss = loss + args.state_contrast_weight * contrast_loss
                mod2_contrast_loss, mod2_weight_contrast = weight_contrast_loss(
                    params_out["atac_lambda_1"],
                    state_labels,
                    selected_mod2_features,
                    selected_mod2_top_states,
                    args.modality2_state_contrast_margin,
                )
                loss = loss + args.modality2_state_contrast_weight * mod2_contrast_loss
                order_loss, expression_gap = expression_order_loss(
                    params_out["rna_lambda_1"],
                    low_expression,
                    high_expression,
                    args.expression_order_margin,
                )
                loss = loss + args.expression_order_weight * order_loss
            if args.method == "state_resolved" and state_targets is not None and epoch >= args.state_start_epoch:
                state_loss = -(state_targets * torch.nn.functional.log_softmax(params_out["state_logits"], dim=1)).sum(dim=1).mean()
                state_target_loss = state_loss.detach()
                if params_out["rna_prototype_probs"] is not None:
                    state_gate_max_probability = (
                        params_out["rna_prototype_probs"].max(dim=1).values.mean().detach()
                    )
                if args.state_ramp_epochs > 0:
                    ramp = min(1.0, (epoch - args.state_start_epoch + 1) / args.state_ramp_epochs)
                else:
                    ramp = 1.0
                loss = loss + args.state_ce_weight * ramp * state_loss
                desired = torch.clamp(
                    desired_rna_log2fc * args.state_mean_target_fraction,
                    min=0.0,
                    max=args.state_mean_target_max_log2fc,
                )
                mean_loss, rna_mean_contrast = state_mean_contrast_loss(
                    rna_expected_mean(params_out),
                    state_targets,
                    selected_genes,
                    selected_top_states,
                    desired,
                )
                loss = loss + args.state_mean_contrast_weight * ramp * mean_loss
            loss.backward()
            optimizer.step()
            epoch_totals["loss"] += float(loss.item())
            epoch_totals["rna_recon_loss"] += float(
                losses["rna_recon_loss"].item()
            )
            epoch_totals["modality2_recon_loss"] += float(
                losses["atac_recon_loss"].item()
            )
            epoch_totals["kl_loss"] += float(losses["kl_loss"].item())
            epoch_totals["shared_reg_loss"] += float(
                losses["shared_reg_loss"].item()
            )
            epoch_totals["adv_loss"] += float(losses["adv_loss"].item())
            epoch_totals["prior_reg_loss"] += float(
                losses["prior_reg_loss"].item()
            )
            epoch_totals["rna_weight_contrast"] += float(rna_weight_contrast.item())
            epoch_totals["mod2_weight_contrast"] += float(mod2_weight_contrast.item())
            epoch_totals["rna_mean_contrast"] += float(rna_mean_contrast.item())
            epoch_totals["expression_gap"] += float(expression_gap.item())
            epoch_totals["state_target_loss"] += float(state_target_loss.item())
            epoch_totals["state_gate_max_probability"] += float(
                state_gate_max_probability.item()
            )
            batches += 1
        elapsed = time.time() - started
        per_epoch = elapsed / (epoch + 1)
        eta = per_epoch * (args.epochs - epoch - 1)
        averages = {
            key: value / max(batches, 1) for key, value in epoch_totals.items()
        }
        validation_nll = math.nan
        should_validate = (
            valid_loader is not None
            and (
                (epoch + 1) % args.validation_interval == 0
                or epoch + 1 == args.epochs
            )
        )
        if should_validate:
            validation_nll = validation_joint_nll(model, valid_loader, device)
            if validation_nll < best_validation - args.validation_min_delta:
                best_validation = validation_nll
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale_validations = 0
            else:
                stale_validations += 1
        history.append({
            "epoch": epoch + 1,
            "mode": mode,
            **averages,
            "validation_joint_nll": validation_nll,
            "elapsed_seconds": elapsed,
        })
        validation_text = (
            f" | val_nll={validation_nll:.6f}" if should_validate else ""
        )
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs}"
            f" | mode={mode}"
            f" | loss={averages['loss']:.6f}"
            f" | RNA={averages['rna_recon_loss']:.6f}"
            f" | {args.modality2_name}={averages['modality2_recon_loss']:.6f}"
            f" | KL={averages['kl_loss']:.6f}"
            f" | MMD={averages['shared_reg_loss']:.6f}"
            f" | ADV={averages['adv_loss']:.6f}"
            f" | RNA权重差={averages['rna_weight_contrast']:.4f}"
            f" | 二模态权重差={averages['mod2_weight_contrast']:.4f}"
            f" | RNA均值log2差={averages['rna_mean_contrast']:.4f}"
            f" | 高低表达权重差={averages['expression_gap']:.4f}"
            f" | 软状态最大概率={averages['state_gate_max_probability']:.3f}"
            f"{validation_text}"
            f" | elapsed={elapsed/60:.1f} min | eta={eta/60:.1f} min",
            flush=True,
        )
        if (
            should_validate
            and args.early_stopping_patience > 0
            and stale_validations >= args.early_stopping_patience
        ):
            print(
                f"Early stopping after {epoch + 1} epochs; "
                f"best validation joint NLL={best_validation:.6f}.",
                flush=True,
            )
            break
    if best_state is not None and not args.keep_last_checkpoint:
        model.load_state_dict(best_state)
        print(
            f"Restored best decoder checkpoint; validation joint NLL={best_validation:.6f}.",
            flush=True,
        )
    elif args.keep_last_checkpoint:
        print(
            "Kept the final epoch checkpoint as requested; "
            f"best observed validation joint NLL={best_validation:.6f}.",
            flush=True,
        )
    return history


def main(args):
    if args.method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    setup_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    output = (
        Path(args.output_dir)
        / args.dataset_name
        / METHOD_OUTPUT_NAMES[args.method]
        / f"随机种子_{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "运行配置.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    rna = ad.read_h5ad(args.rna_h5ad)
    mod2 = ad.read_h5ad(args.modality2_h5ad)
    if rna.n_obs != mod2.n_obs or not np.array_equal(rna.obs_names, mod2.obs_names):
        raise ValueError("Paired modalities must have identical cell order.")
    for name, item in (("RNA", rna), (args.modality2_name, mod2)):
        if item.uns.get("count_scale") != "raw_integer":
            raise ValueError(f"{name} is not declared raw_integer; cannot evaluate raw-count NLL.")
    split_path = (
        Path(args.split_file)
        if args.split_file
        else Path(args.output_dir)
        / "数据划分"
        / f"{args.dataset_name}_随机种子_{args.seed}.npz"
    )
    train_idx, valid_idx, test_idx = load_or_create_split(rna, split_path, args.seed)
    print(f"Dataset={args.dataset_name}; method={args.method}; split train/valid/test={len(train_idx)}/{len(valid_idx)}/{len(test_idx)}", flush=True)

    if args.preprocessing_mode == "recommended_checkpoint":
        (
            rna_raw,
            mod2_raw,
            rna_input,
            mod2_input,
            rna_mask,
            mod2_mask,
            hvg,
        ) = fit_recommended_checkpoint_preprocessing(
            rna,
            mod2,
            args.n_top_genes,
        )
    else:
        rna_mask = train_nonzero_mask(rna.X[train_idx])
        if args.modality2_transform == "tfidf":
            train_detection = vector((mod2.X[train_idx] > 0).sum(axis=0))
            mod2_mask = train_detection >= np.ceil(len(train_idx) * 0.10)
        else:
            mod2_mask = train_nonzero_mask(mod2.X[train_idx])
        rna_raw = rna.X[:, rna_mask].tocsr()
        mod2_raw = mod2.X[:, mod2_mask].tocsr()
        rna_input, hvg = fit_rna(
            rna_raw[train_idx],
            rna_raw,
            args.n_top_genes,
        )
        mod2_input = fit_modality2(
            mod2_raw[train_idx],
            mod2_raw,
            args.modality2_transform,
            tfidf_norm=args.tfidf_norm,
        )
    print(
        "Preprocessing="
        f"{args.preprocessing_mode}; target dims={rna_raw.shape[1]}/{mod2_raw.shape[1]}; "
        f"encoder dims={rna_input.shape[1]}/{mod2_input.shape[1]}",
        flush=True,
    )
    np.savez_compressed(
        output / "预处理特征.npz",
        rna_target_mask=rna_mask,
        mod2_target_mask=mod2_mask,
        rna_hvg_local_indices=hvg,
    )

    pseudo_labels = None
    pseudo_targets = None
    selected_genes = None
    selected_top_states = None
    desired_rna_log2fc = None
    selected_mod2_features = None
    selected_mod2_top_states = None
    low_expression = None
    high_expression = None
    if args.method == "state_resolved" and not args.evaluate_only:
        fixed_labels = None
        if args.fixed_state_obs_key:
            if args.fixed_state_obs_key not in rna.obs:
                raise KeyError(
                    f"RNA obs does not contain fixed state key: {args.fixed_state_obs_key}"
                )
            fixed_labels = np.asarray(
                rna.obs[args.fixed_state_obs_key].to_numpy()[train_idx],
                dtype=np.int64,
            )

        excluded_rna_features = None
        if args.state_contrast_exclude_truth_csv:
            exclusion_table = pd.read_csv(args.state_contrast_exclude_truth_csv)
            if "gene" not in exclusion_table:
                raise KeyError("State-contrast exclusion table must contain a gene column.")
            target_names = np.asarray(rna.var_names[rna_mask], dtype=str)
            excluded_names = set(exclusion_table["gene"].astype(str))
            excluded_rna_features = np.flatnonzero(
                np.isin(target_names, list(excluded_names))
            ).astype(np.int64)
            if len(excluded_rna_features) != len(excluded_names):
                missing = sorted(excluded_names.difference(target_names))
                raise ValueError(
                    "Some excluded evaluation genes are absent after train-fitted filtering: "
                    + ", ".join(missing[:20])
                )
            print(
                f"Excluded {len(excluded_rna_features)} evaluation genes from "
                "the state-contrast auxiliary loss.",
                flush=True,
            )

        pseudo_state_data = compute_pseudostates(
            rna_input,
            mod2_input,
            rna_raw,
            mod2_raw,
            train_idx,
            args.state_clusters,
            args.seed,
            args.state_contrast_top_genes,
            args.modality2_state_contrast_top_features,
            fixed_labels=fixed_labels,
            excluded_rna_features=excluded_rna_features,
        )
        pseudo_labels = pseudo_state_data["labels"]
        pseudo_targets = pseudo_state_data["targets"]
        pseudo_temperature = pseudo_state_data["temperature"]
        selected_genes = pseudo_state_data["rna_selected"]
        selected_top_states = pseudo_state_data["rna_top_states"]
        selected_specificity = pseudo_state_data["rna_specificity"]
        desired_rna_log2fc = pseudo_state_data["rna_observed_log2fc"]
        selected_mod2_features = pseudo_state_data["mod2_selected"]
        selected_mod2_top_states = pseudo_state_data["mod2_top_states"]
        low_expression = pseudo_state_data["low_expression"]
        high_expression = pseudo_state_data["high_expression"]
        np.save(output / "训练集伪状态标签.npy", pseudo_labels)
        np.save(output / "训练集伪状态软目标.npy", pseudo_targets)
        pd.DataFrame({
            "rna_target_index": selected_genes,
            "top_state": selected_top_states,
            "specificity_score": selected_specificity,
            "observed_train_log2fc": desired_rna_log2fc,
            "target_log2fc": np.clip(
                desired_rna_log2fc * args.state_mean_target_fraction,
                0.0,
                args.state_mean_target_max_log2fc,
            ),
        }).to_csv(output / "状态对比入选基因.csv", index=False)
        pd.DataFrame({
            "modality2_target_index": selected_mod2_features,
            "top_state": selected_mod2_top_states,
            "specificity_score": pseudo_state_data["mod2_specificity"],
        }).to_csv(output / "第二模态状态对比入选特征.csv", index=False)
        pd.DataFrame({
            "rna_target_index": np.concatenate([low_expression, high_expression]),
            "expression_group": (
                ["普遍低表达"] * len(low_expression)
                + ["普遍高表达"] * len(high_expression)
            ),
        }).to_csv(output / "RNA表达次序特征.csv", index=False)
        with (output / "伪状态元数据.json").open("w", encoding="utf-8") as handle:
            json.dump({
                "target_mean_max_probability": 0.90,
                "temperature": pseudo_temperature,
                "state_counts": np.bincount(pseudo_labels, minlength=args.state_clusters).astype(int).tolist(),
                "selected_rna_genes": int(len(selected_genes)),
                "selected_modality2_features": int(len(selected_mod2_features)),
                "low_expression_cutoff": pseudo_state_data["low_expression_cutoff"],
                "high_expression_cutoff": pseudo_state_data["high_expression_cutoff"],
                "low_expression_genes": int(len(low_expression)),
                "high_expression_genes": int(len(high_expression)),
                "fixed_state_obs_key": args.fixed_state_obs_key,
                "excluded_evaluation_genes": int(
                    0 if excluded_rna_features is None else len(excluded_rna_features)
                ),
            }, handle, indent=2)
    train_set = PairTensorDataset(
        rna_input,
        mod2_input,
        rna_raw,
        mod2_raw,
        train_idx,
        state_labels=pseudo_labels,
        state_targets=pseudo_targets,
    )
    valid_set = PairTensorDataset(rna_input, mod2_input, rna_raw, mod2_raw, valid_idx)
    test_set = PairTensorDataset(rna_input, mod2_input, rna_raw, mod2_raw, test_idx)
    all_indices = np.arange(rna.n_obs, dtype=np.int64)
    all_set = PairTensorDataset(rna_input, mod2_input, rna_raw, mod2_raw, all_indices)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    all_loader = DataLoader(all_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    rna_mean = vector(rna_raw[train_idx].mean(axis=0))
    rna_zero = 1.0 - vector((rna_raw[train_idx] > 0).mean(axis=0))
    model = ControlledModel(
        rna_raw.shape[1], mod2_raw.shape[1], rna_input.shape[1], mod2_input.shape[1],
        np.log10(rna_mean + 1e-8), rna_zero, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
        dropout_rate=args.dropout_rate, method=args.method,
        state_head_source=args.state_head_source,
        state_gate_input=args.state_gate_input,
        state_count=args.state_clusters,
        modality2_prototype_scale=args.modality2_prototype_scale,
        modality2_prototype_penalty=args.modality2_prototype_penalty,
        rna_state_mean_scale=args.rna_state_mean_scale,
        rna_state_mean_penalty=args.rna_state_mean_penalty,
        auxiliary_state_gate_gradient=args.auxiliary_state_gate_gradient,
    )
    device = torch.device(args.device)
    model.to(device)
    model_path = Path(args.model_path) if args.model_path else output / "模型权重.pt"
    if args.evaluate_only:
        print(f"Loading checkpoint for deterministic held-out evaluation: {model_path}", flush=True)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        if args.init_model_path:
            print(f"Warm-starting from checkpoint: {args.init_model_path}", flush=True)
            incompatible = model.load_state_dict(
                torch.load(
                    args.init_model_path,
                    map_location=device,
                    weights_only=True,
                ),
                strict=False,
            )
            print(f"  missing_keys={list(incompatible.missing_keys)}", flush=True)
            print(f"  unexpected_keys={list(incompatible.unexpected_keys)}", flush=True)
        if args.freeze_latent:
            freeze_latent_modules(model)
            print(
                "Encoder/adversarial/state-classifier parameters are frozen; "
                "latent embeddings will remain unchanged.",
                flush=True,
            )
        embeddings_before = extract_encoder_means(model, all_loader, device)
        history = train(
            model,
            train_loader,
            valid_loader,
            args,
            device,
            selected_genes=selected_genes,
            selected_top_states=selected_top_states,
            desired_rna_log2fc=desired_rna_log2fc,
            selected_mod2_features=selected_mod2_features,
            selected_mod2_top_states=selected_mod2_top_states,
            low_expression=low_expression,
            high_expression=high_expression,
        )
        pd.DataFrame(history).to_csv(output / "训练历史.csv", index=False)
        torch.save(model.state_dict(), model_path)
        embeddings_after = extract_encoder_means(model, all_loader, device)
        embedding_differences = {
            key: float(np.max(np.abs(embeddings_after[key] - embeddings_before[key])))
            for key in embeddings_after
        }
        np.savez_compressed(
            output / "潜变量嵌入.npz",
            cell_id=np.asarray(rna.obs_names, dtype=str),
            **embeddings_after,
        )
        difference_filename = (
            "潜变量保持.json"
            if args.freeze_latent
            else "潜变量训练更新.json"
        )
        with (output / difference_filename).open("w", encoding="utf-8") as handle:
            json.dump(embedding_differences, handle, indent=2)
        difference_label = (
            "Frozen-encoder preservation max-abs differences"
            if args.freeze_latent
            else "Initialization-to-trained embedding max-abs changes"
        )
        print(f"{difference_label}: {embedding_differences}", flush=True)
    metrics, joint_nll = evaluate(
        model,
        test_loader,
        device,
        args.modality2_name,
        args.seed,
        coverage_draws=args.coverage_draws,
    )
    for row in metrics:
        row.update({"dataset": args.dataset_name, "method": args.method, "seed": args.seed, "joint_nll": joint_nll})
    pd.DataFrame(metrics).to_csv(output / "测试指标.csv", index=False)
    with (output / "完成状态.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"状态": "成功", "联合负对数似然": joint_nll, "输出目录": str(output)},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print(f"Completed: {output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--rna_h5ad", required=True)
    parser.add_argument("--modality2_h5ad", required=True)
    parser.add_argument("--modality2_name", required=True)
    parser.add_argument("--modality2_transform", choices=("tfidf", "log1p"), required=True)
    parser.add_argument(
        "--tfidf_norm",
        choices=("l1", "l2"),
        default="l1",
        help="Row norm for train-fitted TF-IDF; l1 matches the recommended model.",
    )
    parser.add_argument(
        "--preprocessing_mode",
        choices=("train_fitted", "recommended_checkpoint"),
        default="train_fitted",
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument(
        "--output_dir",
        default=str(ROOT / "scLAMP-SR" / "results" / "PBMC_留出概率基准_正式版"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split_file",
        default="",
        help="Optional precomputed NPZ containing train, valid, and test indices.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_top_genes", type=int, default=3000)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--main_lr", type=float, default=1e-3)
    parser.add_argument("--mixture_lr", type=float, default=1e-4)
    parser.add_argument("--prior_lr", type=float, default=5e-5)
    parser.add_argument("--gene_residual_lr", type=float, default=5e-4)
    parser.add_argument("--prior_reg_weight", type=float, default=2e-3)
    parser.add_argument("--kl_weight", type=float, default=1e-7)
    parser.add_argument("--mmd_weight", type=float, default=1e-4)
    parser.add_argument("--adversarial_weight", type=float, default=0.1)
    parser.add_argument(
        "--decoder_training_schedule",
        choices=("alternating", "shared_only"),
        default="alternating",
        help=(
            "Decoder-input schedule after warmup. shared_only is the Experiment-3 "
            "ablation that never trains either decoder from modality-specific latents."
        ),
    )
    parser.add_argument("--state_clusters", type=int, default=12)
    parser.add_argument(
        "--fixed_state_obs_key",
        default="",
        help=(
            "Optional RNA obs column containing frozen label-free reference states. "
            "When set, these training-cell states replace internally fitted K-means labels."
        ),
    )
    parser.add_argument(
        "--state_contrast_exclude_truth_csv",
        default="",
        help=(
            "Optional CSV with a gene column. Listed genes remain reconstruction targets "
            "but are excluded from the state-contrast auxiliary loss."
        ),
    )
    parser.add_argument("--state_contrast_top_genes", type=int, default=1024)
    parser.add_argument("--state_contrast_weight", type=float, default=0.02)
    parser.add_argument("--state_contrast_margin", type=float, default=0.03)
    parser.add_argument("--state_ce_weight", type=float, default=0.02)
    parser.add_argument("--state_start_epoch", type=int, default=100)
    parser.add_argument("--state_ramp_epochs", type=int, default=30)
    parser.add_argument("--state_head_source", choices=("classifier", "prototype"), default="classifier")
    parser.add_argument("--state_gate_input", choices=("decoder", "shared_mean"), default="decoder")
    parser.add_argument("--modality2_prototype_scale", type=float, default=0.0)
    parser.add_argument("--modality2_prototype_penalty", type=float, default=0.08)
    parser.add_argument("--modality2_state_contrast_top_features", type=int, default=1024)
    parser.add_argument("--modality2_state_contrast_weight", type=float, default=0.0)
    parser.add_argument("--modality2_state_contrast_margin", type=float, default=0.03)
    parser.add_argument("--rna_state_mean_scale", type=float, default=0.0)
    parser.add_argument("--rna_state_mean_penalty", type=float, default=0.08)
    parser.add_argument("--state_mean_contrast_weight", type=float, default=0.0)
    parser.add_argument("--state_mean_target_fraction", type=float, default=0.25)
    parser.add_argument("--state_mean_target_max_log2fc", type=float, default=1.5)
    parser.add_argument("--expression_order_weight", type=float, default=0.0)
    parser.add_argument("--expression_order_margin", type=float, default=0.08)
    parser.add_argument("--auxiliary_state_gate_gradient", action="store_true")
    parser.add_argument("--freeze_latent", action="store_true")
    parser.add_argument("--init_model_path", default="")
    parser.add_argument("--validation_interval", type=int, default=5)
    parser.add_argument("--validation_min_delta", type=float, default=1e-5)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=4,
        help="Number of validation checks without improvement; zero disables stopping.",
    )
    parser.add_argument(
        "--keep_last_checkpoint",
        action="store_true",
        help="Keep the final epoch instead of restoring the best validation-NLL epoch.",
    )
    parser.add_argument("--coverage_draws", type=int, default=20)
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
