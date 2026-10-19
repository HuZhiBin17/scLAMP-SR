import json
import os
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# These packages are only required by the standalone preprocessing CLI.
sc = None
epi = None

from . import zi_decoder as gentle_zi
from .encoder import MultiModalDataset
from .losses import (
    BASE_ANALYZE_RNA_FEATURE_WEIGHTS,
    log_weight_for_likelihood,
    true_mixture_loss_function as base_true_mixture_loss_function,
)
from .utils import setup_seed


PROTOTYPE_COUNT = 12
PROTOTYPE_SCALE = 0.80
PROTOTYPE_PENALTY = 0.08
PROTOTYPE_ENTROPY_WEIGHT = 0.0
PROTOTYPE_TEMPERATURE = 0.35
PROTOTYPE_BALANCE_WEIGHT = 0.05
MODALITY2_PROTOTYPE_SCALE = 0.0
MODALITY2_PROTOTYPE_PENALTY = 0.08
RNA_STATE_MEAN_SCALE = 0.0
RNA_STATE_MEAN_PENALTY = 0.08
AUXILIARY_STATE_GATE_GRADIENT = False
STATE_HEAD_SOURCE = "classifier"
STATE_GATE_INPUT = "decoder"

STATE_CONTRAST_CLUSTERS = 12
STATE_CONTRAST_TOP_GENES = 1024
STATE_CONTRAST_WEIGHT = 0.02
STATE_CONTRAST_MARGIN = 0.03


class DistancePrototypeGate(nn.Module):
    """Fixed or trainable squared-distance logits for soft pseudo-states."""

    def __init__(self, centers, trainable=False):
        super().__init__()
        centers = torch.as_tensor(centers, dtype=torch.float32)
        if trainable:
            self.centers = nn.Parameter(centers)
        else:
            self.register_buffer("centers", centers)

    def forward(self, z):
        return -torch.cdist(z, self.centers).pow(2)


class FeatureWiseStateContrastResidualZIDecoder(gentle_zi.FeatureWiseMixtureDecoder):
    """ZINB/ZIP decoder with a feature baseline plus prototype-gated cell residual."""

    def __init__(
        self,
        rna_dim,
        atac_dim,
        rna_log_mean_expression,
        rna_zero_rate,
        hidden_dim=512,
        latent_dim=128,
        dropout_rate=0.2,
        prototype_count=12,
        prototype_scale=0.80,
        prototype_penalty=0.08,
        prototype_entropy_weight=0.0,
        prototype_temperature=0.35,
        prototype_balance_weight=0.05,
        modality2_prototype_scale=0.0,
        modality2_prototype_penalty=0.08,
        rna_state_mean_scale=0.0,
        rna_state_mean_penalty=0.08,
        auxiliary_state_gate_gradient=False,
    ):
        super().__init__(
            rna_dim,
            atac_dim,
            rna_log_mean_expression,
            rna_zero_rate,
            hidden_dim,
            latent_dim,
            dropout_rate,
        )
        self.prototype_count = prototype_count
        self.prototype_scale = prototype_scale
        self.prototype_penalty = prototype_penalty
        self.prototype_entropy_weight = prototype_entropy_weight
        self.prototype_temperature = prototype_temperature
        self.prototype_balance_weight = prototype_balance_weight
        self.modality2_prototype_scale = modality2_prototype_scale
        self.modality2_prototype_penalty = modality2_prototype_penalty
        self.rna_state_mean_scale = rna_state_mean_scale
        self.rna_state_mean_penalty = rna_state_mean_penalty
        self.auxiliary_state_gate_gradient = auxiliary_state_gate_gradient
        self.prototype_gate = nn.Sequential(
            nn.Linear(latent_dim, max(16, prototype_count * 2)),
            nn.ReLU(),
            nn.Linear(max(16, prototype_count * 2), prototype_count),
        )
        self.prototype_gene_factors = nn.Parameter(torch.zeros(prototype_count, rna_dim))
        if modality2_prototype_scale > 0.0:
            self.prototype_modality2_factors = nn.Parameter(
                torch.zeros(prototype_count, atac_dim)
            )
        else:
            self.register_parameter("prototype_modality2_factors", None)
        if rna_state_mean_scale > 0.0:
            self.prototype_rna_mean_factors = nn.Parameter(
                torch.zeros(prototype_count, rna_dim)
            )
        else:
            self.register_parameter("prototype_rna_mean_factors", None)

    def get_rna_base_logit(self):
        return self.get_rna_prior_logit() + self.rna_gene_residual

    def get_prototype_probs(self, z):
        logits = self.prototype_gate(z) / max(self.prototype_temperature, 1e-6)
        return torch.softmax(logits, dim=1)

    def set_distance_prototype_gate(
        self,
        centers,
        temperature,
        trainable_centers=False,
    ):
        if len(centers) != self.prototype_count:
            raise ValueError(
                "Distance-gate center count must match prototype_count."
            )
        self.prototype_gate = DistancePrototypeGate(
            centers,
            trainable=trainable_centers,
        )
        self.prototype_temperature = float(temperature)

    def get_cell_residual(self, z):
        prototype_probs = self.get_prototype_probs(z)
        raw_delta = torch.matmul(prototype_probs, self.prototype_gene_factors)
        return self.prototype_scale * torch.tanh(raw_delta)

    def get_auxiliary_prototype_probs(self, z):
        prototype_probs = self.get_prototype_probs(z)
        if not self.auxiliary_state_gate_gradient:
            prototype_probs = prototype_probs.detach()
        return prototype_probs

    def get_modality2_cell_residual(self, z):
        if self.prototype_modality2_factors is None:
            return None
        prototype_probs = self.get_auxiliary_prototype_probs(z)
        raw_delta = torch.matmul(prototype_probs, self.prototype_modality2_factors)
        return self.modality2_prototype_scale * torch.tanh(raw_delta)

    def get_modality2_logits(self, z=None):
        if z is None or self.prototype_modality2_factors is None:
            return self.atac_logits
        residual = self.get_modality2_cell_residual(z)
        first_logit = self.atac_logits[0].unsqueeze(0) + residual
        second_logit = self.atac_logits[1].unsqueeze(0).expand_as(first_logit)
        return torch.stack([first_logit, second_logit], dim=0)

    def get_rna_state_mean_residual(self, z):
        if self.prototype_rna_mean_factors is None:
            return None
        prototype_probs = self.get_auxiliary_prototype_probs(z)
        raw_delta = torch.matmul(prototype_probs, self.prototype_rna_mean_factors)
        return self.rna_state_mean_scale * torch.tanh(raw_delta)

    def get_rna_logits(self, z=None):
        base_logit = self.get_rna_base_logit()
        if z is None:
            return torch.stack([base_logit, torch.zeros_like(base_logit)], dim=0)
        residual = self.get_cell_residual(z)
        rna_nb_logit = base_logit.unsqueeze(0) + residual
        return torch.stack([rna_nb_logit, torch.zeros_like(rna_nb_logit)], dim=0)

    def prior_regularization(self, z=None):
        base_penalty = super().prior_regularization()
        factor_penalty = self.prototype_gene_factors.pow(2).mean()
        if self.prototype_modality2_factors is not None:
            factor_penalty = (
                factor_penalty
                + self.modality2_prototype_penalty
                * self.prototype_modality2_factors.pow(2).mean()
            )
        if self.prototype_rna_mean_factors is not None:
            factor_penalty = (
                factor_penalty
                + self.rna_state_mean_penalty
                * self.prototype_rna_mean_factors.pow(2).mean()
            )
        gate_penalty = sum(param.pow(2).mean() for param in self.prototype_gate.parameters())
        penalty = base_penalty + 0.02 * factor_penalty + 0.002 * gate_penalty
        if z is not None:
            residual = self.get_cell_residual(z)
            prototype_probs = self.get_prototype_probs(z)
            entropy = -(prototype_probs * torch.log(prototype_probs + 1e-8)).sum(dim=1).mean()
            usage = prototype_probs.mean(dim=0)
            target = torch.full_like(usage, 1.0 / max(self.prototype_count, 1))
            balance = (usage - target).pow(2).mean()
            penalty = (
                penalty
                + self.prototype_penalty * residual.pow(2).mean()
                - self.prototype_entropy_weight * entropy
                + self.prototype_balance_weight * balance
            )
        return penalty

    def forward(self, z, sample=True, state_context=None):
        state_z = z if state_context is None else state_context
        rna_mean = self.rna_decoder_mean(z) + 1e-6
        rna_dispersion = self.rna_decoder_dispersion(z) + 1e-6
        rna_poisson = self.rna_decoder_poisson(z) + 1e-6
        rna_state_mean_residual = self.get_rna_state_mean_residual(state_z)
        if rna_state_mean_residual is not None:
            state_mean_multiplier = torch.exp(rna_state_mean_residual)
            rna_mean = rna_mean * state_mean_multiplier
            rna_poisson = rna_poisson * state_mean_multiplier
        rna_nb_zero_logits = self.rna_decoder_nb_zero_logits(z)
        rna_ps_zero_logits = self.rna_decoder_ps_zero_logits(z)

        atac_mean = self.atac_decoder_mean(z) + 1e-6
        atac_dispersion = self.atac_decoder_dispersion(z) + 1e-6
        atac_poisson = self.atac_decoder_poisson(z) + 1e-6
        atac_nb_zero_logits = self.atac_decoder_nb_zero_logits(z)
        atac_ps_zero_logits = self.atac_decoder_ps_zero_logits(z)

        rna_weights = torch.softmax(self.get_rna_logits(state_z), dim=0)
        atac_weights = torch.softmax(self.get_modality2_logits(state_z), dim=0)
        rna_lambda_1, rna_lambda_2 = rna_weights[0], rna_weights[1]
        atac_lambda_1, atac_lambda_2 = atac_weights[0], atac_weights[1]

        rna_nb_dist = gentle_zi.negative_binomial_from_mean_dispersion(
            rna_mean,
            rna_dispersion,
        )
        rna_ps_dist = gentle_zi.Poisson(rate=rna_poisson)

        atac_nb_dist = gentle_zi.negative_binomial_from_mean_dispersion(
            atac_mean,
            atac_dispersion,
        )
        atac_ps_dist = gentle_zi.Poisson(rate=atac_poisson)

        if sample is None:
            rna_data = None
            atac_data = None
        elif sample:
            fork_devices = [z.device.index] if z.is_cuda else []
            # Reconstructed samples are output-only. Isolating their RNG prevents
            # them from changing dropout/reparameterization in the training path.
            with torch.random.fork_rng(devices=fork_devices):
                rna_data = (
                    rna_lambda_1 * gentle_zi.sample_zero_inflated(rna_nb_dist, rna_nb_zero_logits)
                    + rna_lambda_2 * gentle_zi.sample_zero_inflated(rna_ps_dist, rna_ps_zero_logits)
                )
                atac_data = (
                    atac_lambda_1
                    * gentle_zi.sample_zero_inflated(atac_nb_dist, atac_nb_zero_logits)
                    + atac_lambda_2
                    * gentle_zi.sample_zero_inflated(atac_ps_dist, atac_ps_zero_logits)
                )
        else:
            rna_first_mean = (1.0 - torch.sigmoid(rna_nb_zero_logits)) * rna_mean
            rna_second_mean = (1.0 - torch.sigmoid(rna_ps_zero_logits)) * rna_poisson
            rna_data = rna_lambda_1 * rna_first_mean + rna_lambda_2 * rna_second_mean
            atac_data = atac_lambda_1 * atac_mean + atac_lambda_2 * atac_poisson

        residual = self.get_cell_residual(state_z)
        modality2_residual = self.get_modality2_cell_residual(state_z)
        prototype_probs = self.get_prototype_probs(state_z)
        outputs = {
            "rna_mean": rna_mean,
            "rna_dispersion": rna_dispersion,
            "rna_poisson": rna_poisson,
            "rna_nb_dist": rna_nb_dist,
            "rna_ps_dist": rna_ps_dist,
            "rna_nb_zero_logits": rna_nb_zero_logits,
            "rna_ps_zero_logits": rna_ps_zero_logits,
            "atac_mean": atac_mean,
            "atac_dispersion": atac_dispersion,
            "atac_bnl": atac_poisson,
            "atac_nb_dist": atac_nb_dist,
            "atac_bnl_dist": atac_ps_dist,
            "atac_nb_zero_logits": atac_nb_zero_logits,
            "atac_ps_zero_logits": atac_ps_zero_logits,
            "rna_lambda_1": rna_lambda_1,
            "rna_lambda_2": rna_lambda_2,
            "atac_lambda_1": atac_lambda_1,
            "atac_lambda_2": atac_lambda_2,
            "rna_nb_zero_rate": torch.sigmoid(rna_nb_zero_logits),
            "rna_ps_zero_rate": torch.sigmoid(rna_ps_zero_logits),
            "atac_nb_zero_rate": torch.sigmoid(atac_nb_zero_logits),
            "atac_ps_zero_rate": torch.sigmoid(atac_ps_zero_logits),
            "mixture_prior_reg": self.prior_regularization(state_z),
            "rna_prior_nb_weight": self.get_rna_prior_nb_weight(),
            "rna_prior_strength": self.rna_prior_strength,
            "rna_cell_residual": residual,
            "atac_cell_residual": modality2_residual,
            "rna_state_mean_residual": rna_state_mean_residual,
            "rna_prototype_probs": prototype_probs,
            "rna_prototype_entropy": -(prototype_probs * torch.log(prototype_probs + 1e-8)).sum(dim=1).mean(),
            "rna_prototype_usage": prototype_probs.mean(dim=0),
        }
        return rna_data, atac_data, outputs


class scAPIFeatureLambda(gentle_zi.scAPIFeatureLambda):
    def __init__(
        self,
        rna_dim,
        atac_dim,
        rna_pro_dim,
        atac_pro_dim,
        rna_log_mean_expression,
        rna_zero_rate,
        hidden_dim=512,
        latent_dim=128,
        dropout_rate=0.2,
    ):
        super().__init__(
            rna_dim,
            atac_dim,
            rna_pro_dim,
            atac_pro_dim,
            rna_log_mean_expression,
            rna_zero_rate,
            hidden_dim,
            latent_dim,
            dropout_rate,
        )
        self.decoder = FeatureWiseStateContrastResidualZIDecoder(
            rna_dim,
            atac_dim,
            rna_log_mean_expression,
            rna_zero_rate,
            hidden_dim,
            latent_dim,
            dropout_rate,
            prototype_count=PROTOTYPE_COUNT,
            prototype_scale=PROTOTYPE_SCALE,
            prototype_penalty=PROTOTYPE_PENALTY,
            prototype_entropy_weight=PROTOTYPE_ENTROPY_WEIGHT,
            prototype_temperature=PROTOTYPE_TEMPERATURE,
            prototype_balance_weight=PROTOTYPE_BALANCE_WEIGHT,
            modality2_prototype_scale=MODALITY2_PROTOTYPE_SCALE,
            modality2_prototype_penalty=MODALITY2_PROTOTYPE_PENALTY,
            rna_state_mean_scale=RNA_STATE_MEAN_SCALE,
            rna_state_mean_penalty=RNA_STATE_MEAN_PENALTY,
            auxiliary_state_gate_gradient=AUXILIARY_STATE_GATE_GRADIENT,
        )
        self.state_gate_input = STATE_GATE_INPUT
        # Keep auxiliary-head initialization from perturbing the main model's RNG.
        with torch.random.fork_rng():
            self.state_classifier = nn.Sequential(
                nn.Linear(latent_dim, max(32, latent_dim)),
                nn.ReLU(),
                nn.Linear(max(32, latent_dim), STATE_CONTRAST_CLUSTERS),
            )

    def get_state_logits(self, shared_mu):
        if STATE_HEAD_SOURCE == "prototype":
            if self.decoder.prototype_count != STATE_CONTRAST_CLUSTERS:
                raise ValueError(
                    "prototype_count must equal state_contrast_clusters when "
                    "state_head_source='prototype'."
                )
            return (
                self.decoder.prototype_gate(shared_mu)
                / max(self.decoder.prototype_temperature, 1e-6)
            )
        return self.state_classifier(shared_mu)

    def forward(
        self,
        rna_data,
        atac_data,
        decoder_input_mode="shared",
        materialize_decoder_samples=False,
    ):
        (
            z_rna,
            z_atac,
            z_shared,
            rna_mu,
            rna_var,
            atac_mu,
            atac_var,
            shared_mu,
            shared_var,
        ) = self.encoder(rna_data, atac_data)

        if decoder_input_mode == "shared":
            decoder_input = z_shared
        elif decoder_input_mode == "rna":
            decoder_input = z_rna
        elif decoder_input_mode == "atac":
            decoder_input = z_atac
        else:
            raise ValueError(f"Invalid decoder_input_mode: {decoder_input_mode}")

        sample_mode = True if materialize_decoder_samples else None
        state_context = (
            shared_mu if self.state_gate_input == "shared_mean" else decoder_input
        )
        rna_data_out, atac_data_out, dist_params = self.decoder(
            decoder_input,
            sample=sample_mode,
            state_context=state_context,
        )
        adv_results = self.adversarial_module(
            z_shared,
            z_rna,
            z_atac,
            mode="adversarial",
        )
        dist_params["rna_mu"] = rna_mu
        dist_params["rna_var"] = rna_var
        dist_params["atac_mu"] = atac_mu
        dist_params["atac_var"] = atac_var
        dist_params["shared_mu"] = shared_mu
        dist_params["shared_var"] = shared_var
        dist_params["state_logits"] = self.get_state_logits(shared_mu)
        dist_params["adv_loss"] = adv_results["adversarial_loss"]
        return rna_data_out, atac_data_out, z_shared, z_rna, z_atac, dist_params


class StateContrastDataset(MultiModalDataset):
    def __init__(
        self,
        rna_adata,
        atac_adata,
        pseudo_states,
        pseudo_state_targets=None,
        backend="tensor",
    ):
        super().__init__(rna_adata, atac_adata)
        self.backend = backend
        self.pseudo_state_targets = pseudo_state_targets
        if backend == "tensor":
            self.rna_data = torch.from_numpy(np.asarray(self.rna_data, dtype=np.float32))
            self.atac_data = torch.from_numpy(np.asarray(self.atac_data, dtype=np.float32))
            self.rna_raw = torch.from_numpy(np.asarray(self.rna_raw, dtype=np.float32))
            self.atac_raw = torch.from_numpy(np.asarray(self.atac_raw, dtype=np.float32))
            self.pseudo_states = torch.as_tensor(pseudo_states, dtype=torch.long)
            if self.pseudo_state_targets is not None:
                self.pseudo_state_targets = torch.as_tensor(
                    self.pseudo_state_targets,
                    dtype=torch.float32,
                )
        else:
            self.pseudo_states = np.asarray(pseudo_states, dtype=np.int64)
            if self.pseudo_state_targets is not None:
                self.pseudo_state_targets = np.asarray(
                    self.pseudo_state_targets,
                    dtype=np.float32,
                )

    def __getitem__(self, idx):
        if self.backend == "tensor":
            result = (
                self.rna_data[idx],
                self.atac_data[idx],
                self.rna_raw[idx],
                self.atac_raw[idx],
                self.pseudo_states[idx],
            )
            if self.pseudo_state_targets is not None:
                result += (self.pseudo_state_targets[idx],)
            return result
        result = super().__getitem__(idx) + (
            torch.tensor(self.pseudo_states[idx], dtype=torch.long),
        )
        if self.pseudo_state_targets is not None:
            result += (
                torch.tensor(self.pseudo_state_targets[idx], dtype=torch.float32),
            )
        return result


def matrix_to_dense_float(matrix_like):
    if scipy.sparse.issparse(matrix_like):
        return matrix_like.toarray().astype(np.float32)
    return np.asarray(matrix_like, dtype=np.float32)


def axis_vector(matrix_like):
    if scipy.sparse.issparse(matrix_like):
        return np.asarray(matrix_like).ravel()
    return np.asarray(matrix_like).ravel()


def feature_mean(matrix_like):
    return axis_vector(matrix_like.mean(axis=0))


def feature_detection(matrix_like):
    if scipy.sparse.issparse(matrix_like):
        return axis_vector((matrix_like > 0).mean(axis=0))
    return np.mean(matrix_like > 0, axis=0)


def centered_log_ratio_input(matrix_like):
    """Return Seurat-style per-cell CLR input while preserving count targets."""
    dense = matrix_to_dense_float(matrix_like)
    geometric_mean = np.exp(np.mean(np.log1p(dense), axis=1, keepdims=True))
    return np.log1p(dense / np.maximum(geometric_mean, 1e-8)).astype(np.float32)


def low_rank_embedding(matrix_like, n_components, random_state):
    dense = matrix_to_dense_float(matrix_like)
    if dense.shape[1] <= n_components:
        return dense
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    return svd.fit_transform(dense)


def calibrated_kmeans_soft_targets(squared_distances, target_mean_max_probability):
    squared_distances = np.asarray(squared_distances, dtype=np.float64)
    n_states = squared_distances.shape[1]
    minimum = 1.0 / n_states
    target = float(target_mean_max_probability)
    if not minimum < target < 1.0:
        raise ValueError(
            "latent_state_soft_target_max_prob must be between "
            f"{minimum:.6f} and 1.0 for {n_states} pseudo-states."
        )

    def probabilities(temperature):
        logits = -squared_distances / max(float(temperature), 1e-12)
        logits -= logits.max(axis=1, keepdims=True)
        values = np.exp(logits)
        return values / values.sum(axis=1, keepdims=True)

    low = 0.0
    high = max(float(np.median(squared_distances)), 1e-6)
    while probabilities(high).max(axis=1).mean() > target:
        high *= 2.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if probabilities(middle).max(axis=1).mean() > target:
            low = middle
        else:
            high = middle
    targets = probabilities(high).astype(np.float32)
    return targets, float(high)


def compute_pseudo_state_targets(
    rna_adata,
    modality2_adata,
    args,
    output_dir,
    return_soft_targets=False,
):
    print("Computing label-free pseudo-states...")
    n_components = min(
        30,
        rna_adata.n_obs - 1,
        rna_adata.n_vars - 1,
        modality2_adata.n_vars - 1,
    )
    if n_components < 2:
        raise ValueError("Not enough cells/features to compute pseudo-state embedding.")

    rna_embed = low_rank_embedding(rna_adata.X, n_components, args.seed)
    modality2_embed = low_rank_embedding(
        modality2_adata.X,
        n_components,
        args.seed + 17,
    )
    embedding = np.concatenate([rna_embed, modality2_embed], axis=1)
    embedding = StandardScaler().fit_transform(embedding)

    kmeans = KMeans(n_clusters=args.state_contrast_clusters, random_state=args.seed, n_init=10)
    pseudo_states = kmeans.fit_predict(embedding).astype(np.int64)
    soft_targets = None
    soft_target_temperature = None
    requested_mean_max_probability = float(
        getattr(args, "latent_state_soft_target_max_prob", 1.0)
    )
    if requested_mean_max_probability < 1.0:
        squared_distances = np.square(kmeans.transform(embedding))
        soft_targets, soft_target_temperature = calibrated_kmeans_soft_targets(
            squared_distances,
            requested_mean_max_probability,
        )

    raw_matrix = rna_adata.raw.X if rna_adata.raw is not None else rna_adata.X
    gene_names = np.asarray(rna_adata.raw.var_names if rna_adata.raw is not None else rna_adata.var_names).astype(str)
    n_states = args.state_contrast_clusters
    means = np.zeros((n_states, raw_matrix.shape[1]), dtype=np.float32)
    detections = np.zeros_like(means)
    counts = np.zeros(n_states, dtype=np.int64)

    for state in range(n_states):
        mask = pseudo_states == state
        counts[state] = int(mask.sum())
        if counts[state] == 0:
            continue
        state_matrix = raw_matrix[mask]
        means[state] = feature_mean(state_matrix)
        detections[state] = feature_detection(state_matrix)

    top_state = means.argmax(axis=0).astype(np.int64)
    top_mean = means[top_state, np.arange(means.shape[1])]
    top_det = detections[top_state, np.arange(detections.shape[1])]
    sorted_means = np.sort(means, axis=0)
    sorted_dets = np.sort(detections, axis=0)
    second_mean = sorted_means[-2] if n_states > 1 else np.zeros_like(top_mean)
    second_det = sorted_dets[-2] if n_states > 1 else np.zeros_like(top_det)
    specificity_score = (top_mean - second_mean) + 0.25 * (top_det - second_det)
    eligible = np.where((top_mean > 0) & (specificity_score > 0))[0]
    if eligible.size == 0:
        raise ValueError("No state-specific RNA genes were found for contrast loss.")
    ranked = eligible[np.argsort(specificity_score[eligible])[::-1]]
    selected = ranked[: min(args.state_contrast_top_genes, ranked.size)].astype(np.int64)

    state_df = pd.DataFrame(
        {
            "state": np.arange(n_states, dtype=np.int64),
            "n_cells": counts,
        }
    )
    gene_df = pd.DataFrame(
        {
            "raw_index": selected,
            "gene": gene_names[selected],
            "top_state": top_state[selected],
            "top_mean": top_mean[selected],
            "second_mean": second_mean[selected],
            "top_detection": top_det[selected],
            "second_detection": second_det[selected],
            "specificity_score": specificity_score[selected],
        }
    )
    os.makedirs(output_dir, exist_ok=True)
    state_df.to_csv(os.path.join(output_dir, "伪状态细胞数.csv"), index=False)
    assignment_df = pd.DataFrame(
        {
            "cell": np.asarray(rna_adata.obs_names).astype(str),
            "pseudo_state": pseudo_states,
        }
    )
    if soft_targets is not None:
        assignment_df["pseudo_state_target_max_probability"] = soft_targets.max(axis=1)
        assignment_df["pseudo_state_target_entropy"] = -np.sum(
            soft_targets * np.log(soft_targets + 1e-12),
            axis=1,
        )
        np.save(
            os.path.join(output_dir, "伪状态软目标.npy"),
            soft_targets,
        )
    for label_column in ("label", "cell_type"):
        if label_column in rna_adata.obs:
            assignment_df[label_column] = rna_adata.obs[label_column].astype(str).to_numpy()
    assignment_df.to_csv(os.path.join(output_dir, "伪状态分配.csv"), index=False)
    gene_df.to_csv(os.path.join(output_dir, "状态对比入选基因.csv"), index=False)
    with open(os.path.join(output_dir, "状态对比元数据.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "n_states": int(n_states),
                "n_selected_genes": int(len(selected)),
                "embedding_components_per_modality": int(n_components),
                "state_counts": counts.astype(int).tolist(),
                "mean_specificity_score": float(specificity_score[selected].mean()),
                "soft_target_requested_mean_max_probability": (
                    requested_mean_max_probability
                ),
                "soft_target_actual_mean_max_probability": (
                    float(soft_targets.max(axis=1).mean())
                    if soft_targets is not None
                    else 1.0
                ),
                "soft_target_temperature": soft_target_temperature,
            },
            handle,
            indent=2,
        )
    print(f"Pseudo-state counts: {counts.tolist()}")
    if soft_targets is not None:
        print(
            "Pseudo-state soft targets: "
            f"mean max probability={soft_targets.max(axis=1).mean():.4f}, "
            f"temperature={soft_target_temperature:.6f}"
        )
    print(f"Selected {len(selected)} state-specific genes for lambda contrast.")
    if return_soft_targets:
        return pseudo_states, selected, top_state[selected], gene_df, soft_targets
    return pseudo_states, selected, top_state[selected], gene_df


def state_contrast_loss(dist_params, batch_states, selected_gene_indices, selected_top_states, margin):
    if selected_gene_indices.numel() == 0:
        zero = dist_params["rna_lambda_1"].sum() * 0.0
        return zero, zero.detach()

    weights = dist_params["rna_lambda_1"].index_select(1, selected_gene_indices)
    losses = []
    contrasts = []
    unique_states = torch.unique(batch_states)
    for state in unique_states:
        top_cell_mask = batch_states == state
        other_cell_mask = ~top_cell_mask
        if not bool(top_cell_mask.any()) or not bool(other_cell_mask.any()):
            continue
        gene_mask = selected_top_states == state
        if not bool(gene_mask.any()):
            continue
        top_mean = weights[top_cell_mask][:, gene_mask].mean(dim=0)
        other_mean = weights[other_cell_mask][:, gene_mask].mean(dim=0)
        contrast = top_mean - other_mean
        losses.append(torch.relu(margin - contrast))
        contrasts.append(contrast)

    if not losses:
        zero = weights.sum() * 0.0
        return zero, zero.detach()

    loss = torch.cat(losses).mean()
    contrast_mean = torch.cat(contrasts).mean().detach()
    return loss, contrast_mean


def supervised_contrastive_state_loss(shared_mu, labels, temperature=0.20):
    features = torch.nn.functional.normalize(shared_mu, dim=1)
    logits = torch.matmul(features, features.t()) / max(temperature, 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    valid = positive_mask.any(dim=1)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    mean_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
    return -mean_log_prob[valid].mean()


def true_mixture_loss_function(
    rna_data,
    atac_data,
    dist_params,
    rna_z,
    atac_z,
    z_shared=None,
    gamma=0.1,
    alpha=1.0,
    beta=0.5,
    alignment_type=None,
    prior_reg_weight=2e-3,
    batch_states=None,
    selected_gene_indices=None,
    selected_top_states=None,
    state_contrast_weight=0.02,
    state_contrast_margin=0.03,
    latent_state_ce_weight=0.0,
    latent_state_label_smoothing=0.0,
    batch_state_targets=None,
    latent_state_supcon_weight=0.0,
    latent_state_temperature=0.20,
):
    losses = base_true_mixture_loss_function(
        rna_data,
        atac_data,
        dist_params,
        rna_z,
        atac_z,
        z_shared=z_shared,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        alignment_type=alignment_type,
        prior_reg_weight=prior_reg_weight,
    )
    if batch_states is None or selected_gene_indices is None or selected_top_states is None:
        zero = losses["total_loss"] * 0.0
        losses["state_contrast_loss"] = zero
        losses["state_lambda_contrast_mean"] = zero.detach()
        losses["latent_state_ce_loss"] = zero
        losses["latent_state_supcon_loss"] = zero
        losses["latent_state_accuracy"] = zero.detach()
        return losses

    contrast_loss, contrast_mean = state_contrast_loss(
        dist_params,
        batch_states,
        selected_gene_indices,
        selected_top_states,
        state_contrast_margin,
    )
    losses["state_contrast_loss"] = contrast_loss
    losses["state_lambda_contrast_mean"] = contrast_mean
    state_logits = dist_params["state_logits"]
    if batch_state_targets is None:
        def state_ce_loss():
            return torch.nn.functional.cross_entropy(
                state_logits,
                batch_states,
                label_smoothing=latent_state_label_smoothing,
            )
    else:
        state_targets = batch_state_targets
        if latent_state_label_smoothing > 0.0:
            state_targets = (
                (1.0 - latent_state_label_smoothing) * state_targets
                + latent_state_label_smoothing / state_targets.shape[1]
            )

        def state_ce_loss():
            return -(
                state_targets * torch.nn.functional.log_softmax(state_logits, dim=1)
            ).sum(dim=1).mean()

    if latent_state_ce_weight > 0.0:
        ce_loss = state_ce_loss()
    else:
        with torch.no_grad():
            ce_loss = state_ce_loss()
    if latent_state_supcon_weight > 0.0:
        supcon_loss = supervised_contrastive_state_loss(
            dist_params["shared_mu"],
            batch_states,
            temperature=latent_state_temperature,
        )
    else:
        supcon_loss = dist_params["shared_mu"].detach().new_zeros(())
    with torch.no_grad():
        state_accuracy = (state_logits.argmax(dim=1) == batch_states).float().mean()
    losses["latent_state_ce_loss"] = ce_loss
    losses["latent_state_supcon_loss"] = supcon_loss
    losses["latent_state_accuracy"] = state_accuracy
    losses["total_loss"] = (
        losses["total_loss"]
        + state_contrast_weight * contrast_loss
        + latent_state_ce_weight * ce_loss
        + latent_state_supcon_weight * supcon_loss
    )
    return losses


def tensor_p95_abs(value):
    flat = value.detach().abs().reshape(-1)
    if flat.numel() == 0:
        return 0.0
    return float(torch.quantile(flat, 0.95).item())


def train_model(
    model,
    train_loader,
    optimizer,
    selected_gene_indices,
    selected_top_states,
    num_epochs=80,
    warmup_epochs=20,
    device="cuda",
    prior_reg_weight=2e-3,
    state_contrast_weight=0.02,
    state_contrast_margin=0.03,
    early_stopping_patience=30,
    checkpoint_epochs=None,
    checkpoint_dir=None,
    seed=0,
    materialize_decoder_samples=False,
    residual_p95_interval=10,
    pin_memory=True,
    max_batches_per_epoch=0,
    latent_state_ce_weight=0.0,
    latent_state_label_smoothing=0.0,
    latent_state_ramp_epochs=0,
    latent_state_supcon_weight=0.0,
    latent_state_temperature=0.20,
    latent_state_start_epoch=0,
    latent_state_supcon_start_epoch=-1,
    modality2_name="ATAC",
):
    model.train()
    patience = early_stopping_patience
    best_loss = float("inf")
    epochs_no_improve = 0
    checkpoint_epochs = set(checkpoint_epochs or [])
    history = []
    selected_gene_indices = torch.as_tensor(selected_gene_indices, dtype=torch.long, device=device)
    selected_top_states = torch.as_tensor(selected_top_states, dtype=torch.long, device=device)
    if latent_state_supcon_start_epoch < 0:
        latent_state_supcon_start_epoch = latent_state_start_epoch

    epoch_progress = tqdm(
        range(num_epochs),
        total=num_epochs,
        desc="scLAMP-SR epochs",
        unit="epoch",
        dynamic_ncols=True,
    )
    for epoch in epoch_progress:
        total_loss = 0.0
        total_batches = 0
        rna_loss_sum = 0.0
        atac_loss_sum = 0.0
        kl_loss_sum = 0.0
        adv_loss_sum = 0.0
        shared_reg_loss_sum = 0.0
        prior_reg_loss_sum = 0.0
        state_contrast_loss_sum = 0.0
        state_lambda_contrast_sum = 0.0
        residual_abs_mean_sum = 0.0
        residual_abs_p95_sum = 0.0
        prototype_entropy_sum = 0.0
        usage_min_sum = 0.0
        usage_max_sum = 0.0
        latent_state_ce_sum = 0.0
        latent_state_supcon_sum = 0.0
        latent_state_accuracy_sum = 0.0
        residual_p95_batches = 0
        total_cells = 0
        epoch_start = time.perf_counter()

        if epoch < warmup_epochs:
            decoder_input_mode = "shared"
        else:
            post_warmup_step = epoch - warmup_epochs
            if post_warmup_step % 2 == 0:
                decoder_input_mode = "shared"
            elif (post_warmup_step // 2) % 2 == 0:
                decoder_input_mode = "rna"
            else:
                decoder_input_mode = "atac"
        if epoch < latent_state_start_epoch:
            active_latent_state_ce_weight = 0.0
        elif latent_state_ramp_epochs > 0:
            ramp_progress = min(
                1.0,
                (epoch - latent_state_start_epoch + 1)
                / float(latent_state_ramp_epochs),
            )
            active_latent_state_ce_weight = (
                latent_state_ce_weight * ramp_progress
            )
        else:
            active_latent_state_ce_weight = latent_state_ce_weight
        active_latent_state_supcon_weight = (
            latent_state_supcon_weight
            if epoch >= latent_state_supcon_start_epoch
            else 0.0
        )

        for batch_idx, batch in enumerate(train_loader):
            rna_batch, atac_batch, rna_raw, atac_raw, batch_states = batch[:5]
            batch_state_targets = batch[5] if len(batch) > 5 else None
            non_blocking = bool(pin_memory and str(device).startswith("cuda"))
            rna_batch = rna_batch.to(device, non_blocking=non_blocking)
            atac_batch = atac_batch.to(device, non_blocking=non_blocking)
            rna_raw = rna_raw.to(device, non_blocking=non_blocking)
            atac_raw = atac_raw.to(device, non_blocking=non_blocking)
            batch_states = batch_states.to(device, non_blocking=non_blocking)
            if batch_state_targets is not None:
                batch_state_targets = batch_state_targets.to(
                    device,
                    non_blocking=non_blocking,
                )

            if batch_idx % 5 == 0:
                with torch.no_grad():
                    z_rna, z_atac, z_shared, *_ = model.encoder(rna_batch, atac_batch)
                model.adversarial_module.train_discriminator_step(z_shared, z_rna, z_atac)

            optimizer.zero_grad(set_to_none=True)
            _, _, z_shared, z_rna, z_atac, dist_params = model(
                rna_batch,
                atac_batch,
                decoder_input_mode=decoder_input_mode,
                materialize_decoder_samples=materialize_decoder_samples,
            )
            losses = true_mixture_loss_function(
                rna_raw,
                atac_raw,
                dist_params,
                z_rna,
                z_atac,
                z_shared,
                gamma=0.1,
                alpha=1e-7,
                beta=1e-4,
                alignment_type="mmd",
                prior_reg_weight=prior_reg_weight,
                batch_states=batch_states,
                selected_gene_indices=selected_gene_indices,
                selected_top_states=selected_top_states,
                state_contrast_weight=state_contrast_weight,
                state_contrast_margin=state_contrast_margin,
                latent_state_ce_weight=active_latent_state_ce_weight,
                latent_state_label_smoothing=latent_state_label_smoothing,
                batch_state_targets=batch_state_targets,
                latent_state_supcon_weight=active_latent_state_supcon_weight,
                latent_state_temperature=latent_state_temperature,
            )

            loss = losses["total_loss"]
            loss.backward()
            optimizer.step()

            residual = dist_params.get("rna_cell_residual")
            prototype_probs = dist_params.get("rna_prototype_probs")
            usage = dist_params.get("rna_prototype_usage")

            total_loss += loss.item()
            rna_loss_sum += losses["rna_recon_loss"].item()
            atac_loss_sum += losses["atac_recon_loss"].item()
            kl_loss_sum += losses["kl_loss"].item()
            adv_loss_sum += losses["adv_loss"].item()
            shared_reg_loss_sum += losses["shared_reg_loss"].item()
            prior_reg_loss_sum += losses["prior_reg_loss"].item()
            state_contrast_loss_sum += losses["state_contrast_loss"].item()
            state_lambda_contrast_sum += float(losses["state_lambda_contrast_mean"].item())
            latent_state_ce_sum += float(losses["latent_state_ce_loss"].item())
            latent_state_supcon_sum += float(losses["latent_state_supcon_loss"].item())
            latent_state_accuracy_sum += float(losses["latent_state_accuracy"].item())
            residual_abs_mean_sum += float(residual.detach().abs().mean().item()) if residual is not None else 0.0
            if (
                residual is not None
                and residual_p95_interval > 0
                and batch_idx % residual_p95_interval == 0
            ):
                residual_abs_p95_sum += tensor_p95_abs(residual)
                residual_p95_batches += 1
            if prototype_probs is not None:
                entropy = -(prototype_probs * torch.log(prototype_probs + 1e-8)).sum(dim=1).mean()
                prototype_entropy_sum += float(entropy.item())
            if usage is not None:
                usage_min_sum += float(usage.detach().min().item())
                usage_max_sum += float(usage.detach().max().item())
            total_batches += 1
            total_cells += int(rna_batch.shape[0])
            if max_batches_per_epoch > 0 and total_batches >= max_batches_per_epoch:
                break

        if total_batches == 0:
            continue

        avg_loss = total_loss / total_batches
        avg_rna_loss = rna_loss_sum / total_batches
        avg_atac_loss = atac_loss_sum / total_batches
        avg_kl_loss = kl_loss_sum / total_batches
        avg_adv_loss = adv_loss_sum / total_batches
        avg_shared_reg_loss = shared_reg_loss_sum / total_batches
        avg_prior_reg_loss = prior_reg_loss_sum / total_batches
        avg_state_contrast_loss = state_contrast_loss_sum / total_batches
        avg_state_lambda_contrast = state_lambda_contrast_sum / total_batches
        avg_residual_abs_mean = residual_abs_mean_sum / total_batches
        avg_prototype_entropy = prototype_entropy_sum / total_batches
        avg_usage_min = usage_min_sum / total_batches
        avg_usage_max = usage_max_sum / total_batches
        avg_latent_state_ce = latent_state_ce_sum / total_batches
        avg_latent_state_supcon = latent_state_supcon_sum / total_batches
        avg_latent_state_accuracy = latent_state_accuracy_sum / total_batches
        avg_residual_abs_p95 = (
            residual_abs_p95_sum / residual_p95_batches
            if residual_p95_batches
            else float("nan")
        )
        epoch_seconds = time.perf_counter() - epoch_start
        cells_per_second = total_cells / max(epoch_seconds, 1e-8)
        history.append(
            {
                "epoch": epoch + 1,
                "decoder_input_mode": decoder_input_mode,
                "total_loss": avg_loss,
                "rna_recon_loss": avg_rna_loss,
                "atac_recon_loss": avg_atac_loss,
                "kl_loss": avg_kl_loss,
                "adv_loss": avg_adv_loss,
                "shared_reg_loss": avg_shared_reg_loss,
                "prior_reg_loss": avg_prior_reg_loss,
                "state_contrast_loss": avg_state_contrast_loss,
                "state_lambda_contrast_mean": avg_state_lambda_contrast,
                "latent_state_ce_loss": avg_latent_state_ce,
                "latent_state_supcon_loss": avg_latent_state_supcon,
                "latent_state_accuracy": avg_latent_state_accuracy,
                "latent_state_ce_weight": active_latent_state_ce_weight,
                "latent_state_label_smoothing": latent_state_label_smoothing,
                "latent_state_supcon_weight": active_latent_state_supcon_weight,
                "rna_cell_residual_abs_mean": avg_residual_abs_mean,
                "rna_cell_residual_abs_p95": avg_residual_abs_p95,
                "prototype_entropy": avg_prototype_entropy,
                "prototype_usage_min": avg_usage_min,
                "prototype_usage_max": avg_usage_max,
                "epoch_seconds": epoch_seconds,
                "cells_per_second": cells_per_second,
            }
        )
        if checkpoint_dir:
            pd.DataFrame(history).to_csv(
                os.path.join(checkpoint_dir, "训练历史.csv"),
                index=False,
            )
        if epoch + 1 in checkpoint_epochs:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"模型_随机种子_{seed}_第{epoch + 1}轮.pt",
            )
            torch.save(model.state_dict(), checkpoint_path)
            tqdm.write(f"Saved epoch checkpoint: {checkpoint_path}")

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        epoch_progress.set_postfix(
            {
                "mode": decoder_input_mode,
                "total": f"{avg_loss:.4f}",
                "rna": f"{avg_rna_loss:.4f}",
                str(modality2_name).lower(): f"{avg_atac_loss:.4f}",
                "state": f"{avg_state_lambda_contrast:.4f}",
            }
        )
        tqdm.write(
            f"Epoch {epoch + 1}/{num_epochs}, mode={decoder_input_mode}, "
            f"TL={avg_loss:.6f}, RNA={avg_rna_loss:.6f}, "
            f"{str(modality2_name).upper()}={avg_atac_loss:.6f}, "
            f"KL={avg_kl_loss:.6f}, ADV={avg_adv_loss:.6f}, Shared={avg_shared_reg_loss:.6f}, "
            f"Prior={avg_prior_reg_loss:.6f}, state_contrast_loss={avg_state_contrast_loss:.6f}, "
            f"state_lambda_contrast_mean={avg_state_lambda_contrast:.6f}, "
            f"rna_cell_residual_abs_mean={avg_residual_abs_mean:.6f}, "
            f"rna_cell_residual_abs_p95={avg_residual_abs_p95:.6f}, "
            f"prototype_entropy={avg_prototype_entropy:.6f}, "
            f"prototype_usage_min={avg_usage_min:.4f}, prototype_usage_max={avg_usage_max:.4f}, "
            f"latent_state_ce={avg_latent_state_ce:.6f}, "
            f"latent_state_supcon={avg_latent_state_supcon:.6f}, "
            f"latent_state_acc={avg_latent_state_accuracy:.4f}, "
            f"latent_state_weight={active_latent_state_ce_weight:.4f}, "
            f"epoch_seconds={epoch_seconds:.2f}, cells_per_second={cells_per_second:.1f}, "
            f"{gentle_zi.format_weight_summary('rna_nb', dist_params['rna_lambda_1'])}, "
            f"{gentle_zi.format_weight_summary('rna_ps', dist_params['rna_lambda_2'])}"
        )

        if patience > 0 and epochs_no_improve >= patience:
            tqdm.write(f"Early stopping triggered at epoch {epoch + 1}. Best loss: {best_loss:.6f}")
            break
    return history


def compute_state_contrast_report(model, loader, selected_gene_indices, selected_top_states, gene_df, output_dir, device):
    model.eval()
    selected_gene_indices_t = torch.as_tensor(selected_gene_indices, dtype=torch.long, device=device)
    selected_top_states = np.asarray(selected_top_states, dtype=np.int64)
    n_states = int(max(selected_top_states.max() + 1, STATE_CONTRAST_CLUSTERS)) if len(selected_top_states) else STATE_CONTRAST_CLUSTERS
    sums = np.zeros((n_states, len(selected_gene_indices)), dtype=np.float64)
    counts = np.zeros(n_states, dtype=np.int64)
    residual_abs_sum = 0.0
    residual_abs_count = 0
    residual_abs_p95_values = []
    prototype_usages = []
    prototype_entropies = []

    with torch.no_grad():
        for batch in loader:
            rna_batch, atac_batch, _, _, batch_states = batch[:5]
            rna_batch = rna_batch.to(device)
            atac_batch = atac_batch.to(device)
            batch_states_np = batch_states.numpy().astype(np.int64)
            _, _, _, _, _, dist_params = model(rna_batch, atac_batch, decoder_input_mode="shared")
            weights = dist_params["rna_lambda_1"].index_select(1, selected_gene_indices_t).detach().cpu().numpy()
            for state in np.unique(batch_states_np):
                mask = batch_states_np == state
                sums[state] += weights[mask].sum(axis=0)
                counts[state] += int(mask.sum())
            residual = dist_params.get("rna_cell_residual")
            if residual is not None:
                residual_abs = residual.detach().abs()
                residual_abs_sum += float(residual_abs.sum().item())
                residual_abs_count += int(residual_abs.numel())
                residual_abs_p95_values.append(tensor_p95_abs(residual))
            prototype_probs = dist_params.get("rna_prototype_probs")
            if prototype_probs is not None:
                prototype_usages.append(prototype_probs.detach().cpu())
                entropy = -(prototype_probs * torch.log(prototype_probs + 1e-8)).sum(dim=1)
                prototype_entropies.append(entropy.detach().cpu())

    state_means = sums / np.maximum(counts[:, None], 1)
    state_weight_rows = []
    for pos, row in gene_df.reset_index(drop=True).iterrows():
        for state in range(n_states):
            state_weight_rows.append(
                {
                    "raw_index": int(row["raw_index"]),
                    "gene": row["gene"],
                    "state": int(state),
                    "n_cells": int(counts[state]),
                    "state_nb_weight_mean": float(state_means[state, pos]),
                    "assigned_top_state": int(selected_top_states[pos]),
                }
            )
    pd.DataFrame(state_weight_rows).to_csv(
        os.path.join(output_dir, "状态权重均值_按基因和状态.csv"),
        index=False,
    )
    rows = []
    for pos, row in gene_df.reset_index(drop=True).iterrows():
        top_state = int(selected_top_states[pos])
        top_weight = float(state_means[top_state, pos])
        other_states = [state for state in range(n_states) if state != top_state and counts[state] > 0]
        other_mean = float(state_means[other_states, pos].mean()) if other_states else np.nan
        other_max = float(state_means[other_states, pos].max()) if other_states else np.nan
        rows.append(
            {
                "raw_index": int(row["raw_index"]),
                "gene": row["gene"],
                "top_state": top_state,
                "top_state_nb_weight": top_weight,
                "non_top_state_nb_weight_mean": other_mean,
                "non_top_state_nb_weight_max": other_max,
                "top_minus_non_top_mean": top_weight - other_mean,
                "top_is_max_nb_weight": bool(top_weight >= other_max - 1e-12) if not np.isnan(other_max) else True,
                "specificity_score": float(row["specificity_score"]),
                "top_mean": float(row["top_mean"]),
                "second_mean": float(row["second_mean"]),
                "top_detection": float(row["top_detection"]),
                "second_detection": float(row["second_detection"]),
            }
        )
    report_df = pd.DataFrame(rows)
    report_df.to_csv(os.path.join(output_dir, "状态权重对比_按基因.csv"), index=False)

    proto = torch.cat(prototype_usages, dim=0) if prototype_usages else torch.zeros(1, PROTOTYPE_COUNT)
    proto_usage = proto.mean(dim=0)
    proto_entropy = torch.cat(prototype_entropies).mean() if prototype_entropies else torch.tensor(0.0)
    summary = {
        "n_selected_genes": int(len(report_df)),
        "state_lambda_contrast_mean": float(report_df["top_minus_non_top_mean"].mean()),
        "state_lambda_contrast_median": float(report_df["top_minus_non_top_mean"].median()),
        "state_lambda_top_is_max_fraction": float(report_df["top_is_max_nb_weight"].mean()),
        "rna_cell_residual_abs_mean": float(residual_abs_sum / max(residual_abs_count, 1)),
        "rna_cell_residual_abs_p95": float(np.mean(residual_abs_p95_values)) if residual_abs_p95_values else 0.0,
        "prototype_entropy": float(proto_entropy.item()),
        "prototype_usage_min": float(proto_usage.min().item()),
        "prototype_usage_max": float(proto_usage.max().item()),
    }
    with open(os.path.join(output_dir, "状态权重对比汇总.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    top_display = report_df.sort_values("top_minus_non_top_mean", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_display["gene"].iloc[::-1], top_display["top_minus_non_top_mean"].iloc[::-1])
    ax.set_xlabel("Top-state NB weight minus non-top-state mean")
    ax.set_title("State lambda contrast: top selected genes")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "状态混合权重对比_高排名基因_en.png"), dpi=220)
    plt.close(fig)

    with open(os.path.join(output_dir, "状态权重对比报告.md"), "w", encoding="utf-8") as handle:
        handle.write("# 无标签状态混合权重对比\n\n")
        handle.write("## 汇总\n\n")
        for key, value in summary.items():
            handle.write(f"- `{key}`: {value:.6f}" if isinstance(value, float) else f"- `{key}`: {value}")
            handle.write("\n")
        handle.write("\n## 按混合权重对比排序的高排名基因\n\n")
        display = top_display[
            [
                "gene",
                "top_state",
                "top_state_nb_weight",
                "non_top_state_nb_weight_mean",
                "top_minus_non_top_mean",
                "top_is_max_nb_weight",
            ]
        ].copy()
        for column in display.select_dtypes(include=[np.number]).columns:
            display[column] = display[column].map(lambda value: f"{value:.4g}")
        handle.write("| " + " | ".join(display.columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(display.columns)) + " |\n")
        for _, row in display.iterrows():
            handle.write("| " + " | ".join(str(row[column]) for column in display.columns) + " |\n")
    print("State lambda contrast summary:")
    print(json.dumps(summary, indent=2))
    return summary


def analyze_rna_feature_weights(model, rna_adata, output_dir):
    BASE_ANALYZE_RNA_FEATURE_WEIGHTS(model, rna_adata, output_dir)
    localized_names = {
        "rna_feature_weights.csv": "RNA特征权重.csv",
        "rna_feature_weights_high_low_groups.csv": "RNA特征权重_高低表达分组.csv",
        "rna_weight_summary.json": "RNA权重汇总.json",
        "RNA权重直方图.png": "RNA权重直方图_en.png",
        "RNA_NB权重_与表达量关系.png": "RNA_NB权重_与表达量关系_en.png",
        "RNA_NB权重_按表达分组.png": "RNA_NB权重_按表达分组_en.png",
        "RNA_NB权重_高低表达高排名基因.png": "RNA_NB权重_高低表达高排名基因_en.png",
    }
    for source_name, target_name in localized_names.items():
        source = os.path.join(output_dir, source_name)
        target = os.path.join(output_dir, target_name)
        if os.path.exists(source):
            os.replace(source, target)


def save_deterministic_embeddings(
    model,
    loader,
    cell_ids,
    output_dir,
    modality2_name,
    device,
):
    """Export encoder means from the exact preprocessed tensors used in training."""
    model.eval()
    integrated_parts = []
    rna_parts = []
    modality2_parts = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Exporting deterministic embeddings"):
            rna_batch = batch[0].to(device, non_blocking=True)
            modality2_batch = batch[1].to(device, non_blocking=True)
            encoder_outputs = model.encoder(rna_batch, modality2_batch)
            rna_mu = encoder_outputs[3]
            modality2_mu = encoder_outputs[5]
            shared_mu = encoder_outputs[7]
            rna_parts.append(rna_mu.cpu().numpy())
            modality2_parts.append(modality2_mu.cpu().numpy())
            integrated_parts.append(shared_mu.cpu().numpy())

    payload = {
        "integrated": np.vstack(integrated_parts),
        "rna": np.vstack(rna_parts),
        "modality2": np.vstack(modality2_parts),
        "cell_id": np.asarray(cell_ids, dtype=str),
        "modality2_name": np.asarray(str(modality2_name)),
    }
    modality_key = str(modality2_name).strip().lower()
    if modality_key in {"atac", "adt"}:
        payload[modality_key] = payload["modality2"]
    np.savez_compressed(os.path.join(output_dir, "嵌入.npz"), **payload)
    print(f"Saved deterministic embeddings for {len(cell_ids)} cells.")


def parse_args():
    args = gentle_zi.parse_args()
    setattr(args, "variant", "continuous_prior_gentle_zi_state_contrast_residual")
    return args


def configure_globals(args):
    global PROTOTYPE_COUNT, PROTOTYPE_SCALE, PROTOTYPE_PENALTY
    global PROTOTYPE_ENTROPY_WEIGHT, PROTOTYPE_TEMPERATURE, PROTOTYPE_BALANCE_WEIGHT
    global MODALITY2_PROTOTYPE_SCALE, MODALITY2_PROTOTYPE_PENALTY
    global RNA_STATE_MEAN_SCALE, RNA_STATE_MEAN_PENALTY
    global AUXILIARY_STATE_GATE_GRADIENT, STATE_HEAD_SOURCE
    global STATE_GATE_INPUT
    global STATE_CONTRAST_CLUSTERS, STATE_CONTRAST_TOP_GENES, STATE_CONTRAST_WEIGHT, STATE_CONTRAST_MARGIN
    PROTOTYPE_COUNT = args.prototype_count
    PROTOTYPE_SCALE = args.prototype_scale
    PROTOTYPE_PENALTY = args.prototype_penalty
    PROTOTYPE_ENTROPY_WEIGHT = args.prototype_entropy_weight
    PROTOTYPE_TEMPERATURE = args.prototype_temperature
    PROTOTYPE_BALANCE_WEIGHT = args.prototype_balance_weight
    MODALITY2_PROTOTYPE_SCALE = args.modality2_prototype_scale
    MODALITY2_PROTOTYPE_PENALTY = args.modality2_prototype_penalty
    RNA_STATE_MEAN_SCALE = args.rna_state_mean_scale
    RNA_STATE_MEAN_PENALTY = args.rna_state_mean_penalty
    AUXILIARY_STATE_GATE_GRADIENT = args.auxiliary_state_gate_gradient
    STATE_HEAD_SOURCE = args.state_head_source
    STATE_GATE_INPUT = args.state_gate_input
    STATE_CONTRAST_CLUSTERS = args.state_contrast_clusters
    STATE_CONTRAST_TOP_GENES = args.state_contrast_top_genes
    STATE_CONTRAST_WEIGHT = args.state_contrast_weight
    STATE_CONTRAST_MARGIN = args.state_contrast_margin


def main(args):
    global sc, epi
    import scanpy as sc
    import episcanpy.api as epi
    configure_globals(args)
    setup_seed(args.seed)
    run_dir_name = args.run_dir_name or f"随机种子_{args.seed}"
    run_output_dir = os.path.join(args.output_dir, run_dir_name)
    os.makedirs(run_output_dir, exist_ok=True)
    run_config = vars(args).copy()
    run_config["negative_binomial_parameterization"] = (
        gentle_zi.NEGATIVE_BINOMIAL_PARAMETERIZATION
    )
    with open(os.path.join(run_output_dir, "运行配置.json"), "w", encoding="utf-8") as fp:
        json.dump(run_config, fp, indent=2)

    print(f"seed: {args.seed}")
    print("Loading data...")
    rna_adata = sc.read_h5ad(args.rna_h5ad)
    modality2_path = args.modality2_h5ad or args.atac_h5ad
    modality2_name = str(args.modality2_name).upper()
    modality2_adata = sc.read_h5ad(modality2_path)

    if not rna_adata.obs_names.equals(modality2_adata.obs_names):
        raise ValueError("RNA and second-modality cell order must match exactly.")

    rna_adata = rna_adata[:, gentle_zi.feature_nonzero_mask(rna_adata.X)]
    modality2_adata = modality2_adata[
        :,
        gentle_zi.feature_nonzero_mask(modality2_adata.X),
    ]
    rna_mean_expression_raw = gentle_zi.compute_mean_expression(rna_adata.X)
    rna_zero_rate_raw = 1.0 - gentle_zi.compute_detection_rate(rna_adata.X)
    rna_log_mean_expression_raw = np.log10(rna_mean_expression_raw + 1e-8)

    rna_adata.raw = rna_adata.copy()
    modality2_adata.raw = modality2_adata.copy()
    rna_dim = rna_adata.shape[1]
    modality2_dim = modality2_adata.shape[1]

    print("Preprocessing RNA data...")
    sc.pp.normalize_total(rna_adata)
    sc.pp.log1p(rna_adata)
    sc.pp.highly_variable_genes(rna_adata, n_top_genes=args.n_top_genes)
    rna_adata = rna_adata[:, rna_adata.var.highly_variable].copy()

    transform = args.modality2_input_transform
    if transform == "auto":
        transform = args.atac_input_transform
    print(f"Preprocessing {modality2_name} data...")
    if transform == "tfidf":
        epi.pp.filter_features(
            modality2_adata,
            min_cells=np.ceil(modality2_adata.shape[0] * 0.10),
        )
        epi.pp.tfidf(modality2_adata)
        modality2_adata.X = modality2_adata.layers["tf-idf"].copy()
        print(f"{modality2_name} encoder input: TF-IDF")
    elif transform == "clr":
        modality2_adata.X = centered_log_ratio_input(modality2_adata.X)
        print(f"{modality2_name} encoder input: CLR")
    elif transform == "log1p":
        sc.pp.normalize_total(modality2_adata)
        sc.pp.log1p(modality2_adata)
        print(f"{modality2_name} encoder input: library-normalized log1p")
    elif transform == "raw":
        print(f"{modality2_name} encoder input: raw counts")
    else:
        raise ValueError(f"Unsupported second-modality transform: {transform}")

    (
        pseudo_states,
        selected_genes,
        selected_top_states,
        gene_df,
        pseudo_state_targets,
    ) = compute_pseudo_state_targets(
        rna_adata,
        modality2_adata,
        args,
        run_output_dir,
        return_soft_targets=True,
    )
    dataset = StateContrastDataset(
        rna_adata,
        modality2_adata,
        pseudo_states,
        pseudo_state_targets=pseudo_state_targets,
        backend=args.training_data_backend,
    )
    use_pin_memory = bool(args.pin_memory and str(args.device).startswith("cuda"))
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_pin_memory,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=use_pin_memory,
    )

    rna_pro_dim = rna_adata.shape[1]
    modality2_pro_dim = modality2_adata.shape[1]
    print(f"RNA dimensions: {rna_dim}, {modality2_name} dimensions: {modality2_dim}")
    print(
        f"RNA processing dimensions: {rna_pro_dim}, "
        f"{modality2_name} processing dimensions: {modality2_pro_dim}"
    )

    model = scAPIFeatureLambda(
        rna_dim,
        modality2_dim,
        rna_pro_dim,
        modality2_pro_dim,
        rna_log_mean_expression_raw,
        rna_zero_rate_raw,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout_rate=args.dropout_rate,
    )
    print(f"Using device: {args.device}")
    model.to(args.device)

    if args.init_model_path and not args.analyze_only:
        print(f"Warm-starting from checkpoint: {args.init_model_path}")
        state_dict = torch.load(args.init_model_path, map_location=args.device)
        incompatible = model.load_state_dict(state_dict, strict=False)
        print(f"  missing_keys: {list(incompatible.missing_keys)}")
        print(f"  unexpected_keys: {list(incompatible.unexpected_keys)}")

    if args.analyze_only:
        model_path = args.model_path or os.path.join(
            run_output_dir,
            f"模型_随机种子_{args.seed}.pt",
        )
        print(f"Loading checkpoint for analysis only: {model_path}")
        state_dict = torch.load(model_path, map_location=args.device)
        model.load_state_dict(state_dict)
        model.eval()
        save_deterministic_embeddings(
            model,
            eval_loader,
            rna_adata.obs_names,
            run_output_dir,
            modality2_name,
            args.device,
        )
        analyze_rna_feature_weights(model, rna_adata, run_output_dir)
        compute_state_contrast_report(
            model,
            eval_loader,
            selected_genes,
            selected_top_states,
            gene_df,
            run_output_dir,
            args.device,
        )
        print(f"Saved analysis outputs to {run_output_dir}")
        return

    special_param_ids = {
        id(model.decoder.atac_logits),
        id(model.decoder.rna_prior_bias),
        id(model.decoder.rna_prior_feature_weights),
        id(model.decoder.rna_gene_residual),
    }
    main_params = [p for p in model.parameters() if id(p) not in special_param_ids]
    optimizer = torch.optim.Adam(
        [
            {"params": main_params},
            {"params": [model.decoder.atac_logits], "lr": 1e-4, "weight_decay": 1e-5},
            {
                "params": [model.decoder.rna_prior_bias, model.decoder.rna_prior_feature_weights],
                "lr": args.prior_lr,
                "weight_decay": 1e-5,
            },
            {"params": [model.decoder.rna_gene_residual], "lr": args.gene_residual_lr, "weight_decay": 0.0},
        ],
        lr=args.main_lr,
    )

    try:
        checkpoint_epochs = {
            int(value)
            for value in args.checkpoint_epochs.split(",")
            if value.strip()
        }
        train_model(
            model,
            train_loader,
            optimizer,
            selected_genes,
            selected_top_states,
            num_epochs=args.num_epochs,
            warmup_epochs=args.warmup_epochs,
            device=args.device,
            prior_reg_weight=args.prior_reg_weight,
            state_contrast_weight=args.state_contrast_weight,
            state_contrast_margin=args.state_contrast_margin,
            early_stopping_patience=args.early_stopping_patience,
            checkpoint_epochs=checkpoint_epochs,
            checkpoint_dir=run_output_dir,
            seed=args.seed,
            materialize_decoder_samples=args.materialize_decoder_samples,
            residual_p95_interval=args.residual_p95_interval,
            pin_memory=use_pin_memory,
            max_batches_per_epoch=args.max_batches_per_epoch,
            latent_state_ce_weight=args.latent_state_ce_weight,
            latent_state_label_smoothing=args.latent_state_label_smoothing,
            latent_state_ramp_epochs=args.latent_state_ramp_epochs,
            latent_state_supcon_weight=args.latent_state_supcon_weight,
            latent_state_temperature=args.latent_state_temperature,
            latent_state_start_epoch=args.latent_state_start_epoch,
            latent_state_supcon_start_epoch=args.latent_state_supcon_start_epoch,
            modality2_name=modality2_name,
        )
        if not args.skip_checkpoint_save:
            torch.save(
                model.state_dict(),
                os.path.join(run_output_dir, f"模型_随机种子_{args.seed}.pt"),
            )
        save_deterministic_embeddings(
            model,
            eval_loader,
            rna_adata.obs_names,
            run_output_dir,
            modality2_name,
            args.device,
        )
        if args.skip_post_training_analysis:
            print(f"Completed profiling run in {run_output_dir}")
            return
        analyze_rna_feature_weights(model, rna_adata, run_output_dir)
        compute_state_contrast_report(
            model,
            eval_loader,
            selected_genes,
            selected_top_states,
            gene_df,
            run_output_dir,
            args.device,
        )
        print(f"Saved outputs to {run_output_dir}")
    except Exception as exc:
        print("Error during training:", exc)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main(parse_args())
