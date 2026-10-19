import argparse
import json
import os
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
from torch.distributions import Poisson

# Scanpy/episcanpy are needed only when this legacy standalone CLI performs
# preprocessing.  Avoid paying their import cost for model-only consumers.
sc = None
epi = None

from .encoder import MultiModalDataset, compute_kld, compute_mmd, encoder
from .utils import AdversarialAlignmentModule, setup_seed
from .distributions import (
    NEGATIVE_BINOMIAL_PARAMETERIZATION,
    negative_binomial_from_mean_dispersion,
)


def make_decoder_head(latent_dim, hidden_dim, output_dim, dropout_rate, final_activation="softplus"):
    layers = [
        nn.Linear(latent_dim, hidden_dim // 2),
        nn.BatchNorm1d(hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim // 2, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim, output_dim),
    ]
    if final_activation == "softplus":
        layers.append(nn.Softplus())
    elif final_activation != "identity":
        raise ValueError(f"Unsupported final activation: {final_activation}")
    return nn.Sequential(*layers)


def zero_inflated_log_prob(base_dist, value, zi_logits):
    zi_prob = torch.sigmoid(zi_logits)
    zi_prob = torch.clamp(zi_prob, min=1e-6, max=1 - 1e-6)
    log_zi = torch.log(zi_prob)
    log_one_minus_zi = torch.log1p(-zi_prob)
    zero_value = torch.zeros_like(value)
    base_zero_logprob = base_dist.log_prob(zero_value)
    zero_logprob = torch.logsumexp(
        torch.stack([log_zi, log_one_minus_zi + base_zero_logprob], dim=0),
        dim=0,
    )
    nonzero_logprob = log_one_minus_zi + base_dist.log_prob(value)
    return torch.where(value == 0, zero_logprob, nonzero_logprob)


def sample_zero_inflated(base_dist, zi_logits):
    base_sample = base_dist.sample()
    zi_prob = torch.sigmoid(zi_logits)
    zero_mask = torch.rand_like(base_sample.float()) < zi_prob
    return torch.where(zero_mask, torch.zeros_like(base_sample), base_sample)


class FeatureWiseMixtureDecoder(nn.Module):
    def __init__(
        self,
        rna_dim,
        atac_dim,
        rna_log_mean_expression,
        rna_zero_rate,
        hidden_dim=512,
        latent_dim=128,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.rna_decoder_mean = make_decoder_head(latent_dim, hidden_dim, rna_dim, dropout_rate, final_activation="softplus")
        self.rna_decoder_dispersion = make_decoder_head(latent_dim, hidden_dim, rna_dim, dropout_rate, final_activation="softplus")
        self.rna_decoder_poisson = make_decoder_head(latent_dim, hidden_dim, rna_dim, dropout_rate, final_activation="softplus")
        self.rna_decoder_nb_zero_logits = make_decoder_head(latent_dim, hidden_dim, rna_dim, dropout_rate, final_activation="identity")
        self.rna_decoder_ps_zero_logits = make_decoder_head(latent_dim, hidden_dim, rna_dim, dropout_rate, final_activation="identity")

        self.atac_decoder_mean = make_decoder_head(latent_dim, hidden_dim, atac_dim, dropout_rate, final_activation="softplus")
        self.atac_decoder_dispersion = make_decoder_head(latent_dim, hidden_dim, atac_dim, dropout_rate, final_activation="softplus")
        self.atac_decoder_poisson = make_decoder_head(latent_dim, hidden_dim, atac_dim, dropout_rate, final_activation="softplus")
        self.atac_decoder_nb_zero_logits = make_decoder_head(latent_dim, hidden_dim, atac_dim, dropout_rate, final_activation="identity")
        self.atac_decoder_ps_zero_logits = make_decoder_head(latent_dim, hidden_dim, atac_dim, dropout_rate, final_activation="identity")

        # RNA mixture prior is a continuous function of gene-level summary statistics.
        # We standardize the inputs so the learned coefficients remain interpretable and stable.
        log_mean_expression = np.asarray(rna_log_mean_expression, dtype=np.float32)
        zero_rate = np.asarray(rna_zero_rate, dtype=np.float32)
        log_mean_std = np.maximum(log_mean_expression.std(), 1e-6)
        zero_rate_std = np.maximum(zero_rate.std(), 1e-6)
        standardized_log_mean = (log_mean_expression - log_mean_expression.mean()) / log_mean_std
        standardized_zero_rate = (zero_rate - zero_rate.mean()) / zero_rate_std
        prior_features = np.stack([standardized_log_mean, standardized_zero_rate], axis=1)

        expr_rank = pd.Series(log_mean_expression).rank(method="average", pct=True).to_numpy()
        zero_rank = pd.Series(zero_rate).rank(method="average", pct=True).to_numpy()
        low_information_score = np.clip(0.75 * (1.0 - expr_rank) + 0.25 * zero_rank, 0.0, 1.0)

        # A gentler shrinkage schedule: low-information genes still borrow from the
        # continuous prior, but the residual has much more room to respond to data.
        prior_strength = 0.02 + 0.18 * np.power(low_information_score, 2.0)

        self.register_buffer("rna_prior_features", torch.tensor(prior_features, dtype=torch.float32))
        self.register_buffer("rna_log_mean_expression", torch.tensor(log_mean_expression, dtype=torch.float32))
        self.register_buffer("rna_zero_rate", torch.tensor(zero_rate, dtype=torch.float32))
        self.register_buffer("rna_prior_strength", torch.tensor(prior_strength, dtype=torch.float32))
        self.register_buffer("rna_low_information_score", torch.tensor(low_information_score, dtype=torch.float32))

        self.rna_prior_bias = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.rna_prior_feature_weights = nn.Parameter(torch.tensor([[0.30, -0.05]], dtype=torch.float32))
        self.rna_gene_residual = nn.Parameter(torch.zeros(rna_dim))
        self.atac_logits = nn.Parameter(torch.zeros(2, atac_dim))

    def get_rna_prior_logit(self):
        return self.rna_prior_bias + torch.matmul(self.rna_prior_features, self.rna_prior_feature_weights.t()).squeeze(-1)

    def get_rna_prior_nb_weight(self):
        return torch.sigmoid(self.get_rna_prior_logit())

    def get_rna_logits(self):
        rna_nb_logit = self.get_rna_prior_logit() + self.rna_gene_residual
        return torch.stack([rna_nb_logit, torch.zeros_like(rna_nb_logit)], dim=0)

    def prior_regularization(self):
        weighted_gene_residual = self.rna_gene_residual.pow(2) * self.rna_prior_strength
        bias_penalty = 0.005 * self.rna_prior_bias.pow(2).mean()
        log_mean_penalty = 0.01 * self.rna_prior_feature_weights[:, 0].pow(2).mean()
        zero_rate_penalty = 0.10 * self.rna_prior_feature_weights[:, 1].pow(2).mean()
        gene_penalty = weighted_gene_residual.mean()
        return bias_penalty + log_mean_penalty + zero_rate_penalty + gene_penalty

    def forward(self, z, sample=True):
        rna_mean = self.rna_decoder_mean(z) + 1e-6
        rna_dispersion = self.rna_decoder_dispersion(z) + 1e-6
        rna_poisson = self.rna_decoder_poisson(z) + 1e-6
        rna_nb_zero_logits = self.rna_decoder_nb_zero_logits(z)
        rna_ps_zero_logits = self.rna_decoder_ps_zero_logits(z)

        atac_mean = self.atac_decoder_mean(z) + 1e-6
        atac_dispersion = self.atac_decoder_dispersion(z) + 1e-6
        atac_poisson = self.atac_decoder_poisson(z) + 1e-6
        atac_nb_zero_logits = self.atac_decoder_nb_zero_logits(z)
        atac_ps_zero_logits = self.atac_decoder_ps_zero_logits(z)

        rna_weights = torch.softmax(self.get_rna_logits(), dim=0)
        atac_weights = torch.softmax(self.atac_logits, dim=0)
        rna_lambda_1, rna_lambda_2 = rna_weights[0], rna_weights[1]
        atac_lambda_1, atac_lambda_2 = atac_weights[0], atac_weights[1]

        rna_nb_dist = negative_binomial_from_mean_dispersion(rna_mean, rna_dispersion)
        rna_ps_dist = Poisson(rate=rna_poisson)

        atac_nb_dist = negative_binomial_from_mean_dispersion(atac_mean, atac_dispersion)
        atac_ps_dist = Poisson(rate=atac_poisson)

        if sample:
            rna_data = rna_lambda_1 * sample_zero_inflated(rna_nb_dist, rna_nb_zero_logits) + rna_lambda_2 * sample_zero_inflated(rna_ps_dist, rna_ps_zero_logits)
            atac_data = atac_lambda_1 * sample_zero_inflated(atac_nb_dist, atac_nb_zero_logits) + atac_lambda_2 * sample_zero_inflated(atac_ps_dist, atac_ps_zero_logits)
        else:
            rna_data = rna_mean
            atac_data = atac_mean

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
            "mixture_prior_reg": self.prior_regularization(),
            "rna_prior_nb_weight": self.get_rna_prior_nb_weight(),
            "rna_prior_strength": self.rna_prior_strength,
        }
        return rna_data, atac_data, outputs


def format_weight_summary(name, weights):
    weights_cpu = weights.detach().cpu()
    return (
        f"{name}_mean={weights_cpu.mean().item():.3f}, "
        f"{name}_min={weights_cpu.min().item():.3f}, "
        f"{name}_max={weights_cpu.max().item():.3f}"
    )


class scAPIFeatureLambda(nn.Module):
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
        super().__init__()
        self.encoder = encoder(rna_dim, atac_dim, rna_pro_dim, atac_pro_dim, hidden_dim, latent_dim, dropout_rate)
        self.decoder = FeatureWiseMixtureDecoder(
            rna_dim,
            atac_dim,
            rna_log_mean_expression,
            rna_zero_rate,
            hidden_dim,
            latent_dim,
            dropout_rate,
        )
        self.adversarial_module = AdversarialAlignmentModule(
            latent_dim=latent_dim,
            hidden_dims=[256, 128],
            lambda_grl=1.0,
            discriminator_lr=1e-4,
        )

    def forward(self, rna_data, atac_data, decoder_input_mode="shared"):
        z_rna, z_atac, z_shared, rna_mu, rna_var, atac_mu, atac_var, shared_mu, shared_var = self.encoder(rna_data, atac_data)

        if decoder_input_mode == "shared":
            decoder_input = z_shared
        elif decoder_input_mode == "rna":
            decoder_input = z_rna
        elif decoder_input_mode == "atac":
            decoder_input = z_atac
        else:
            raise ValueError(f"Invalid decoder_input_mode: {decoder_input_mode}")

        rna_data_out, atac_data_out, dist_params = self.decoder(decoder_input)
        adv_results = self.adversarial_module(z_shared, z_rna, z_atac, mode="adversarial")

        dist_params["rna_mu"] = rna_mu
        dist_params["rna_var"] = rna_var
        dist_params["atac_mu"] = atac_mu
        dist_params["atac_var"] = atac_var
        dist_params["shared_mu"] = shared_mu
        dist_params["shared_var"] = shared_var
        dist_params["adv_loss"] = adv_results["adversarial_loss"]

        return rna_data_out, atac_data_out, z_shared, z_rna, z_atac, dist_params


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
):
    rna_data = rna_data.round().long()
    atac_data = atac_data.round().long()

    rna_nb_logprob = zero_inflated_log_prob(dist_params["rna_nb_dist"], rna_data, dist_params["rna_nb_zero_logits"])
    rna_ps_logprob = zero_inflated_log_prob(dist_params["rna_ps_dist"], rna_data, dist_params["rna_ps_zero_logits"])
    atac_nb_logprob = zero_inflated_log_prob(dist_params["atac_nb_dist"], atac_data, dist_params["atac_nb_zero_logits"])
    atac_ps_logprob = zero_inflated_log_prob(dist_params["atac_bnl_dist"], atac_data, dist_params["atac_ps_zero_logits"])

    rna_log_lambda_1 = torch.log(dist_params["rna_lambda_1"].unsqueeze(0) + 1e-8)
    rna_log_lambda_2 = torch.log(dist_params["rna_lambda_2"].unsqueeze(0) + 1e-8)
    atac_log_lambda_1 = torch.log(dist_params["atac_lambda_1"].unsqueeze(0) + 1e-8)
    atac_log_lambda_2 = torch.log(dist_params["atac_lambda_2"].unsqueeze(0) + 1e-8)

    rna_log_likelihood = torch.logsumexp(
        torch.stack([rna_log_lambda_1 + rna_nb_logprob, rna_log_lambda_2 + rna_ps_logprob], dim=0),
        dim=0,
    )
    atac_log_likelihood = torch.logsumexp(
        torch.stack([atac_log_lambda_1 + atac_nb_logprob, atac_log_lambda_2 + atac_ps_logprob], dim=0),
        dim=0,
    )

    rna_recon_loss = -rna_log_likelihood.mean()
    atac_recon_loss = -atac_log_likelihood.mean()

    kl_loss = compute_kld(
        dist_params["shared_mu"],
        dist_params["shared_var"],
        dist_params["rna_mu"],
        dist_params["rna_var"],
        dist_params["atac_mu"],
        dist_params["atac_var"],
    ) * alpha

    shared_reg_loss = 0
    if z_shared is not None:
        shared_rna_loss = compute_mmd(z_shared, rna_z) * beta
        shared_atac_loss = compute_mmd(z_shared, atac_z) * beta
        shared_reg_loss = shared_rna_loss + shared_atac_loss

    adv_loss = dist_params["adv_loss"] * gamma
    prior_reg_loss = dist_params.get("mixture_prior_reg", torch.tensor(0.0, device=rna_recon_loss.device)) * prior_reg_weight
    total_loss = rna_recon_loss + atac_recon_loss + shared_reg_loss + kl_loss + adv_loss + prior_reg_loss

    return {
        "total_loss": total_loss,
        "rna_recon_loss": rna_recon_loss,
        "atac_recon_loss": atac_recon_loss,
        "kl_loss": kl_loss,
        "adv_loss": adv_loss,
        "shared_reg_loss": shared_reg_loss,
        "prior_reg_loss": prior_reg_loss,
    }


def train_model(model, train_loader, optimizer, num_epochs=80, warmup_epochs=20, device="cuda", prior_reg_weight=2e-3):
    model.train()
    patience = 30
    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_batches = 0
        rna_loss_sum = 0.0
        atac_loss_sum = 0.0
        kl_loss_sum = 0.0
        adv_loss_sum = 0.0
        shared_reg_loss_sum = 0.0
        prior_reg_loss_sum = 0.0

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

        for batch_idx, (rna_batch, atac_batch, rna_raw, atac_raw) in enumerate(train_loader):
            rna_batch = rna_batch.to(device)
            atac_batch = atac_batch.to(device)
            rna_raw = rna_raw.to(device)
            atac_raw = atac_raw.to(device)

            if batch_idx % 5 == 0:
                with torch.no_grad():
                    z_rna, z_atac, z_shared, *_ = model.encoder(rna_batch, atac_batch)
                model.adversarial_module.train_discriminator_step(z_shared, z_rna, z_atac)

            optimizer.zero_grad()
            _, _, z_shared, z_rna, z_atac, dist_params = model(
                rna_batch, atac_batch, decoder_input_mode=decoder_input_mode
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
            )

            loss = losses["total_loss"]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            rna_loss_sum += losses["rna_recon_loss"].item()
            atac_loss_sum += losses["atac_recon_loss"].item()
            kl_loss_sum += losses["kl_loss"].item()
            adv_loss_sum += losses["adv_loss"].item()
            shared_reg_loss_sum += losses["shared_reg_loss"].item()
            prior_reg_loss_sum += losses["prior_reg_loss"].item()
            total_batches += 1

        if total_batches == 0:
            continue

        avg_loss = total_loss / total_batches
        avg_rna_loss = rna_loss_sum / total_batches
        avg_atac_loss = atac_loss_sum / total_batches
        avg_kl_loss = kl_loss_sum / total_batches
        avg_adv_loss = adv_loss_sum / total_batches
        avg_shared_reg_loss = shared_reg_loss_sum / total_batches
        avg_prior_reg_loss = prior_reg_loss_sum / total_batches

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch + 1}/{num_epochs}, mode={decoder_input_mode}, "
            f"TL={avg_loss:.6f}, RNA={avg_rna_loss:.6f}, ATAC={avg_atac_loss:.6f}, "
            f"KL={avg_kl_loss:.6f}, ADV={avg_adv_loss:.6f}, Shared={avg_shared_reg_loss:.6f}, Prior={avg_prior_reg_loss:.6f}, "
            f"{format_weight_summary('rna_nb', dist_params['rna_lambda_1'])}, "
            f"{format_weight_summary('rna_ps', dist_params['rna_lambda_2'])}"
        )

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}. Best loss: {best_loss:.6f}")
            break


def feature_nonzero_mask(matrix_like):
    if scipy.sparse.issparse(matrix_like):
        return np.asarray(matrix_like.sum(axis=0)).ravel() > 0
    return np.sum(matrix_like, axis=0) > 0


def vector_from_axis_op(matrix_like):
    if scipy.sparse.issparse(matrix_like):
        return np.asarray(matrix_like).ravel()
    return np.asarray(matrix_like).ravel()


def compute_mean_expression(matrix_like):
    return vector_from_axis_op(matrix_like.mean(axis=0))


def compute_detection_rate(matrix_like):
    if scipy.sparse.issparse(matrix_like):
        return vector_from_axis_op((matrix_like > 0).mean(axis=0))
    return np.mean(matrix_like > 0, axis=0)


def analyze_rna_feature_weights(model, rna_adata, output_dir):
    from scipy.stats import spearmanr

    os.makedirs(output_dir, exist_ok=True)

    rna_logits = model.decoder.get_rna_logits().detach().cpu()
    rna_weights = torch.softmax(rna_logits, dim=0).numpy()
    rna_nb_weight = rna_weights[0]
    rna_ps_weight = rna_weights[1]

    raw_matrix = rna_adata.raw.X if rna_adata.raw is not None else rna_adata.X
    gene_names = np.asarray(rna_adata.raw.var_names if rna_adata.raw is not None else rna_adata.var_names).astype(str)
    mean_expression = compute_mean_expression(raw_matrix)
    detection_rate = compute_detection_rate(raw_matrix)
    log_mean_expression = np.log10(mean_expression + 1e-8)

    df = pd.DataFrame(
        {
            "gene": gene_names,
            "mean_expression": mean_expression,
            "log10_mean_expression": log_mean_expression,
            "detection_rate": detection_rate,
            "zero_rate": 1.0 - detection_rate,
            "rna_nb_weight": rna_nb_weight,
            "rna_ps_weight": rna_ps_weight,
        }
    )
    if hasattr(model.decoder, "get_rna_prior_nb_weight"):
        df["rna_prior_nb_weight"] = model.decoder.get_rna_prior_nb_weight().detach().cpu().numpy()
        df["rna_prior_strength"] = model.decoder.rna_prior_strength.detach().cpu().numpy()
        df["rna_low_information_score"] = model.decoder.rna_low_information_score.detach().cpu().numpy()

    low_cutoff = df["mean_expression"].quantile(0.10)
    high_cutoff = df["mean_expression"].quantile(0.90)
    df["expression_group"] = "mid"
    df.loc[df["mean_expression"] <= low_cutoff, "expression_group"] = "low"
    df.loc[df["mean_expression"] >= high_cutoff, "expression_group"] = "high"

    spearman_corr, spearman_pvalue = spearmanr(df["mean_expression"], df["rna_nb_weight"])
    pearson_corr = float(np.corrcoef(df["log10_mean_expression"], df["rna_nb_weight"])[0, 1])

    summary = {
        "num_genes": int(df.shape[0]),
        "rna_nb_weight_mean": float(df["rna_nb_weight"].mean()),
        "rna_nb_weight_std": float(df["rna_nb_weight"].std()),
        "rna_nb_weight_min": float(df["rna_nb_weight"].min()),
        "rna_nb_weight_max": float(df["rna_nb_weight"].max()),
        "rna_ps_weight_mean": float(df["rna_ps_weight"].mean()),
        "low_expression_mean_nb_weight": float(df.loc[df["expression_group"] == "low", "rna_nb_weight"].mean()),
        "mid_expression_mean_nb_weight": float(df.loc[df["expression_group"] == "mid", "rna_nb_weight"].mean()),
        "high_expression_mean_nb_weight": float(df.loc[df["expression_group"] == "high", "rna_nb_weight"].mean()),
        "low_expression_mean_ps_weight": float(df.loc[df["expression_group"] == "low", "rna_ps_weight"].mean()),
        "high_expression_mean_ps_weight": float(df.loc[df["expression_group"] == "high", "rna_ps_weight"].mean()),
        "spearman_corr_mean_expression_vs_nb_weight": float(spearman_corr),
        "spearman_pvalue_mean_expression_vs_nb_weight": float(spearman_pvalue),
        "pearson_corr_log_expression_vs_nb_weight": pearson_corr,
    }
    if "rna_prior_nb_weight" in df.columns:
        summary["low_expression_mean_prior_nb_weight"] = float(df.loc[df["expression_group"] == "low", "rna_prior_nb_weight"].mean())
        summary["high_expression_mean_prior_nb_weight"] = float(df.loc[df["expression_group"] == "high", "rna_prior_nb_weight"].mean())
        summary["low_expression_mean_prior_strength"] = float(df.loc[df["expression_group"] == "low", "rna_prior_strength"].mean())
        summary["high_expression_mean_prior_strength"] = float(df.loc[df["expression_group"] == "high", "rna_prior_strength"].mean())
    if hasattr(model.decoder, "rna_decoder_nb_zero_logits"):
        decoder_was_training = model.decoder.training
        model.decoder.eval()
        with torch.no_grad():
            outputs = model.decoder(
                torch.zeros(1, model.decoder.latent_dim, device=next(model.parameters()).device),
                sample=False,
            )[2]
        if decoder_was_training:
            model.decoder.train()
        summary["rna_nb_zero_rate_mean"] = float(outputs["rna_nb_zero_rate"].detach().cpu().mean())
        summary["rna_ps_zero_rate_mean"] = float(outputs["rna_ps_zero_rate"].detach().cpu().mean())
    if hasattr(model.decoder, "rna_prior_bias"):
        summary["prior_bias"] = float(model.decoder.rna_prior_bias.detach().cpu().item())
        summary["prior_log_mean_coef"] = float(model.decoder.rna_prior_feature_weights.detach().cpu()[0, 0].item())
        summary["prior_zero_rate_coef"] = float(model.decoder.rna_prior_feature_weights.detach().cpu()[0, 1].item())

    df.sort_values("mean_expression", ascending=False).to_csv(
        os.path.join(output_dir, "rna_feature_weights.csv"),
        index=False,
    )
    df[df["expression_group"].isin(["low", "high"])].sort_values(
        ["expression_group", "mean_expression"], ascending=[True, False]
    ).to_csv(
        os.path.join(output_dir, "rna_feature_weights_high_low_groups.csv"),
        index=False,
    )

    with open(os.path.join(output_dir, "rna_weight_summary.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["rna_nb_weight"], bins=50, color="#1f77b4", alpha=0.85)
    axes[0].set_title("RNA ZINB Weight Distribution")
    axes[0].set_xlabel("ZINB weight")
    axes[0].set_ylabel("Gene count")
    axes[1].hist(df["rna_ps_weight"], bins=50, color="#ff7f0e", alpha=0.85)
    axes[1].set_title("RNA ZIP Weight Distribution")
    axes[1].set_xlabel("ZIP weight")
    axes[1].set_ylabel("Gene count")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "RNA权重直方图.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["log10_mean_expression"], df["rna_nb_weight"], s=8, alpha=0.25, color="#1f77b4")
    ax.set_title("RNA ZINB Weight vs Mean Gene Expression")
    ax.set_xlabel("log10(mean raw expression + 1e-8)")
    ax.set_ylabel("NB weight")
    ax.text(
        0.03,
        0.97,
        f"Spearman: {summary['spearman_corr_mean_expression_vs_nb_weight']:.3f}\n"
        f"Pearson(log): {summary['pearson_corr_log_expression_vs_nb_weight']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "RNA_NB权重_与表达量关系.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    box_data = [
        df.loc[df["expression_group"] == "low", "rna_nb_weight"].values,
        df.loc[df["expression_group"] == "mid", "rna_nb_weight"].values,
        df.loc[df["expression_group"] == "high", "rna_nb_weight"].values,
    ]
    ax.boxplot(box_data, labels=["Low expr", "Mid expr", "High expr"], showfliers=False)
    ax.set_title("RNA ZINB Weights by Expression Group")
    ax.set_ylabel("ZINB weight")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "RNA_NB权重_按表达分组.png"), dpi=200)
    plt.close(fig)

    top_high = df.sort_values("mean_expression", ascending=False).head(20).iloc[::-1]
    top_low = df.sort_values("mean_expression", ascending=True).head(20)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    axes[0].barh(top_high["gene"], top_high["rna_nb_weight"], color="#2ca02c")
    axes[0].set_title("Top 20 High-Expression Genes")
    axes[0].set_xlabel("ZINB weight")
    axes[1].barh(top_low["gene"], top_low["rna_nb_weight"], color="#d62728")
    axes[1].set_title("Top 20 Low-Expression Genes")
    axes[1].set_xlabel("ZINB weight")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "RNA_NB权重_高低表达高排名基因.png"), dpi=200)
    plt.close(fig)

    print("RNA feature-weight summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Feature-wise lambda variant with a gentler continuous RNA prior and ZINB/ZIP components")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--rna_h5ad",
        type=str,
        default="/data/hzb/dataset/10XMultiomics_pbmc_10k_granulocyte_plus_1_rna.h5ad",
    )
    parser.add_argument(
        "--atac_h5ad",
        type=str,
        default="/data/hzb/dataset/10XMultiomics_pbmc_10k_granulocyte_plus_1_atac.h5ad",
    )
    parser.add_argument(
        "--modality2_h5ad",
        type=str,
        default=None,
        help="Optional generic second-modality input; overrides --atac_h5ad.",
    )
    parser.add_argument(
        "--modality2_name",
        type=str,
        default="ATAC",
        help="Display name for the second modality, for example ATAC or ADT.",
    )
    parser.add_argument(
        "--modality2_input_transform",
        choices=["auto", "raw", "tfidf", "log1p", "clr"],
        default="auto",
        help=(
            "Encoder transform for the second modality. 'auto' reuses "
            "--atac_input_transform for backward compatibility."
        ),
    )
    parser.add_argument(
        "--run_dir_name",
        type=str,
        default=None,
        help="Optional result subdirectory name; defaults to a Chinese seed name.",
    )
    parser.add_argument("--output_dir", type=str, default="feature_lambda_runs/pbmc_10k")
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=30,
        help="Epochs without improvement before stopping; set to 0 to disable.",
    )
    parser.add_argument(
        "--checkpoint_epochs",
        type=str,
        default="",
        help="Comma-separated 1-based epochs at which to save model checkpoints.",
    )
    parser.add_argument(
        "--atac_input_transform",
        choices=["raw", "tfidf"],
        default="raw",
        help="ATAC encoder input after feature filtering. Raw counts remain the decoder target.",
    )
    parser.add_argument(
        "--training_data_backend",
        choices=["legacy", "tensor"],
        default="legacy",
        help="Use prebuilt CPU tensors or legacy per-sample NumPy-to-tensor conversion.",
    )
    parser.add_argument(
        "--materialize_decoder_samples",
        action="store_true",
        help="Generate decoder samples even though the training loss does not consume them.",
    )
    parser.add_argument(
        "--residual_p95_interval",
        type=int,
        default=1,
        help="Compute the expensive residual p95 diagnostic every N batches; 0 disables it.",
    )
    pin_memory_group = parser.add_mutually_exclusive_group()
    pin_memory_group.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        help="Pin DataLoader batches when CUDA is used.",
    )
    pin_memory_group.add_argument(
        "--no_pin_memory",
        dest="pin_memory",
        action="store_false",
        help="Disable pinned DataLoader batches.",
    )
    parser.set_defaults(pin_memory=False)
    parser.add_argument(
        "--max_batches_per_epoch",
        type=int,
        default=0,
        help="Limit batches per epoch for profiling; 0 uses the full dataset.",
    )
    parser.add_argument(
        "--skip_post_training_analysis",
        action="store_true",
        help="Skip checkpoint analysis for short profiling runs.",
    )
    parser.add_argument(
        "--skip_checkpoint_save",
        action="store_true",
        help="Do not save a model checkpoint for disposable profiling runs.",
    )
    parser.add_argument("--latent_state_ce_weight", type=float, default=0.0)
    parser.add_argument(
        "--latent_state_label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for latent pseudo-state cross-entropy.",
    )
    parser.add_argument(
        "--latent_state_soft_target_max_prob",
        type=float,
        default=1.0,
        help=(
            "Target mean maximum probability for distance-aware KMeans soft "
            "targets; 1.0 retains hard pseudo-state labels."
        ),
    )
    parser.add_argument(
        "--latent_state_ramp_epochs",
        type=int,
        default=0,
        help="Linearly ramp latent pseudo-state CE weight over this many epochs.",
    )
    parser.add_argument("--latent_state_supcon_weight", type=float, default=0.0)
    parser.add_argument("--latent_state_temperature", type=float, default=0.20)
    parser.add_argument(
        "--latent_state_start_epoch",
        type=int,
        default=0,
        help="Zero-based epoch at which latent pseudo-state cross-entropy becomes active.",
    )
    parser.add_argument(
        "--latent_state_supcon_start_epoch",
        type=int,
        default=-1,
        help="Zero-based epoch for pseudo-state contrastive loss; negative reuses latent_state_start_epoch.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_top_genes", type=int, default=3000)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--prior_reg_weight", type=float, default=2e-3)
    parser.add_argument("--main_lr", type=float, default=1e-3)
    parser.add_argument("--prior_lr", type=float, default=5e-5)
    parser.add_argument("--gene_residual_lr", type=float, default=5e-4)
    parser.add_argument("--init_model_path", type=str, default=None)
    parser.add_argument("--cell_residual_rank", type=int, default=8)
    parser.add_argument("--cell_residual_scale", type=float, default=0.35)
    parser.add_argument("--cell_residual_penalty", type=float, default=0.50)
    parser.add_argument("--prototype_count", type=int, default=12)
    parser.add_argument("--prototype_scale", type=float, default=0.45)
    parser.add_argument("--prototype_penalty", type=float, default=0.30)
    parser.add_argument("--prototype_entropy_weight", type=float, default=0.005)
    parser.add_argument("--prototype_temperature", type=float, default=0.70)
    parser.add_argument("--prototype_balance_weight", type=float, default=0.05)
    parser.add_argument(
        "--modality2_prototype_scale",
        type=float,
        default=0.0,
        help="Maximum state residual applied to second-modality mixture logits.",
    )
    parser.add_argument("--modality2_prototype_penalty", type=float, default=0.08)
    parser.add_argument(
        "--rna_state_mean_scale",
        type=float,
        default=0.0,
        help="Maximum log-scale state residual applied to both RNA component means.",
    )
    parser.add_argument("--rna_state_mean_penalty", type=float, default=0.08)
    parser.add_argument(
        "--auxiliary_state_gate_gradient",
        action="store_true",
        help="Allow second-modality and RNA-mean residuals to update the state gate.",
    )
    parser.add_argument(
        "--state_head_source",
        choices=("classifier", "prototype"),
        default="classifier",
        help="Head used by pseudo-state cross-entropy.",
    )
    parser.add_argument(
        "--state_gate_input",
        choices=("decoder", "shared_mean"),
        default="decoder",
        help="Latent input used by prototype-gated state residuals.",
    )
    parser.add_argument("--state_contrast_clusters", type=int, default=12)
    parser.add_argument("--state_contrast_top_genes", type=int, default=1024)
    parser.add_argument("--state_contrast_weight", type=float, default=0.02)
    parser.add_argument("--state_contrast_margin", type=float, default=0.03)
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    return parser.parse_args()


def main(args):
    global sc, epi
    import scanpy as sc
    import episcanpy.api as epi
    setup_seed(args.seed)
    run_output_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
    os.makedirs(run_output_dir, exist_ok=True)

    with open(os.path.join(run_output_dir, "run_config.json"), "w", encoding="utf-8") as fp:
        json.dump(vars(args), fp, indent=2)

    print(f"seed: {args.seed}")
    print("Loading data...")
    rna_adata = sc.read_h5ad(args.rna_h5ad)
    atac_adata = sc.read_h5ad(args.atac_h5ad)

    rna_adata = rna_adata[:, feature_nonzero_mask(rna_adata.X)]
    atac_adata = atac_adata[:, feature_nonzero_mask(atac_adata.X)]
    rna_mean_expression_raw = compute_mean_expression(rna_adata.X)
    rna_zero_rate_raw = 1.0 - compute_detection_rate(rna_adata.X)
    rna_log_mean_expression_raw = np.log10(rna_mean_expression_raw + 1e-8)

    rna_adata.raw = rna_adata.copy()
    atac_adata.raw = atac_adata.copy()
    rna_dim = rna_adata.shape[1]
    atac_dim = atac_adata.shape[1]

    print("Preprocessing RNA data...")
    sc.pp.normalize_total(rna_adata)
    sc.pp.log1p(rna_adata)
    sc.pp.highly_variable_genes(rna_adata, n_top_genes=args.n_top_genes)
    rna_adata = rna_adata[:, rna_adata.var.highly_variable].copy()

    print("Preprocessing ATAC data...")
    epi.pp.filter_features(atac_adata, min_cells=np.ceil(atac_adata.shape[0] * 0.10))
    epi.pp.tfidf(atac_adata)

    dataset = MultiModalDataset(rna_adata, atac_adata)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    rna_pro_dim = rna_adata.shape[1]
    atac_pro_dim = atac_adata.shape[1]
    print(f"RNA dimensions: {rna_dim}, ATAC dimensions: {atac_dim}")
    print(f"RNA processing dimensions: {rna_pro_dim}, ATAC processing dimensions: {atac_pro_dim}")

    model = scAPIFeatureLambda(
        rna_dim,
        atac_dim,
        rna_pro_dim,
        atac_pro_dim,
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
            run_output_dir, f"feature_lambda_model_seed_{args.seed}.pt"
        )
        print(f"Loading checkpoint for analysis only: {model_path}")
        state_dict = torch.load(model_path, map_location=args.device)
        model.load_state_dict(state_dict)
        model.eval()
        analyze_rna_feature_weights(model, rna_adata, run_output_dir)
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
        train_model(
            model,
            train_loader,
            optimizer,
            num_epochs=args.num_epochs,
            warmup_epochs=args.warmup_epochs,
            device=args.device,
            prior_reg_weight=args.prior_reg_weight,
        )
        torch.save(model.state_dict(), os.path.join(run_output_dir, f"feature_lambda_model_seed_{args.seed}.pt"))
        analyze_rna_feature_weights(model, rna_adata, run_output_dir)
        print(f"Saved outputs to {run_output_dir}")
    except Exception as exc:
        print("Error during training:", exc)
        traceback.print_exc()


if __name__ == "__main__":
    main(parse_args())
