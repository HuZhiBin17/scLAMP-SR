"""Calibrate scLAMP-SR mixture decoders on a frozen recommended latent space."""

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments"
sys.path[:0] = [str(EXPERIMENT_DIR), str(ROOT)]

import run_heldout_probability_ablation as heldout  # noqa: E402
from sclamp_sr import model as state_sr  # noqa: E402
from sclamp_sr.utils import setup_seed  # noqa: E402


class LatentCountDataset(Dataset):
    def __init__(
        self,
        latent,
        rna_raw,
        mod2_raw,
        indices,
        state_labels=None,
        state_targets=None,
    ):
        self.latent = torch.from_numpy(
            np.asarray(latent[indices], dtype=np.float32)
        )
        self.rna_raw = torch.from_numpy(heldout.dense_float(rna_raw[indices]))
        self.mod2_raw = torch.from_numpy(
            heldout.dense_float(mod2_raw[indices])
        )
        self.state_labels = (
            None
            if state_labels is None
            else torch.from_numpy(np.asarray(state_labels, dtype=np.int64))
        )
        self.state_targets = (
            None
            if state_targets is None
            else torch.from_numpy(np.asarray(state_targets, dtype=np.float32))
        )

    def __len__(self):
        return self.latent.shape[0]

    def __getitem__(self, index):
        result = (
            self.latent[index],
            self.rna_raw[index],
            self.mod2_raw[index],
        )
        if self.state_labels is not None:
            result += (self.state_labels[index],)
        if self.state_targets is not None:
            result += (self.state_targets[index],)
        return result


def soft_targets_at_temperature(squared_distances, temperature):
    logits = -np.asarray(squared_distances, dtype=np.float64) / max(
        float(temperature),
        1e-12,
    )
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return (values / values.sum(axis=1, keepdims=True)).astype(np.float32)


def compute_latent_pseudostates(
    latent,
    train_indices,
    rna_raw,
    mod2_raw,
    n_states,
    seed,
    top_rna,
    top_mod2,
):
    train_latent = np.asarray(latent[train_indices], dtype=np.float32)
    kmeans = KMeans(
        n_clusters=n_states,
        random_state=seed,
        n_init=10,
    ).fit(train_latent)
    train_distances = np.square(kmeans.transform(train_latent))
    train_targets, temperature = state_sr.calibrated_kmeans_soft_targets(
        train_distances,
        0.90,
    )
    all_distances = np.square(kmeans.transform(latent))
    all_targets = soft_targets_at_temperature(all_distances, temperature)
    all_labels = kmeans.predict(latent).astype(np.int64)
    train_labels = all_labels[train_indices]

    (
        rna_selected,
        rna_states,
        rna_specificity,
        rna_log2fc,
        _,
    ) = heldout.state_feature_statistics(
        rna_raw[train_indices],
        train_labels,
        n_states,
        top_rna,
    )
    (
        mod2_selected,
        mod2_states,
        mod2_specificity,
        _,
        _,
    ) = heldout.state_feature_statistics(
        mod2_raw[train_indices],
        train_labels,
        n_states,
        top_mod2,
    )
    rna_mean = heldout.vector(rna_raw[train_indices].mean(axis=0))
    positive = rna_mean[rna_mean > 0]
    low_cutoff, high_cutoff = np.quantile(positive, [0.20, 0.80])
    low = np.flatnonzero(
        (rna_mean > 0) & (rna_mean <= low_cutoff)
    ).astype(np.int64)
    high = np.flatnonzero(rna_mean >= high_cutoff).astype(np.int64)
    return {
        "all_labels": all_labels,
        "all_targets": all_targets,
        "train_labels": train_labels,
        "train_targets": train_targets,
        "temperature": float(temperature),
        "centers": kmeans.cluster_centers_.astype(np.float32),
        "rna_selected": rna_selected,
        "rna_states": rna_states,
        "rna_specificity": rna_specificity,
        "rna_log2fc": rna_log2fc,
        "mod2_selected": mod2_selected,
        "mod2_states": mod2_states,
        "mod2_specificity": mod2_specificity,
        "low": low,
        "high": high,
        "rna_global_mean": rna_mean.astype(np.float32),
        "low_cutoff": float(low_cutoff),
        "high_cutoff": float(high_cutoff),
    }


def build_decoder(args, rna_raw, mod2_raw, checkpoint):
    rna_mean = heldout.vector(rna_raw.mean(axis=0))
    rna_zero = 1.0 - heldout.vector((rna_raw > 0).mean(axis=0))
    decoder = heldout.ControlledDecoder(
        rna_raw.shape[1],
        mod2_raw.shape[1],
        np.log10(rna_mean + 1e-8),
        rna_zero,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout_rate=args.dropout_rate,
        method=args.method,
        modality2_prototype_scale=(
            args.modality2_prototype_scale
            if args.method == "state_resolved"
            else 0.0
        ),
        modality2_prototype_penalty=args.modality2_prototype_penalty,
        rna_state_mean_scale=(
            args.rna_state_mean_scale
            if args.method == "state_resolved"
            else 0.0
        ),
        rna_state_mean_penalty=args.rna_state_mean_penalty,
        auxiliary_state_gate_gradient=args.auxiliary_state_gate_gradient,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    decoder_state = {
        key[len("decoder.") :]: value
        for key, value in state.items()
        if key.startswith("decoder.")
    }
    incompatible = decoder.load_state_dict(decoder_state, strict=False)
    if args.method == "feature":
        with torch.no_grad():
            decoder.rna_feature_logits.zero_()
            decoder.rna_feature_logits[0].copy_(decoder.get_rna_base_logit())
            decoder.mod2_feature_logits.copy_(decoder.atac_logits)
    elif args.feature_baseline_decoder:
        feature_state = torch.load(
            args.feature_baseline_decoder,
            map_location="cpu",
            weights_only=True,
        )
        decoder.load_state_dict(feature_state, strict=False)
        with torch.no_grad():
            feature_rna_log_odds = (
                feature_state["rna_feature_logits"][0]
                - feature_state["rna_feature_logits"][1]
            )
            decoder.rna_gene_residual.copy_(
                feature_rna_log_odds - decoder.get_rna_prior_logit()
            )
            decoder.atac_logits.copy_(feature_state["mod2_feature_logits"])
            decoder.prototype_gene_factors.zero_()
            if decoder.prototype_modality2_factors is not None:
                decoder.prototype_modality2_factors.zero_()
            if decoder.prototype_rna_mean_factors is not None:
                decoder.prototype_rna_mean_factors.zero_()
        print(
            "State-resolved decoder initialized to the feature-level "
            "mixture weights with zero state residuals.",
            flush=True,
        )
    if args.freeze_base_decoder and args.method == "state_resolved":
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
        decoder.prototype_gene_factors.requires_grad_(True)
        if decoder.prototype_modality2_factors is not None:
            decoder.prototype_modality2_factors.requires_grad_(True)
        if decoder.prototype_rna_mean_factors is not None:
            decoder.prototype_rna_mean_factors.requires_grad_(True)
        for parameter in decoder.prototype_gate.parameters():
            parameter.requires_grad_(True)
        print(
            "Base decoder frozen; only the state gate and residual factors "
            "are trainable.",
            flush=True,
        )
    print(f"Decoder missing keys: {list(incompatible.missing_keys)}", flush=True)
    print(f"Decoder unexpected keys: {list(incompatible.unexpected_keys)}", flush=True)
    return decoder


def build_optimizer(decoder, args):
    if args.method == "feature":
        special = [
            (
                [decoder.rna_feature_logits, decoder.mod2_feature_logits],
                args.mixture_lr,
                1e-5,
            ),
        ]
    else:
        special = [
            ([decoder.atac_logits], args.mixture_lr, 1e-5),
            (
                [decoder.rna_prior_bias, decoder.rna_prior_feature_weights],
                args.prior_lr,
                1e-5,
            ),
            ([decoder.rna_gene_residual], args.gene_residual_lr, 0.0),
        ]
    special_ids = {
        id(parameter)
        for parameters, _, _ in special
        for parameter in parameters
    }
    main = [
        parameter
        for parameter in decoder.parameters()
        if parameter.requires_grad and id(parameter) not in special_ids
    ]
    groups = [{"params": main, "lr": args.main_lr}]
    groups.extend(
        {
            "params": parameters,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        }
        for parameters, learning_rate, weight_decay in special
    )
    return torch.optim.Adam(groups)


def decode(decoder, latent):
    return decoder(latent, sample=None, state_context=latent)[2]


def reconstruction_loss(params, rna, mod2, prior_weight):
    rna_nll = -heldout.mixture_log_probability(
        rna,
        params,
        "rna",
    ).mean()
    mod2_nll = -heldout.mixture_log_probability(
        mod2,
        params,
        "mod2",
    ).mean()
    prior = params["mixture_prior_reg"] * prior_weight
    return rna_nll + mod2_nll + prior, rna_nll, mod2_nll, prior


@torch.no_grad()
def validation_nll(decoder, loader, device):
    decoder.eval()
    totals = {"rna": 0.0, "mod2": 0.0}
    entries = {"rna": 0, "mod2": 0}
    for batch in loader:
        latent, rna, mod2 = [
            value.to(device, non_blocking=True) for value in batch[:3]
        ]
        params = decode(decoder, latent)
        for key, target, prefix in (
            ("rna", rna, "rna"),
            ("mod2", mod2, "mod2"),
        ):
            totals[key] += float(
                -heldout.mixture_log_probability(
                    target,
                    params,
                    prefix,
                ).sum().item()
            )
            entries[key] += target.numel()
    return float(
        np.mean([totals[key] / max(entries[key], 1) for key in totals])
    )


def train_decoder(decoder, train_loader, valid_loader, pseudo, args, device):
    optimizer = build_optimizer(decoder, args)
    selected_rna = torch.as_tensor(
        pseudo["rna_selected"],
        dtype=torch.long,
        device=device,
    )
    selected_rna_states = torch.as_tensor(
        pseudo["rna_states"],
        dtype=torch.long,
        device=device,
    )
    desired_log2fc = torch.as_tensor(
        pseudo["rna_log2fc"],
        dtype=torch.float32,
        device=device,
    )
    selected_mod2 = torch.as_tensor(
        pseudo["mod2_selected"],
        dtype=torch.long,
        device=device,
    )
    selected_mod2_states = torch.as_tensor(
        pseudo["mod2_states"],
        dtype=torch.long,
        device=device,
    )
    low = torch.as_tensor(pseudo["low"], dtype=torch.long, device=device)
    high = torch.as_tensor(pseudo["high"], dtype=torch.long, device=device)
    rna_global_mean = torch.as_tensor(
        pseudo["rna_global_mean"],
        dtype=torch.float32,
        device=device,
    )
    initial_validation = validation_nll(decoder, valid_loader, device)
    history = [
        {
            "epoch": 0,
            "validation_joint_nll": initial_validation,
            "elapsed_seconds": 0.0,
            "eta_seconds": math.nan,
        }
    ]
    best_validation = initial_validation
    best_state = copy.deepcopy(decoder.state_dict())
    stale = 0
    started = time.time()
    print(
        f"Epoch 000/{args.epochs} | validation anchor={initial_validation:.6f}",
        flush=True,
    )

    for epoch in range(args.epochs):
        decoder.train()
        totals = {
            "loss": 0.0,
            "rna_nll": 0.0,
            "mod2_nll": 0.0,
            "rna_weight_contrast": 0.0,
            "mod2_weight_contrast": 0.0,
            "rna_mean_log2_contrast": 0.0,
            "high_low_weight_gap": 0.0,
            "state_target_loss": 0.0,
            "state_gate_max_probability": 0.0,
            "expression_mean_calibration_loss": 0.0,
        }
        batches = 0
        ramp = min(1.0, (epoch + 1) / max(args.ramp_epochs, 1))
        for batch in train_loader:
            latent, rna, mod2 = [
                value.to(device, non_blocking=True) for value in batch[:3]
            ]
            state_labels = batch[3].to(device, non_blocking=True)
            state_targets = batch[4].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            params = decode(decoder, latent)
            loss, rna_nll, mod2_nll, _ = reconstruction_loss(
                params,
                rna,
                mod2,
                args.prior_reg_weight,
            )
            rna_weight = loss.detach().new_zeros(())
            mod2_weight = loss.detach().new_zeros(())
            rna_mean = loss.detach().new_zeros(())
            high_low = loss.detach().new_zeros(())
            state_target_loss = loss.detach().new_zeros(())
            gate_max = loss.detach().new_zeros(())
            expression_mean_calibration = loss.detach().new_zeros(())
            if args.method == "state_resolved":
                contrast_loss, rna_weight = heldout.weight_contrast_loss(
                    params["rna_lambda_1"],
                    state_labels,
                    selected_rna,
                    selected_rna_states,
                    args.rna_weight_contrast_margin,
                )
                mod2_loss, mod2_weight = heldout.weight_contrast_loss(
                    params["atac_lambda_1"],
                    state_labels,
                    selected_mod2,
                    selected_mod2_states,
                    args.modality2_weight_contrast_margin,
                )
                order_loss, high_low = heldout.expression_order_loss(
                    params["rna_lambda_1"],
                    low,
                    high,
                    args.expression_order_margin,
                )
                logits = (
                    decoder.prototype_gate(latent)
                    / max(decoder.prototype_temperature, 1e-6)
                )
                state_target_loss = -(
                    state_targets
                    * torch.nn.functional.log_softmax(logits, dim=1)
                ).sum(dim=1).mean()
                gate_max = params["rna_prototype_probs"].max(
                    dim=1
                ).values.mean()
                desired = torch.clamp(
                    desired_log2fc * args.state_mean_target_fraction,
                    min=0.0,
                    max=args.state_mean_target_max_log2fc,
                )
                mean_loss, rna_mean = heldout.state_mean_contrast_loss(
                    heldout.rna_expected_mean(params),
                    state_targets,
                    selected_rna,
                    selected_rna_states,
                    desired,
                )
                calibration_indices = torch.cat([low, high])
                predicted_group_mean = heldout.rna_expected_mean(
                    params
                ).index_select(1, calibration_indices).mean(dim=0)
                observed_group_mean = rna_global_mean.index_select(
                    0,
                    calibration_indices,
                )
                expression_mean_calibration = (
                    torch.nn.functional.smooth_l1_loss(
                        torch.log1p(predicted_group_mean),
                        torch.log1p(observed_group_mean),
                    )
                )
                loss = (
                    loss
                    + args.rna_weight_contrast_weight * contrast_loss
                    + args.modality2_weight_contrast_weight * mod2_loss
                    + args.expression_order_weight * order_loss
                    + ramp * args.state_ce_weight * state_target_loss
                    + ramp * args.state_mean_contrast_weight * mean_loss
                    + args.expression_mean_calibration_weight
                    * expression_mean_calibration
                )
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.item())
            totals["rna_nll"] += float(rna_nll.item())
            totals["mod2_nll"] += float(mod2_nll.item())
            totals["rna_weight_contrast"] += float(rna_weight.item())
            totals["mod2_weight_contrast"] += float(mod2_weight.item())
            totals["rna_mean_log2_contrast"] += float(rna_mean.item())
            totals["high_low_weight_gap"] += float(high_low.item())
            totals["state_target_loss"] += float(state_target_loss.item())
            totals["state_gate_max_probability"] += float(gate_max.item())
            totals["expression_mean_calibration_loss"] += float(
                expression_mean_calibration.item()
            )
            batches += 1

        averages = {
            key: value / max(batches, 1) for key, value in totals.items()
        }
        value = validation_nll(decoder, valid_loader, device)
        if value < best_validation - args.validation_min_delta:
            best_validation = value
            best_state = copy.deepcopy(decoder.state_dict())
            stale = 0
        else:
            stale += 1
        elapsed = time.time() - started
        eta = elapsed / (epoch + 1) * (args.epochs - epoch - 1)
        history.append(
            {
                "epoch": epoch + 1,
                **averages,
                "validation_joint_nll": value,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
            }
        )
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs}"
            f" | loss={averages['loss']:.6f}"
            f" | RNA NLL={averages['rna_nll']:.6f}"
            f" | ATAC NLL={averages['mod2_nll']:.6f}"
            f" | RNA权重差={averages['rna_weight_contrast']:.4f}"
            f" | ATAC权重差={averages['mod2_weight_contrast']:.4f}"
            f" | RNA均值log2差={averages['rna_mean_log2_contrast']:.4f}"
            f" | 高低表达权重差={averages['high_low_weight_gap']:.4f}"
            f" | 状态门最大概率={averages['state_gate_max_probability']:.3f}"
            f" | 表达均值校准={averages['expression_mean_calibration_loss']:.4f}"
            f" | val NLL={value:.6f}"
            f" | 已耗时={elapsed / 60:.1f}分钟"
            f" | 剩余={eta / 60:.1f}分钟",
            flush=True,
        )
        if args.early_stopping_patience > 0 and stale >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}.", flush=True)
            break

    if best_state is not None and not args.keep_last_checkpoint:
        decoder.load_state_dict(best_state)
        print(
            f"Restored best validation checkpoint: {best_validation:.6f}",
            flush=True,
        )
    return history


@torch.no_grad()
def evaluate_decoder(decoder, loader, device, modality2_name):
    decoder.eval()
    totals = {key: None for key in ("rna", "mod2")}
    cells = 0
    for batch in loader:
        latent, rna, mod2 = [
            value.to(device, non_blocking=True) for value in batch[:3]
        ]
        params = decode(decoder, latent)
        for key, target, prefix in (
            ("rna", rna, "rna"),
            ("mod2", mod2, "mod2"),
        ):
            log_prob = heldout.mixture_log_probability(
                target,
                params,
                prefix,
            )
            expected, variance, zero = heldout.predictive_moments(
                params,
                prefix,
            )
            values = {
                "nll": -log_prob.sum(dim=0),
                "obs_sum": target.sum(dim=0),
                "obs_sq": target.square().sum(dim=0),
                "obs_zero": (target == 0).sum(dim=0),
                "pred_sum": expected.sum(dim=0),
                "pred_second": (
                    variance + expected.square()
                ).sum(dim=0),
                "pred_zero": zero.sum(dim=0),
                "entries": torch.tensor(target.shape[0], device=device),
            }
            if totals[key] is None:
                totals[key] = {
                    name: value.detach().cpu().double()
                    for name, value in values.items()
                }
            else:
                for name, value in values.items():
                    totals[key][name] += value.detach().cpu().double()
        cells += rna.shape[0]

    rows = []
    for key, label in (("rna", "RNA"), ("mod2", modality2_name)):
        values = totals[key]
        n = float(values["entries"].item())
        observed_mean = values["obs_sum"] / n
        predicted_mean = values["pred_sum"] / n
        observed_variance = torch.clamp(
            values["obs_sq"] / n - observed_mean.square(),
            min=0.0,
        )
        predicted_variance = torch.clamp(
            values["pred_second"] / n - predicted_mean.square(),
            min=0.0,
        )
        observed_zero = values["obs_zero"] / n
        predicted_zero = values["pred_zero"] / n
        rows.append(
            {
                "模态": label,
                "测试细胞数": cells,
                "特征数": len(observed_mean),
                "每项负对数似然": float(
                    values["nll"].sum().item()
                    / (n * len(observed_mean))
                ),
                "零比例平均绝对误差": float(
                    torch.mean(
                        torch.abs(observed_zero - predicted_zero)
                    ).item()
                ),
                "均值平均绝对误差": float(
                    torch.mean(
                        torch.abs(observed_mean - predicted_mean)
                    ).item()
                ),
                "方差平均绝对误差": float(
                    torch.mean(
                        torch.abs(
                            observed_variance - predicted_variance
                        )
                    ).item()
                ),
            }
        )
    joint = float(np.mean([row["每项负对数似然"] for row in rows]))
    return rows, joint


@torch.no_grad()
def analyze_weight_classes(
    decoder,
    latent,
    indices,
    rna_raw,
    gene_names,
    cell_types,
    pseudo,
    device,
    batch_size,
):
    marker_names = [
        "MS4A1",
        "CD79A",
        "CD74",
        "GNLY",
        "NKG7",
        "KLRD1",
        "LYZ",
        "S100A8",
        "S100A9",
        "CD3D",
        "CD3E",
        "CD8A",
        "CD8B",
    ]
    name_to_index = {str(name): index for index, name in enumerate(gene_names)}
    marker_names = [name for name in marker_names if name in name_to_index]
    marker_indices = np.asarray(
        [name_to_index[name] for name in marker_names],
        dtype=np.int64,
    )
    groups = {
        "普遍低表达基因": pseudo["low"],
        "普遍高表达基因": pseudo["high"],
        "经典标志基因": marker_indices,
    }
    group_weight_sum = {key: 0.0 for key in groups}
    group_nll_sum = {key: 0.0 for key in groups}
    group_entries = {key: 0 for key in groups}
    group_observed_sum = {
        key: np.zeros(len(feature_indices), dtype=np.float64)
        for key, feature_indices in groups.items()
    }
    group_predicted_sum = {
        key: np.zeros(len(feature_indices), dtype=np.float64)
        for key, feature_indices in groups.items()
    }
    marker_weight_batches = []
    marker_expected_batches = []

    decoder.eval()
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        z = torch.as_tensor(
            latent[batch_indices],
            dtype=torch.float32,
            device=device,
        )
        rna = torch.as_tensor(
            heldout.dense_float(rna_raw[batch_indices]),
            dtype=torch.float32,
            device=device,
        )
        params = decode(decoder, z)
        expected, _, _ = heldout.predictive_moments(params, "rna")
        nll = -heldout.mixture_log_probability(rna, params, "rna")
        weights = params["rna_lambda_1"]
        if weights.dim() == 1:
            weights = weights.unsqueeze(0).expand(len(batch_indices), -1)
        for label, feature_indices in groups.items():
            feature_tensor = torch.as_tensor(
                feature_indices,
                dtype=torch.long,
                device=device,
            )
            group_weight_sum[label] += float(
                weights.index_select(1, feature_tensor).sum().item()
            )
            group_nll_sum[label] += float(
                nll.index_select(1, feature_tensor).sum().item()
            )
            group_observed_sum[label] += (
                rna.index_select(1, feature_tensor)
                .sum(dim=0)
                .detach()
                .cpu()
                .double()
                .numpy()
            )
            group_predicted_sum[label] += (
                expected.index_select(1, feature_tensor)
                .sum(dim=0)
                .detach()
                .cpu()
                .double()
                .numpy()
            )
            group_entries[label] += int(
                len(batch_indices) * len(feature_indices)
            )
        marker_tensor = torch.as_tensor(
            marker_indices,
            dtype=torch.long,
            device=device,
        )
        marker_weight_batches.append(
            weights.index_select(1, marker_tensor)
            .detach()
            .cpu()
            .numpy()
        )
        marker_expected_batches.append(
            expected.index_select(1, marker_tensor)
            .detach()
            .cpu()
            .numpy()
        )

    category_rows = []
    for label in groups:
        n_cells = group_entries[label] / max(len(groups[label]), 1)
        observed_means = group_observed_sum[label] / max(n_cells, 1)
        predicted_means = group_predicted_sum[label] / max(n_cells, 1)
        category_rows.append(
            {
                "基因类别": label,
                "特征数": len(groups[label]),
                "平均NB权重": group_weight_sum[label]
                / max(group_entries[label], 1),
                "每项负对数似然": group_nll_sum[label]
                / max(group_entries[label], 1),
                "观测平均表达": float(observed_means.mean()),
                "预测平均表达": float(predicted_means.mean()),
                "逐特征均值平均绝对误差": float(
                    np.mean(np.abs(observed_means - predicted_means))
                ),
            }
        )

    marker_rows = []
    test_types = np.asarray(cell_types[indices], dtype=str)
    marker_weights = np.concatenate(marker_weight_batches, axis=0)
    marker_expected = np.concatenate(marker_expected_batches, axis=0)
    for marker_position, (name, feature_index) in enumerate(
        zip(marker_names, marker_indices)
    ):
        observed = heldout.vector(rna_raw[indices, feature_index].toarray())
        means = {
            cell_type: float(observed[test_types == cell_type].mean())
            for cell_type in np.unique(test_types)
        }
        target_type = max(means, key=means.get)
        target_mask = test_types == target_type
        other_mask = ~target_mask
        target_predicted = float(
            marker_expected[target_mask, marker_position].mean()
        )
        other_predicted = float(
            marker_expected[other_mask, marker_position].mean()
        )
        marker_rows.append(
            {
                "基因": name,
                "目标细胞类型": target_type,
                "目标类型观测均值": means[target_type],
                "其他类型观测均值": float(
                    observed[test_types != target_type].mean()
                ),
                "观测目标对其他log2倍数": float(
                    np.log2(
                        (means[target_type] + 0.05)
                        / (
                            float(
                                observed[test_types != target_type].mean()
                            )
                            + 0.05
                        )
                    )
                ),
                "目标类型NB权重": float(
                    marker_weights[target_mask, marker_position].mean()
                ),
                "其他类型NB权重": float(
                    marker_weights[other_mask, marker_position].mean()
                ),
                "目标对其他NB权重差": float(
                    marker_weights[target_mask, marker_position].mean()
                    - marker_weights[other_mask, marker_position].mean()
                ),
                "目标类型预测表达": target_predicted,
                "其他类型预测表达": other_predicted,
                "预测目标对其他log2倍数": float(
                    np.log2(
                        (target_predicted + 0.05)
                        / (other_predicted + 0.05)
                    )
                ),
            }
        )
    return category_rows, marker_rows


@torch.no_grad()
def summarize_state_residuals(decoder, latent, pseudo, device, batch_size):
    rna_contrasts = []
    mod2_contrasts = []
    gate_max = []
    labels = pseudo["all_labels"]
    rna_selected = torch.as_tensor(
        pseudo["rna_selected"],
        dtype=torch.long,
        device=device,
    )
    rna_states = np.asarray(pseudo["rna_states"])
    mod2_selected = torch.as_tensor(
        pseudo["mod2_selected"],
        dtype=torch.long,
        device=device,
    )
    mod2_states = np.asarray(pseudo["mod2_states"])
    rna_weights = []
    mod2_weights = []
    for start in range(0, len(latent), batch_size):
        z = torch.as_tensor(
            latent[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        params = decode(decoder, z)
        rna_weights.append(
            params["rna_lambda_1"]
            .index_select(1, rna_selected)
            .detach()
            .cpu()
            .numpy()
        )
        mod2_weights.append(
            params["atac_lambda_1"]
            .index_select(1, mod2_selected)
            .detach()
            .cpu()
            .numpy()
        )
        gate_max.append(
            params["rna_prototype_probs"]
            .max(dim=1)
            .values.detach()
            .cpu()
            .numpy()
        )
    rna_weights = np.concatenate(rna_weights)
    mod2_weights = np.concatenate(mod2_weights)
    for state in range(decoder.prototype_count):
        target = labels == state
        other = ~target
        rna_mask = rna_states == state
        mod2_mask = mod2_states == state
        if target.any() and other.any() and rna_mask.any():
            rna_contrasts.extend(
                (
                    rna_weights[target][:, rna_mask].mean(axis=0)
                    - rna_weights[other][:, rna_mask].mean(axis=0)
                ).tolist()
            )
        if target.any() and other.any() and mod2_mask.any():
            mod2_contrasts.extend(
                (
                    mod2_weights[target][:, mod2_mask].mean(axis=0)
                    - mod2_weights[other][:, mod2_mask].mean(axis=0)
                ).tolist()
            )
    return {
        "RNA状态NB权重差均值": float(np.mean(rna_contrasts)),
        "RNA状态NB权重差中位数": float(np.median(rna_contrasts)),
        "ATAC状态NB权重差均值": float(np.mean(mod2_contrasts)),
        "ATAC状态NB权重差中位数": float(np.median(mod2_contrasts)),
        "状态门平均最大概率": float(np.concatenate(gate_max).mean()),
        "软目标训练集平均最大概率": float(
            pseudo["train_targets"].max(axis=1).mean()
        ),
    }


def main(args):
    setup_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    method_directory = {
        "feature": "特征级权重",
        "state_resolved": "状态分辨权重",
    }[args.method]
    output = (
        Path(args.output_dir)
        / method_directory
        / f"随机种子_{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "运行配置.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)

    rna = ad.read_h5ad(args.rna_h5ad)
    mod2 = ad.read_h5ad(args.modality2_h5ad)
    if not np.array_equal(rna.obs_names, mod2.obs_names):
        raise ValueError("RNA and modality-2 cell order does not match.")
    latent_file = np.load(args.latent_npz)
    latent = np.asarray(latent_file["integrated"], dtype=np.float32)
    latent_ids = np.asarray(latent_file["cell_id"], dtype=str)
    if not np.array_equal(latent_ids, np.asarray(rna.obs_names, dtype=str)):
        raise ValueError("Latent cell order does not match the count matrices.")

    rna_raw = sp.csr_matrix(rna.X, dtype=np.float32)
    mod2_raw = sp.csr_matrix(mod2.X, dtype=np.float32)
    split_file = np.load(args.split_npz)
    train_indices = split_file["train"]
    valid_indices = split_file["valid"]
    test_indices = split_file["test"]
    pseudo = compute_latent_pseudostates(
        latent,
        train_indices,
        rna_raw,
        mod2_raw,
        args.state_clusters,
        args.seed,
        args.state_contrast_top_genes,
        args.modality2_state_contrast_top_features,
    )
    np.save(output / "训练集软伪状态目标.npy", pseudo["train_targets"])
    pd.DataFrame(
        {
            "细胞编号": np.asarray(rna.obs_names, dtype=str),
            "伪状态": pseudo["all_labels"],
            "最大软目标概率": pseudo["all_targets"].max(axis=1),
        }
    ).to_csv(output / "全部细胞伪状态.csv", index=False)
    pd.DataFrame(
        {
            "RNA特征编号": pseudo["rna_selected"],
            "RNA特征名": np.asarray(rna.var_names, dtype=str)[
                pseudo["rna_selected"]
            ],
            "目标伪状态": pseudo["rna_states"],
            "特异性得分": pseudo["rna_specificity"],
            "训练集观测log2倍数": pseudo["rna_log2fc"],
        }
    ).to_csv(output / "RNA状态对比特征.csv", index=False)
    pd.DataFrame(
        {
            "ATAC特征编号": pseudo["mod2_selected"],
            "ATAC特征名": np.asarray(mod2.var_names, dtype=str)[
                pseudo["mod2_selected"]
            ],
            "目标伪状态": pseudo["mod2_states"],
            "特异性得分": pseudo["mod2_specificity"],
        }
    ).to_csv(output / "ATAC状态对比特征.csv", index=False)
    with (output / "软伪状态信息.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "训练集目标平均最大概率": float(
                    pseudo["train_targets"].max(axis=1).mean()
                ),
                "距离温度": pseudo["temperature"],
                "训练集各状态细胞数": np.bincount(
                    pseudo["train_labels"],
                    minlength=args.state_clusters,
                ).tolist(),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    test_set = LatentCountDataset(
        latent,
        rna_raw,
        mod2_raw,
        test_indices,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    if not args.evaluate_only:
        train_set = LatentCountDataset(
            latent,
            rna_raw,
            mod2_raw,
            train_indices,
            pseudo["train_labels"],
            pseudo["train_targets"],
        )
        valid_set = LatentCountDataset(
            latent,
            rna_raw,
            mod2_raw,
            valid_indices,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
        )
    decoder = build_decoder(
        args,
        rna_raw[train_indices],
        mod2_raw[train_indices],
        args.checkpoint,
    )
    device = torch.device(args.device)
    if args.method == "state_resolved" and args.distance_state_gate:
        decoder.set_distance_prototype_gate(
            pseudo["centers"],
            pseudo["temperature"],
            trainable_centers=False,
        )
        print(
            "Using the fixed squared-distance state gate calibrated to "
            "training-set mean max probability 0.90.",
            flush=True,
        )
    decoder.to(device)
    print(
        f"方法={args.method}; 数据集=GLUE PBMC10k; "
        f"训练/验证/测试={len(train_indices)}/{len(valid_indices)}/{len(test_indices)}; "
        f"潜变量维度={latent.shape[1]}; 潜空间冻结=True",
        flush=True,
    )
    if args.evaluate_only:
        decoder_path = (
            Path(args.decoder_path)
            if args.decoder_path
            else output / "模型解码器.pt"
        )
        decoder.load_state_dict(
            torch.load(
                decoder_path,
                map_location=device,
                weights_only=True,
            )
        )
        print(f"仅评估，已加载解码器：{decoder_path}", flush=True)
    else:
        history = train_decoder(
            decoder,
            train_loader,
            valid_loader,
            pseudo,
            args,
            device,
        )
        pd.DataFrame(history).to_csv(output / "训练历史.csv", index=False)
        torch.save(decoder.state_dict(), output / "模型解码器.pt")
    with (output / "解码器元数据.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "方法": args.method,
                "状态门类型": (
                    "固定平方距离门"
                    if args.distance_state_gate
                    else "多层感知机门"
                ),
                "状态门温度": float(decoder.prototype_temperature),
                "二模态状态残差尺度": float(
                    decoder.modality2_prototype_scale
                ),
                "RNA状态均值残差尺度": float(
                    decoder.rna_state_mean_scale
                ),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    rows, joint_nll = evaluate_decoder(
        decoder,
        test_loader,
        device,
        args.modality2_name,
    )
    for row in rows:
        row["联合负对数似然"] = joint_nll
        row["方法"] = args.method
        row["随机种子"] = args.seed
    pd.DataFrame(rows).to_csv(output / "测试指标.csv", index=False)

    category_rows, marker_rows = analyze_weight_classes(
        decoder,
        latent,
        test_indices,
        rna_raw,
        np.asarray(rna.var_names, dtype=str),
        rna.obs["cell_type"].astype(str).to_numpy(),
        pseudo,
        device,
        args.batch_size,
    )
    pd.DataFrame(category_rows).to_csv(
        output / "三类基因权重与拟合.csv",
        index=False,
    )
    pd.DataFrame(marker_rows).to_csv(
        output / "经典标志基因拟合.csv",
        index=False,
    )
    residual_summary = (
        summarize_state_residuals(
            decoder,
            latent,
            pseudo,
            device,
            args.batch_size,
        )
        if args.method == "state_resolved"
        else {}
    )
    with (output / "状态残差汇总.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            residual_summary,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    source_embeddings = {
        key: latent_file[key] for key in ("integrated", "rna", "atac", "cell_id")
    }
    np.savez_compressed(output / "潜变量.npz", **source_embeddings)
    preservation = {
        "整合潜变量最大绝对差": 0.0,
        "RNA潜变量最大绝对差": 0.0,
        "ATAC潜变量最大绝对差": 0.0,
        "说明": "解码器校准直接复用推荐检查点保存的潜变量，编码器未参与训练。",
    }
    with (output / "潜变量保持.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            preservation,
            handle,
            ensure_ascii=False,
            indent=2,
        )
    with (output / "完成状态.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "状态": "成功",
                "联合负对数似然": joint_nll,
                "输出目录": str(output),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"完成：{output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna_h5ad", required=True)
    parser.add_argument("--modality2_h5ad", required=True)
    parser.add_argument("--modality2_name", default="ATAC")
    parser.add_argument("--latent_npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature_baseline_decoder", default="")
    parser.add_argument("--split_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--method",
        choices=("feature", "state_resolved"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--main_lr", type=float, default=3e-4)
    parser.add_argument("--mixture_lr", type=float, default=1e-4)
    parser.add_argument("--prior_lr", type=float, default=5e-5)
    parser.add_argument("--gene_residual_lr", type=float, default=2e-4)
    parser.add_argument("--prior_reg_weight", type=float, default=2e-3)
    parser.add_argument("--state_clusters", type=int, default=12)
    parser.add_argument("--state_contrast_top_genes", type=int, default=1024)
    parser.add_argument(
        "--modality2_state_contrast_top_features",
        type=int,
        default=1024,
    )
    parser.add_argument("--rna_weight_contrast_weight", type=float, default=0.05)
    parser.add_argument("--rna_weight_contrast_margin", type=float, default=0.03)
    parser.add_argument(
        "--modality2_weight_contrast_weight",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--modality2_weight_contrast_margin",
        type=float,
        default=0.03,
    )
    parser.add_argument("--modality2_prototype_scale", type=float, default=0.35)
    parser.add_argument(
        "--modality2_prototype_penalty",
        type=float,
        default=0.08,
    )
    parser.add_argument("--rna_state_mean_scale", type=float, default=0.50)
    parser.add_argument("--rna_state_mean_penalty", type=float, default=0.08)
    parser.add_argument("--state_ce_weight", type=float, default=0.02)
    parser.add_argument("--state_mean_contrast_weight", type=float, default=0.10)
    parser.add_argument("--state_mean_target_fraction", type=float, default=0.25)
    parser.add_argument(
        "--state_mean_target_max_log2fc",
        type=float,
        default=1.50,
    )
    parser.add_argument("--expression_order_weight", type=float, default=0.01)
    parser.add_argument("--expression_order_margin", type=float, default=0.08)
    parser.add_argument(
        "--expression_mean_calibration_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--ramp_epochs", type=int, default=5)
    parser.add_argument("--auxiliary_state_gate_gradient", action="store_true")
    parser.add_argument("--validation_min_delta", type=float, default=1e-5)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--keep_last_checkpoint", action="store_true")
    parser.add_argument("--freeze_base_decoder", action="store_true")
    parser.add_argument("--distance_state_gate", action="store_true")
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--decoder_path", default="")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
