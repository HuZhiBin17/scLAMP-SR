"""Evaluate held-out RNA<->ATAC prediction from posterior means or samples."""

import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
)
from sklearn.neighbors import KNeighborsClassifier
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments"
sys.path[:0] = [str(EXPERIMENT_DIR), str(ROOT)]

import run_heldout_probability_ablation as heldout  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--correlation-features", type=int, default=300)
    parser.add_argument("--classification-components", type=int, default=30)
    parser.add_argument("--classification-neighbors", type=int, default=15)
    parser.add_argument("--max-test-cells", type=int, default=0)
    parser.add_argument(
        "--latent-source",
        choices=("mu", "z"),
        default="z",
        help="Decode from deterministic posterior means or one posterior sample.",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=20260813,
        help="Fixed random seed used when latent-source is z.",
    )
    parser.add_argument(
        "--shared-posterior-snr-min",
        type=float,
        default=0.01,
        help="Abort when shared posterior signal/noise variance ratio is smaller.",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def checkpoint_artifact(checkpoint_dir, chinese_name, legacy_name):
    preferred = checkpoint_dir / chinese_name
    if preferred.is_file():
        return preferred
    legacy = checkpoint_dir / legacy_name
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(
        f"检查点缺少 {chinese_name}（也未找到历史名称 {legacy_name}）：{checkpoint_dir}"
    )


def sparse_log1p(matrix):
    result = sp.csr_matrix(matrix, dtype=np.float32).copy()
    result.data = np.log1p(result.data)
    return result


def feature_variance(matrix):
    matrix = sp.csr_matrix(matrix, dtype=np.float32)
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    second = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    return np.maximum(second - mean**2, 0.0)


def rowwise_spearman(observed, predicted, chunk_size=64):
    values = np.full(observed.shape[0], np.nan, dtype=np.float64)
    for start in range(0, observed.shape[0], chunk_size):
        stop = min(start + chunk_size, observed.shape[0])
        first = rankdata(observed[start:stop], axis=1)
        second = rankdata(predicted[start:stop], axis=1)
        first -= first.mean(axis=1, keepdims=True)
        second -= second.mean(axis=1, keepdims=True)
        denominator = np.sqrt(
            np.sum(first**2, axis=1) * np.sum(second**2, axis=1)
        )
        valid = denominator > 0
        chunk = np.full(stop - start, np.nan, dtype=np.float64)
        chunk[valid] = np.sum(first[valid] * second[valid], axis=1) / denominator[valid]
        values[start:stop] = chunk
    return values


def columnwise_spearman(observed, predicted, chunk_size=256):
    values = np.full(observed.shape[1], np.nan, dtype=np.float64)
    for start in range(0, observed.shape[1], chunk_size):
        stop = min(start + chunk_size, observed.shape[1])
        first = rankdata(observed[:, start:stop], axis=0)
        second = rankdata(predicted[:, start:stop], axis=0)
        first -= first.mean(axis=0, keepdims=True)
        second -= second.mean(axis=0, keepdims=True)
        denominator = np.sqrt(
            np.sum(first**2, axis=0) * np.sum(second**2, axis=0)
        )
        valid = denominator > 0
        chunk = np.full(stop - start, np.nan, dtype=np.float64)
        chunk[valid] = np.sum(first[:, valid] * second[:, valid], axis=0) / denominator[valid]
        values[start:stop] = chunk
    return values


def finite_summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, 0
    return float(np.mean(values)), float(np.median(values)), int(len(values))


def decode_predictions(model, loader, device, latent_source, sampling_seed):
    paths = {
        "ATAC→RNA": ("atac", "rna"),
        "双模态联合→RNA": ("shared", "rna"),
        "RNA→ATAC": ("rna", "mod2"),
        "双模态联合→ATAC": ("shared", "mod2"),
    }
    prediction_parts = {name: [] for name in paths}
    latent_parts = {name: [] for name in ("rna", "atac", "shared")}
    posterior_parts = {
        name: {kind: [] for kind in ("mu", "var")}
        for name in ("rna", "atac", "shared")
    }
    nll_sums = {name: 0.0 for name in paths}
    entries = {name: 0 for name in paths}
    model.eval()
    if latent_source == "z":
        torch.manual_seed(sampling_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sampling_seed)
    with torch.no_grad():
        for batch in tqdm(loader, desc="解码测试细胞", unit="batch"):
            rna_x, mod2_x, rna_y, mod2_y = [value.to(device) for value in batch[:4]]
            encoded = model.encoder(rna_x, mod2_x)
            for name, mu_index, var_index in (
                ("rna", 3, 4),
                ("atac", 5, 6),
                ("shared", 7, 8),
            ):
                posterior_parts[name]["mu"].append(
                    encoded[mu_index].detach().cpu().numpy().astype(np.float32)
                )
                posterior_parts[name]["var"].append(
                    encoded[var_index].detach().cpu().numpy().astype(np.float32)
                )
            if latent_source == "z":
                latents = {
                    "rna": encoded[0],
                    "atac": encoded[1],
                    "shared": encoded[2],
                }
            else:
                latents = {
                    "rna": encoded[3],
                    "atac": encoded[5],
                    "shared": encoded[7],
                }
            for name, latent in latents.items():
                latent_parts[name].append(
                    latent.detach().cpu().numpy().astype(np.float32)
                )
            parameter_sets = {
                name: model.decoder(latents[source], sample=None)[2]
                for name, (source, _) in paths.items()
            }
            for name, (_, target_name) in paths.items():
                target = rna_y if target_name == "rna" else mod2_y
                prefix = "rna" if target_name == "rna" else "mod2"
                params = parameter_sets[name]
                expected = heldout.predictive_moments(params, prefix)[0]
                prediction_parts[name].append(expected.detach().cpu().numpy().astype(np.float32))
                nll_sums[name] += float(
                    -heldout.mixture_log_probability(target, params, prefix).sum().item()
                )
                entries[name] += int(target.numel())
    predictions = {
        name: np.concatenate(parts, axis=0) for name, parts in prediction_parts.items()
    }
    latents = {
        name: np.concatenate(parts, axis=0) for name, parts in latent_parts.items()
    }
    posteriors = {
        name: {
            kind: np.concatenate(parts, axis=0)
            for kind, parts in values.items()
        }
        for name, values in posterior_parts.items()
    }
    nll = {name: nll_sums[name] / entries[name] for name in paths}
    return predictions, nll, latents, posteriors


def posterior_signal_audit(posteriors):
    rows = []
    for name, values in posteriors.items():
        signal_variance = float(np.var(values["mu"], axis=0).mean())
        noise_variance = float(values["var"].mean())
        rows.append(
            {
                "潜变量": name,
                "后验均值逐维细胞间标准差": float(
                    np.std(values["mu"], axis=0).mean()
                ),
                "平均后验标准差": float(np.sqrt(values["var"]).mean()),
                "细胞信号方差": signal_variance,
                "后验噪声方差": noise_variance,
                "信号噪声方差比": signal_variance / max(noise_variance, 1e-12),
            }
        )
    return rows


def evaluate_prediction(
    observed_counts,
    predicted_counts,
    modality,
    path_name,
    nll,
    cell_ids,
    feature_names,
):
    observed = observed_counts.toarray().astype(np.float32, copy=False)
    predicted = np.maximum(np.asarray(predicted_counts, dtype=np.float32), 0.0)
    observed_log = np.log1p(observed)
    predicted_log = np.log1p(predicted)
    row_corr = rowwise_spearman(observed_log, predicted_log)
    feature_corr = columnwise_spearman(observed_log, predicted_log)
    row_mean, row_median, row_valid = finite_summary(row_corr)
    feature_mean, feature_median, feature_valid = finite_summary(feature_corr)
    difference = predicted_log - observed_log
    mse = float(np.mean(difference**2))
    variance = float(np.var(observed_log))
    metrics = {
        "模态": modality,
        "预测路径": path_name,
        "测试细胞数": observed.shape[0],
        "特征数": observed.shape[1],
        "负对数似然": nll,
        "对数尺度均方误差": mse,
        "对数尺度标准化均方误差": mse / max(variance, 1e-12),
        "对数尺度平均绝对误差": float(np.mean(np.abs(difference))),
        "逐细胞Spearman均值": row_mean,
        "逐细胞Spearman中位数": row_median,
        "逐细胞Spearman有效数": row_valid,
        "逐特征Spearman均值": feature_mean,
        "逐特征Spearman中位数": feature_median,
        "逐特征Spearman有效数": feature_valid,
    }
    cell_frame = pd.DataFrame(
        {
            "细胞": cell_ids,
            "模态": modality,
            "预测路径": path_name,
            "Spearman": row_corr,
        }
    )
    feature_frame = pd.DataFrame(
        {
            "特征": feature_names,
            "模态": modality,
            "预测路径": path_name,
            "Spearman": feature_corr,
        }
    )
    if modality == "ATAC":
        average_precision = np.full(observed.shape[0], np.nan, dtype=np.float64)
        binary = observed > 0
        for index in tqdm(
            range(observed.shape[0]),
            desc=f"ATAC平均精确率：{path_name}",
            unit="cell",
            leave=False,
        ):
            if binary[index].any() and not binary[index].all():
                average_precision[index] = average_precision_score(
                    binary[index], predicted[index]
                )
        ap_mean, ap_median, ap_valid = finite_summary(average_precision)
        metrics.update(
            {
                "ATAC逐细胞平均精确率均值": ap_mean,
                "ATAC逐细胞平均精确率中位数": ap_median,
                "ATAC逐细胞平均精确率有效数": ap_valid,
            }
        )
        cell_frame["平均精确率"] = average_precision
    return metrics, cell_frame, feature_frame


def mean_baseline_metrics(observed_counts, train_counts, modality, cell_ids):
    observed = observed_counts.toarray().astype(np.float32, copy=False)
    train_mean = np.asarray(train_counts.mean(axis=0)).ravel().astype(np.float32)
    predicted = np.broadcast_to(train_mean, observed.shape)
    observed_log = np.log1p(observed)
    predicted_log = np.log1p(predicted)
    row_corr = rowwise_spearman(observed_log, predicted_log)
    row_mean, row_median, row_valid = finite_summary(row_corr)
    difference = predicted_log - observed_log
    mse = float(np.mean(difference**2))
    result = {
        "模态": modality,
        "预测路径": "训练集特征均值",
        "测试细胞数": observed.shape[0],
        "特征数": observed.shape[1],
        "负对数似然": np.nan,
        "对数尺度均方误差": mse,
        "对数尺度标准化均方误差": mse / max(float(np.var(observed_log)), 1e-12),
        "对数尺度平均绝对误差": float(np.mean(np.abs(difference))),
        "逐细胞Spearman均值": row_mean,
        "逐细胞Spearman中位数": row_median,
        "逐细胞Spearman有效数": row_valid,
        "逐特征Spearman均值": np.nan,
        "逐特征Spearman中位数": np.nan,
        "逐特征Spearman有效数": 0,
    }
    cell_frame = pd.DataFrame(
        {
            "细胞": cell_ids,
            "模态": modality,
            "预测路径": "训练集特征均值",
            "Spearman": row_corr,
        }
    )
    if modality == "ATAC":
        binary = observed > 0
        ap = np.full(observed.shape[0], np.nan, dtype=np.float64)
        for index in range(observed.shape[0]):
            if binary[index].any() and not binary[index].all():
                ap[index] = average_precision_score(binary[index], train_mean)
        ap_mean, ap_median, ap_valid = finite_summary(ap)
        result.update(
            {
                "ATAC逐细胞平均精确率均值": ap_mean,
                "ATAC逐细胞平均精确率中位数": ap_median,
                "ATAC逐细胞平均精确率有效数": ap_valid,
            }
        )
        cell_frame["平均精确率"] = ap
    return result, cell_frame


def latent_audit(latents, latent_source):
    scale_rows = []
    display_names = {
        name: f"{latent_source}_{name}" for name in ("rna", "atac", "shared")
    }
    for name, values in latents.items():
        scale_rows.append(
            {
                "潜变量": display_names[name],
                "细胞数": values.shape[0],
                "维度": values.shape[1],
                "全局均值": float(values.mean()),
                "全局标准差": float(values.std()),
                "逐细胞L2范数均值": float(np.linalg.norm(values, axis=1).mean()),
                "逐维度标准差均值": float(values.std(axis=0).mean()),
            }
        )
    distance_rows = []
    for first, second in (("rna", "atac"), ("rna", "shared"), ("atac", "shared")):
        first_values = latents[first]
        second_values = latents[second]
        denominator = np.linalg.norm(first_values, axis=1) * np.linalg.norm(
            second_values, axis=1
        )
        cosine = np.divide(
            np.sum(first_values * second_values, axis=1),
            denominator,
            out=np.full(len(first_values), np.nan, dtype=np.float32),
            where=denominator > 0,
        )
        distance_rows.append(
            {
                "潜变量一": display_names[first],
                "潜变量二": display_names[second],
                "配对欧氏距离均值": float(
                    np.linalg.norm(first_values - second_values, axis=1).mean()
                ),
                "配对余弦相似度均值": float(np.nanmean(cosine)),
            }
        )
    return scale_rows, distance_rows


def correlation_structure(observed, predicted, feature_indices):
    truth = np.log1p(observed[:, feature_indices].toarray().astype(np.float32))
    estimate = np.log1p(np.maximum(predicted[:, feature_indices], 0.0))
    truth_correlation = np.corrcoef(truth, rowvar=False)
    estimate_correlation = np.corrcoef(estimate, rowvar=False)
    upper = np.triu_indices(len(feature_indices), k=1)
    first = truth_correlation[upper]
    second = estimate_correlation[upper]
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 2:
        return np.nan
    return float(np.corrcoef(rankdata(first[valid]), rankdata(second[valid]))[0, 1])


def differential_signal(observed, predicted, labels, feature_indices, modality, path_name):
    truth = np.log1p(observed[:, feature_indices].toarray().astype(np.float32))
    estimate = np.log1p(np.maximum(predicted[:, feature_indices], 0.0))
    rows = []
    all_truth, all_estimate = [], []
    for label in np.unique(labels):
        target = labels == label
        if target.sum() < 10 or (~target).sum() < 10:
            continue
        truth_effect = truth[target].mean(axis=0) - truth[~target].mean(axis=0)
        estimate_effect = estimate[target].mean(axis=0) - estimate[~target].mean(axis=0)
        correlation = np.corrcoef(rankdata(truth_effect), rankdata(estimate_effect))[0, 1]
        cutoff = np.quantile(np.abs(truth_effect), 0.80)
        marker = np.abs(truth_effect) >= cutoff
        sign_agreement = np.mean(
            np.sign(truth_effect[marker]) == np.sign(estimate_effect[marker])
        )
        rows.append(
            {
                "模态": modality,
                "预测路径": path_name,
                "细胞类型": label,
                "细胞数": int(target.sum()),
                "差异效应Spearman": float(correlation),
                "强差异特征方向一致率": float(sign_agreement),
                "强差异特征数": int(marker.sum()),
            }
        )
        all_truth.append(truth_effect)
        all_estimate.append(estimate_effect)
    pooled = np.corrcoef(
        rankdata(np.concatenate(all_truth)),
        rankdata(np.concatenate(all_estimate)),
    )[0, 1]
    return rows, float(pooled)


def normalized_log1p_dense(matrix, target_sum=1e4):
    matrix = np.maximum(np.asarray(matrix, dtype=np.float32), 0.0)
    totals = matrix.sum(axis=1, keepdims=True)
    scaled = np.divide(
        matrix * target_sum,
        totals,
        out=np.zeros_like(matrix),
        where=totals > 0,
    )
    return np.log1p(scaled)


def downstream_classification(
    train_counts,
    test_counts,
    predictions,
    train_labels,
    test_labels,
    modality,
    feature_indices,
    components,
    neighbors,
    tfidf=None,
):
    if modality == "RNA":
        train_matrix = heldout.normalize_log1p(train_counts)[:, feature_indices]
        true_test_matrix = heldout.normalize_log1p(test_counts)[:, feature_indices]
        prepared = {
            name: normalized_log1p_dense(value)[:, feature_indices]
            for name, value in predictions.items()
        }
    else:
        train_matrix = tfidf.transform(train_counts)
        true_test_matrix = tfidf.transform(test_counts)
        prepared = {
            name: tfidf.transform(sp.csr_matrix(np.maximum(value, 0.0)))
            for name, value in predictions.items()
        }
    reducer = TruncatedSVD(
        n_components=min(components, train_matrix.shape[1] - 1),
        random_state=0,
    )
    train_embedding = reducer.fit_transform(train_matrix)
    classifier = KNeighborsClassifier(n_neighbors=neighbors, weights="distance")
    classifier.fit(train_embedding, train_labels)
    matrices = {"真实目标模态": true_test_matrix, **prepared}
    rows = []
    for name, matrix in matrices.items():
        predicted_labels = classifier.predict(reducer.transform(matrix))
        rows.append(
            {
                "模态": modality,
                "输入": name,
                "准确率": accuracy_score(test_labels, predicted_labels),
                "平衡准确率": balanced_accuracy_score(test_labels, predicted_labels),
            }
        )
    return rows


def main():
    args = parse_args()
    started = time.time()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint_artifact(checkpoint_dir, "运行配置.json", "run_config.json")
    preprocessing_path = checkpoint_artifact(
        checkpoint_dir,
        "预处理特征.npz",
        "preprocessing_features.npz",
    )
    model_path = checkpoint_artifact(checkpoint_dir, "模型权重.pt", "model.pt")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    with (output_dir / "运行配置.json").open("w", encoding="utf-8") as handle:
        json.dump({**vars(args), "源模型配置": config}, handle, ensure_ascii=False, indent=2)

    rna = ad.read_h5ad(ROOT / config["rna_h5ad"])
    atac = ad.read_h5ad(ROOT / config["modality2_h5ad"])
    split = np.load(args.split_path, allow_pickle=True)
    train_indices = np.asarray(split["train"], dtype=np.int64)
    test_indices = np.asarray(split["test"], dtype=np.int64)
    if args.max_test_cells > 0:
        test_indices = test_indices[: args.max_test_cells]
    preprocessing = np.load(preprocessing_path)
    rna_mask = preprocessing["rna_target_mask"]
    atac_mask = preprocessing["mod2_target_mask"]
    hvg = preprocessing["rna_hvg_local_indices"]
    rna_raw = rna.X[:, rna_mask].tocsr().astype(np.float32)
    atac_raw = atac.X[:, atac_mask].tocsr().astype(np.float32)
    rna_input = heldout.normalize_log1p(rna_raw)[:, hvg]
    tfidf = TfidfTransformer(
        norm=config.get("tfidf_norm", "l1"),
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    ).fit(atac_raw[train_indices])
    atac_input = tfidf.transform(atac_raw).astype(np.float32)

    test_set = heldout.PairTensorDataset(
        rna_input,
        atac_input,
        rna_raw,
        atac_raw,
        test_indices,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    train_mean = np.asarray(rna_raw[train_indices].mean(axis=0)).ravel()
    train_zero = 1.0 - np.asarray((rna_raw[train_indices] > 0).mean(axis=0)).ravel()
    model = heldout.ControlledModel(
        rna_raw.shape[1],
        atac_raw.shape[1],
        rna_input.shape[1],
        atac_input.shape[1],
        np.log10(train_mean + 1e-8),
        train_zero,
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        dropout_rate=config["dropout_rate"],
        method="state_resolved",
        state_head_source=config.get("state_head_source", "classifier"),
        state_gate_input=config.get("state_gate_input", "decoder"),
        modality2_prototype_scale=config.get("modality2_prototype_scale", 0.0),
        modality2_prototype_penalty=config.get("modality2_prototype_penalty", 0.08),
        rna_state_mean_scale=config.get("rna_state_mean_scale", 0.0),
        rna_state_mean_penalty=config.get("rna_state_mean_penalty", 0.08),
        auxiliary_state_gate_gradient=config.get("auxiliary_state_gate_gradient", False),
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    predictions, nll, latents, posteriors = decode_predictions(
        model,
        test_loader,
        device,
        args.latent_source,
        args.sampling_seed,
    )
    posterior_rows = posterior_signal_audit(posteriors)
    posterior_frame = pd.DataFrame(posterior_rows)
    posterior_frame.to_csv(output_dir / "后验信号噪声审计.csv", index=False)
    shared_snr = float(
        posterior_frame.loc[
            posterior_frame["潜变量"] == "shared", "信号噪声方差比"
        ].iloc[0]
    )
    if shared_snr < args.shared_posterior_snr_min:
        failure = {
            "状态": "失败",
            "原因": "共享后验发生塌缩，不能用于缺失模态正式评价。",
            "共享后验信号噪声方差比": shared_snr,
            "最低允许信号噪声方差比": args.shared_posterior_snr_min,
            "源检查点": str(model_path.resolve()),
        }
        with (output_dir / "完成状态.json").open("w", encoding="utf-8") as handle:
            json.dump(failure, handle, ensure_ascii=False, indent=2)
        raise RuntimeError(
            "共享后验信号噪声方差比仅为 "
            f"{shared_snr:.6g}，检查点发生共享后验塌缩；停止生成正式结果。"
        )

    test_rna = rna_raw[test_indices]
    test_atac = atac_raw[test_indices]
    cell_ids = np.asarray(rna.obs_names[test_indices], dtype=str)
    labels = rna.obs["cell_type"].astype(str).to_numpy()[test_indices]
    rna_names = np.asarray(rna.var_names[rna_mask], dtype=str)
    atac_names = np.asarray(atac.var_names[atac_mask], dtype=str)
    metric_rows = []
    cell_frames = []
    feature_frames = []
    for modality, target, feature_names, path_names in (
        (
            "RNA",
            test_rna,
            rna_names,
            ("ATAC→RNA", "双模态联合→RNA"),
        ),
        (
            "ATAC",
            test_atac,
            atac_names,
            ("RNA→ATAC", "双模态联合→ATAC"),
        ),
    ):
        for path_name in path_names:
            metrics, cells, features = evaluate_prediction(
                target,
                predictions[path_name],
                modality,
                path_name,
                nll[path_name],
                cell_ids,
                feature_names,
            )
            metric_rows.append(metrics)
            cell_frames.append(cells)
            feature_frames.append(features)
        train_target = rna_raw[train_indices] if modality == "RNA" else atac_raw[train_indices]
        baseline_metrics, baseline_cells = mean_baseline_metrics(
            target,
            train_target,
            modality,
            cell_ids,
        )
        metric_rows.append(baseline_metrics)
        cell_frames.append(baseline_cells)

    rna_variable = np.argsort(feature_variance(sparse_log1p(rna_raw[train_indices])))[
        -args.correlation_features :
    ]
    atac_variable = np.argsort(feature_variance(sparse_log1p(atac_raw[train_indices])))[
        -args.correlation_features :
    ]
    structure_rows = []
    differential_rows = []
    pooled_effects = {}
    for modality, observed, feature_indices, path_names in (
        ("RNA", test_rna, rna_variable, ("ATAC→RNA", "双模态联合→RNA")),
        ("ATAC", test_atac, atac_variable, ("RNA→ATAC", "双模态联合→ATAC")),
    ):
        for path_name in path_names:
            structure_rows.append(
                {
                    "模态": modality,
                    "预测路径": path_name,
                    "训练集高变特征数": len(feature_indices),
                    "特征相关结构Spearman": correlation_structure(
                        observed,
                        predictions[path_name],
                        feature_indices,
                    ),
                }
            )
            rows, pooled = differential_signal(
                observed,
                predictions[path_name],
                labels,
                feature_indices,
                modality,
                path_name,
            )
            differential_rows.extend(rows)
            pooled_effects[f"{modality}|{path_name}"] = pooled

    classification_rows = []
    classification_rows.extend(
        downstream_classification(
            rna_raw[train_indices],
            test_rna,
            {
                "ATAC→RNA": predictions["ATAC→RNA"],
                "双模态联合→RNA": predictions["双模态联合→RNA"],
                "训练集特征均值": np.broadcast_to(
                    np.asarray(rna_raw[train_indices].mean(axis=0)).ravel(),
                    test_rna.shape,
                ),
            },
            rna.obs["cell_type"].astype(str).to_numpy()[train_indices],
            labels,
            "RNA",
            hvg,
            args.classification_components,
            args.classification_neighbors,
        )
    )
    classification_rows.extend(
        downstream_classification(
            atac_raw[train_indices],
            test_atac,
            {
                "RNA→ATAC": predictions["RNA→ATAC"],
                "双模态联合→ATAC": predictions["双模态联合→ATAC"],
                "训练集特征均值": np.broadcast_to(
                    np.asarray(atac_raw[train_indices].mean(axis=0)).ravel(),
                    test_atac.shape,
                ),
            },
            rna.obs["cell_type"].astype(str).to_numpy()[train_indices],
            labels,
            "ATAC",
            atac_variable,
            args.classification_components,
            args.classification_neighbors,
            tfidf=tfidf,
        )
    )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output_dir / "缺失模态预测_总体指标.csv", index=False)
    pd.concat(cell_frames, ignore_index=True).to_csv(
        output_dir / "缺失模态预测_逐细胞指标.csv", index=False
    )
    pd.concat(feature_frames, ignore_index=True).to_csv(
        output_dir / "缺失模态预测_逐特征指标.csv", index=False
    )
    pd.DataFrame(structure_rows).to_csv(
        output_dir / "特征相关结构保留.csv", index=False
    )
    pd.DataFrame(differential_rows).to_csv(
        output_dir / "细胞类型差异信号保留.csv", index=False
    )
    pd.DataFrame(classification_rows).to_csv(
        output_dir / "补全数据下游分类.csv", index=False
    )
    latent_scale_rows, latent_distance_rows = latent_audit(
        latents,
        args.latent_source,
    )
    pd.DataFrame(latent_scale_rows).to_csv(
        output_dir / "潜变量尺度审计.csv", index=False
    )
    pd.DataFrame(latent_distance_rows).to_csv(
        output_dir / "潜变量配对距离审计.csv", index=False
    )
    metric_lookup = metrics_frame.set_index(["模态", "预测路径"])
    shared_is_better = all(
        metric_lookup.loc[(modality, shared_path), "对数尺度标准化均方误差"]
        <= metric_lookup.loc[(modality, cross_path), "对数尺度标准化均方误差"]
        for modality, cross_path, shared_path in (
            ("RNA", "ATAC→RNA", "双模态联合→RNA"),
            ("ATAC", "RNA→ATAC", "双模态联合→ATAC"),
        )
    )
    latent_label = "单次后验采样 z" if args.latent_source == "z" else "确定性后验均值 mu"
    summary = {
        "状态": "成功",
        "随机种子": 0,
        "解码潜变量": latent_label,
        "采样随机种子": args.sampling_seed if args.latent_source == "z" else None,
        "训练细胞数": int(len(train_indices)),
        "测试细胞数": int(len(test_indices)),
        "数据泄漏控制": "模型与预处理统计仅使用固定训练划分；测试标签仅用于训练后评价。",
        "源检查点": str(model_path.resolve()),
        "源模型轮数": int(config["epochs"]),
        "源模型预热轮数": int(config["warmup_epochs"]),
        "潜空间维度": int(config["latent_dim"]),
        "差异效应合并Spearman": pooled_effects,
        "跨模态相对均值基线": {
            "ATAC→RNA_标准化均方误差改善比例": float(
                1.0
                - metric_lookup.loc[("RNA", "ATAC→RNA"), "对数尺度标准化均方误差"]
                / metric_lookup.loc[("RNA", "训练集特征均值"), "对数尺度标准化均方误差"]
            ),
            "RNA→ATAC_标准化均方误差改善比例": float(
                1.0
                - metric_lookup.loc[("ATAC", "RNA→ATAC"), "对数尺度标准化均方误差"]
                / metric_lookup.loc[("ATAC", "训练集特征均值"), "对数尺度标准化均方误差"]
            ),
            "RNA→ATAC_平均精确率提升": float(
                metric_lookup.loc[("ATAC", "RNA→ATAC"), "ATAC逐细胞平均精确率均值"]
                - metric_lookup.loc[("ATAC", "训练集特征均值"), "ATAC逐细胞平均精确率均值"]
            ),
        },
        "联合输入参考结论": (
            f"{args.latent_source}_shared 在两个目标模态上均不劣于单模态跨模态路径。"
            if shared_is_better
            else f"{args.latent_source}_shared 在本次评价中不是两个目标模态上的一致经验上界。"
        ),
        "耗时秒": time.time() - started,
    }
    with (output_dir / "试点汇总.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (output_dir / "完成状态.json").open("w", encoding="utf-8") as handle:
        json.dump({"状态": "成功", "输出目录": str(output_dir)}, handle, ensure_ascii=False, indent=2)
    print(metrics_frame.to_string(index=False), flush=True)
    print(pd.DataFrame(structure_rows).to_string(index=False), flush=True)
    print(pd.DataFrame(classification_rows).to_string(index=False), flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
