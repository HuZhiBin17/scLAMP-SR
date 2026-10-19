import argparse
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-script", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--output-path")
    return parser.parse_args()


def first_existing(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("None of the expected files exist: " + ", ".join(paths))


class EncoderOnly(nn.Module):
    def __init__(
        self,
        rna_pro_dim,
        atac_pro_dim,
        hidden_dim=512,
        latent_dim=128,
        dropout_rate=0.2,
    ):
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

    def deterministic_means(self, rna_data, atac_data):
        rna_h = self.rna_encoder(rna_data)
        atac_h = self.atac_encoder(atac_data)
        rna_shared = self.shared_layer(rna_h)
        atac_shared = self.shared_layer(atac_h)
        rna_mu = torch.clamp(self.rna_mu(rna_shared), min=-10, max=10)
        atac_mu = torch.clamp(self.atac_mu(atac_shared), min=-10, max=10)
        shared_mu = torch.clamp(self.shared_mu((rna_h + atac_h) / 2), min=-10, max=10)
        return rna_mu, atac_mu, shared_mu


def feature_nonzero_mask(x):
    return np.asarray(x.sum(axis=0)).ravel() > 0


def seurat_hvg_normalize_log(x, n_top_genes, n_bins=20):
    x = sp.csr_matrix(x, dtype=np.float32)
    counts = np.asarray(x.sum(axis=1)).ravel()
    target_sum = np.median(counts[counts > 0])
    scale = np.divide(target_sum, counts, out=np.ones_like(counts), where=counts > 0)
    normalized = sp.diags(scale).dot(x).tocsr()

    mean = np.asarray(normalized.mean(axis=0)).ravel().astype(np.float64)
    mean_sq = np.asarray(normalized.power(2).mean(axis=0)).ravel().astype(np.float64)
    variance = (mean_sq - mean**2) * normalized.shape[0] / (normalized.shape[0] - 1)
    safe_mean = mean.copy()
    safe_mean[safe_mean == 0] = 1e-12
    dispersion = variance / safe_mean
    dispersion[dispersion == 0] = np.nan
    dispersion = np.log(dispersion)
    logged_mean = np.log1p(safe_mean)

    frame = pd.DataFrame({"means": logged_mean, "dispersions": dispersion})
    frame["mean_bin"] = pd.cut(frame["means"], bins=n_bins)
    grouped = frame.groupby("mean_bin", observed=False)["dispersions"]
    bin_mean = grouped.mean()
    bin_std = grouped.std(ddof=1)
    one_gene = bin_std.isnull()
    bin_std.loc[one_gene] = bin_mean.loc[one_gene].values
    bin_mean.loc[one_gene] = 0
    dispersion_norm = (
        frame["dispersions"].values - bin_mean[frame["mean_bin"].values].values
    ) / bin_std[frame["mean_bin"].values].values
    finite_sorted = np.sort(dispersion_norm[~np.isnan(dispersion_norm)])[::-1]
    cutoff = finite_sorted[min(n_top_genes, len(finite_sorted)) - 1]
    selected = np.nan_to_num(dispersion_norm) >= cutoff

    normalized = normalized[:, selected].copy()
    normalized.data = np.log1p(normalized.data)
    return normalized, selected


def dense_rows(x, start, stop):
    rows = x[start:stop]
    if sp.issparse(rows):
        rows = rows.toarray()
    return np.asarray(rows, dtype=np.float32)


def mark_stage(run_dir, message):
    path = os.path.join(run_dir, "嵌入导出阶段.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(message + "\n")
    print(message, flush=True)


def main():
    args = parse_args()
    config_path = first_existing(
        os.path.join(args.run_dir, "运行配置.json"),
        os.path.join(args.run_dir, "run_config.json"),
    )
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    mark_stage(args.run_dir, "stage 1/6: loading AnnData")
    rna = ad.read_h5ad(config["rna_h5ad"])
    modality2_path = config.get("modality2_h5ad") or config["atac_h5ad"]
    modality2_name = str(config.get("modality2_name", "ATAC")).upper()
    atac = ad.read_h5ad(modality2_path)
    if not rna.obs_names.equals(atac.obs_names):
        raise ValueError("RNA and ATAC cell order differs.")
    cell_ids = np.asarray(rna.obs_names, dtype=str)

    rna = rna[:, feature_nonzero_mask(rna.X)].copy()
    atac = atac[:, feature_nonzero_mask(atac.X)].copy()

    mark_stage(args.run_dir, "stage 2/6: selecting RNA highly variable genes")
    rna_normalized, rna_hvg_mask = seurat_hvg_normalize_log(
        rna.X,
        config["n_top_genes"],
    )
    rna = rna[:, rna_hvg_mask].copy()
    rna.X = rna_normalized
    transform = config.get("modality2_input_transform", "auto")
    if transform == "auto":
        transform = config.get("atac_input_transform", "raw")
    mark_stage(args.run_dir, f"stage 3/6: preprocessing {modality2_name}")
    if transform == "tfidf":
        atac_mask = np.asarray((atac.X > 0).sum(axis=0)).ravel() >= np.ceil(atac.n_obs * 0.10)
        atac = atac[:, atac_mask].copy()
        from sklearn.feature_extraction.text import TfidfTransformer

        atac.X = TfidfTransformer(norm="l1").fit_transform(atac.X)
        mark_stage(args.run_dir, f"stage 3/6: filtering {modality2_name} features and computing TF-IDF")
    elif transform == "clr":
        dense = atac.X.toarray() if sp.issparse(atac.X) else np.asarray(atac.X)
        geometric_mean = np.exp(np.mean(np.log1p(dense), axis=1, keepdims=True))
        atac.X = np.log1p(dense / np.maximum(geometric_mean, 1e-8)).astype(np.float32)
        mark_stage(args.run_dir, f"stage 3/6: computing {modality2_name} CLR")
    elif transform == "log1p":
        counts = np.asarray(atac.X.sum(axis=1)).ravel()
        scale = np.divide(1e4, counts, out=np.zeros_like(counts), where=counts > 0)
        atac.X = sp.diags(scale).dot(sp.csr_matrix(atac.X)).tocsr()
        atac.X.data = np.log1p(atac.X.data)
        mark_stage(args.run_dir, f"stage 3/6: computing {modality2_name} log1p")
    elif transform == "raw":
        # Historical runs computed a TF-IDF layer but trained the encoder on .X.
        mark_stage(args.run_dir, f"stage 3/6: retaining raw {modality2_name} counts")
    else:
        raise ValueError(f"Unsupported second-modality transform: {transform}")

    mark_stage(args.run_dir, "stage 4/6: memory-mapping encoder weights")
    model_encoder = EncoderOnly(
        rna.n_vars,
        atac.n_vars,
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        dropout_rate=config["dropout_rate"],
    )
    checkpoint = args.checkpoint_path or first_existing(
        os.path.join(args.run_dir, f"模型_随机种子_{config['seed']}.pt"),
        os.path.join(args.run_dir, f"feature_lambda_model_seed_{config['seed']}.pt"),
    )
    full_state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    encoder_state = {
        key[len("encoder.") :]: value
        for key, value in full_state.items()
        if key.startswith("encoder.")
    }
    model_encoder.load_state_dict(encoder_state)
    del full_state
    mark_stage(args.run_dir, "stage 5/6: running encoder inference")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_encoder.to(device)
    model_encoder.eval()

    integrated_parts = []
    rna_parts = []
    atac_parts = []
    starts = range(0, rna.n_obs, args.batch_size)
    with torch.no_grad():
        for start in tqdm(starts, total=int(np.ceil(rna.n_obs / args.batch_size)), desc="Extracting latents"):
            stop = min(start + args.batch_size, rna.n_obs)
            rna_x = torch.from_numpy(dense_rows(rna.X, start, stop)).to(device)
            atac_x = torch.from_numpy(dense_rows(atac.X, start, stop)).to(device)
            rna_mu, atac_mu, shared_mu = model_encoder.deterministic_means(rna_x, atac_x)
            rna_parts.append(rna_mu.cpu().numpy())
            atac_parts.append(atac_mu.cpu().numpy())
            integrated_parts.append(shared_mu.cpu().numpy())

    output_path = args.output_path or os.path.join(args.run_dir, "嵌入.npz")
    payload = dict(
        integrated=np.vstack(integrated_parts),
        rna=np.vstack(rna_parts),
        modality2=np.vstack(atac_parts),
        modality2_name=np.asarray(modality2_name),
        cell_id=cell_ids,
    )
    modality_key = modality2_name.lower()
    if modality_key in {"atac", "adt"}:
        payload[modality_key] = payload["modality2"]
    np.savez_compressed(output_path, **payload)
    mark_stage(args.run_dir, f"stage 6/6: saved deterministic latent means for {rna.n_obs:,} cells")


if __name__ == "__main__":
    main()
