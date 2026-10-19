"""Run the paper-version MIDAS model on paired RNA+ATAC or RNA+ADT data."""

import argparse
import json
import os
import random
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
MIDAS_ROOT = Path(os.environ.get("MIDAS_ROOT", ROOT / "third_party" / "midas"))
midas_models = None


DETERMINISTIC_LATENT = False


def sample_gaussian(mean, log_variance):
    if DETERMINISTIC_LATENT:
        return mean
    std = (0.5 * log_variance).exp()
    return mean + torch.randn_like(std) * std


def install_midas_io_stub():
    """Load official MIDAS modules without its unused CSV/OpenCV utilities."""
    sys.path.insert(0, str(MIDAS_ROOT))
    utility_stub = types.ModuleType("modules.utils")
    utility_stub.sample_gaussian = sample_gaussian
    utility_stub.get_dict = lambda keys, values: dict(zip(keys, values))
    sys.modules["modules.utils"] = utility_stub




def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna-h5ad", required=True)
    parser.add_argument("--modality2-h5ad", required=True)
    parser.add_argument("--modality2-name", choices=("ATAC", "ADT"), required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rna-features", type=int, default=3000)
    parser.add_argument("--atac-features", type=int, default=20000)
    parser.add_argument("--atac-min-detection", type=float, default=0.10)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--midas-root",
        default=None,
        help="Path to the official MIDAS source tree. Defaults to $MIDAS_ROOT or ./third_party/midas.",
    )
    return parser.parse_args()


def as_csr(matrix):
    return sp.csr_matrix(matrix, dtype=np.float32)


def select_rna_features(matrix, count):
    if matrix.shape[1] <= count:
        return np.arange(matrix.shape[1], dtype=np.int64)
    library = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(1e4, library, out=np.zeros_like(library), where=library > 0)
    normalized = matrix.multiply(scale[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    mean = np.asarray(normalized.mean(axis=0)).ravel()
    mean_square = np.asarray(normalized.power(2).mean(axis=0)).ravel()
    variance = np.maximum(mean_square - mean**2, 0.0)
    dispersion = variance / np.maximum(mean, 1e-8)
    return np.argpartition(dispersion, -count)[-count:].astype(np.int64)


def select_atac_features(matrix, count, minimum_detection):
    detection = np.asarray((matrix > 0).mean(axis=0)).ravel()
    eligible = np.flatnonzero(detection >= minimum_detection)
    if len(eligible) == 0:
        eligible = np.flatnonzero(detection > 0)
    if len(eligible) > count:
        chosen = np.argpartition(detection[eligible], -count)[-count:]
        eligible = eligible[chosen]
    return np.sort(eligible.astype(np.int64))


def chromosome_chunks(feature_count, groups=22):
    if feature_count < groups:
        raise ValueError("MIDAS ATAC encoder requires at least 22 selected features.")
    base, remainder = divmod(feature_count, groups)
    return [base + (index < remainder) for index in range(groups)]


def prepare_data(args):
    rna = ad.read_h5ad(args.rna_h5ad)
    modality2 = ad.read_h5ad(args.modality2_h5ad)
    if not rna.obs_names.equals(modality2.obs_names):
        raise ValueError("RNA and modality-2 cell order differs.")
    accepted_count_scales = {"raw_integer", "source_integer_counts"}
    count_scales = {}
    for name, item in (("RNA", rna), (args.modality2_name, modality2)):
        count_scale = item.uns.get("count_scale")
        if count_scale not in accepted_count_scales:
            raise ValueError(
                f"{name} must be a non-negative integer count matrix; "
                f"received count_scale={count_scale!r}."
            )
        count_scales[name] = count_scale

    indices = np.arange(rna.n_obs, dtype=np.int64)
    if args.max_cells > 0:
        indices = indices[: args.max_cells]
    rna_matrix = as_csr(rna.X[indices])
    modality2_matrix = as_csr(modality2.X[indices])
    rna_selected = select_rna_features(rna_matrix, args.rna_features)
    if args.modality2_name == "ATAC":
        modality2_selected = select_atac_features(
            modality2_matrix,
            args.atac_features,
            args.atac_min_detection,
        )
    else:
        modality2_selected = np.flatnonzero(
            np.asarray(modality2_matrix.sum(axis=0)).ravel() > 0
        ).astype(np.int64)
    rna_matrix = rna_matrix[:, rna_selected].tocsr()
    modality2_matrix = modality2_matrix[:, modality2_selected].tocsr()

    if args.batch_key and args.batch_key in rna.obs:
        batch_values = np.asarray(rna.obs.iloc[indices][args.batch_key], dtype=str)
    else:
        batch_values = np.asarray(["batch_0"] * len(indices), dtype=str)
    _, batch_codes = np.unique(batch_values, return_inverse=True)
    return {
        "rna": rna_matrix,
        args.modality2_name.lower(): modality2_matrix,
        "cell_id": np.asarray(rna.obs_names[indices], dtype=str),
        "batch": batch_codes.astype(np.int64),
        "batch_values": batch_values,
        "rna_selected": rna_selected,
        "modality2_selected": modality2_selected,
        "count_scales": count_scales,
    }


class PairedMIDASDataset(Dataset):
    def __init__(self, matrices, batches):
        self.matrices = matrices
        self.batches = batches
        self.modalities = list(matrices)
        self.masks = {
            name: np.ones(matrix.shape[1], dtype=np.float32)
            for name, matrix in matrices.items()
            if name in {"rna", "adt"}
        }

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        values = {}
        for name, matrix in self.matrices.items():
            row = matrix.getrow(index).toarray().ravel().astype(np.float32)
            if name == "atac":
                row = (row > 0).astype(np.float32)
            values[name] = row
        batch = np.asarray([self.batches[index]], dtype=np.int64)
        return {
            "x": values,
            "e": self.masks,
            "s": {name: batch for name in self.modalities + ["joint"]},
        }


def model_options(args, data):
    modalities = ["rna", args.modality2_name.lower()]
    dimensions = {name: data[name].shape[1] for name in modalities}
    batch_count = int(np.max(data["batch"])) + 1
    options = SimpleNamespace(
        task=args.dataset_name,
        experiment="default",
        ref_mods=modalities,
        mods=modalities,
        dims_x=dimensions,
        dim_c=args.latent_dim,
        dim_b=2,
        dim_z=args.latent_dim + 2,
        dims_enc_s=[16, 16],
        dims_dec_s=[16, 16],
        dims_enc_chr=[128, 32],
        dims_dec_chr=[32, 128],
        dims_enc_x=[1024, 128],
        dims_dec_x=[128, 1024],
        dims_discriminator=[128, 64],
        norm="ln",
        drop=0.2,
        dim_s=batch_count,
        dims_s={name: batch_count for name in modalities + ["joint"]},
        drop_s=0,
        s_drop_rate=0.1,
        sample_num=0,
        debug=0,
    )
    if "atac" in modalities:
        options.dims_chr = chromosome_chunks(dimensions["atac"])
    else:
        options.dims_chr = []
    options.dims_h = {
        name: (options.dims_enc_chr[-1] * 22 if name == "atac" else dimension)
        for name, dimension in dimensions.items()
    }
    return options


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    return {key: to_device(item, device) for key, item in value.items()}


def train_model(model, discriminator, loader, args, device):
    optimizer_model = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    optimizer_discriminator = torch.optim.AdamW(
        discriminator.parameters(), lr=args.learning_rate
    )
    history = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        discriminator.train()
        total_loss = 0.0
        total_model = 0.0
        total_adversarial = 0.0
        batches = 0
        for payload in loader:
            payload = to_device(payload, device)
            model_loss, latent_by_modality = model(payload)
            detached = {
                name: values.detach() for name, values in latent_by_modality.items()
            }
            for _ in range(3):
                optimizer_discriminator.zero_grad(set_to_none=True)
                discriminator_loss = discriminator(detached, payload["s"])
                discriminator_loss.backward()
                optimizer_discriminator.step()

            optimizer_model.zero_grad(set_to_none=True)
            adversarial_loss = -discriminator(latent_by_modality, payload["s"])
            loss = model_loss + adversarial_loss
            loss.backward()
            optimizer_model.step()
            total_loss += float(loss.item())
            total_model += float(model_loss.item())
            total_adversarial += float(adversarial_loss.item())
            batches += 1

        elapsed = time.time() - started
        mean_loss = total_loss / max(batches, 1)
        row = {
            "轮数": epoch + 1,
            "总损失": mean_loss,
            "模型损失": total_model / max(batches, 1),
            "对抗损失": total_adversarial / max(batches, 1),
            "已耗时秒": elapsed,
        }
        history.append(row)
        eta = elapsed / (epoch + 1) * (args.epochs - epoch - 1)
        print(
            f"MIDAS | {args.dataset_name} | epoch {epoch + 1:03d}/{args.epochs} "
            f"| loss={mean_loss:.4f} | elapsed={elapsed / 60:.1f} min "
            f"| eta={eta / 60:.1f} min",
            flush=True,
        )
    return pd.DataFrame(history)


def extract_embeddings(model, loader, modalities, device):
    global DETERMINISTIC_LATENT
    DETERMINISTIC_LATENT = True
    model.eval()
    parts = {name: [] for name in modalities + ["joint"]}
    with torch.no_grad():
        for payload in loader:
            payload = to_device(payload, device)
            outputs = model.sct(payload)
            latent_by_modality = outputs[-1]
            for name in parts:
                parts[name].append(latent_by_modality[name].cpu().numpy())
    DETERMINISTIC_LATENT = False
    return {name: np.concatenate(values) for name, values in parts.items()}


def load_midas(midas_root):
    global MIDAS_ROOT, midas_models
    MIDAS_ROOT = Path(midas_root)
    if not MIDAS_ROOT.exists():
        raise FileNotFoundError(
            "MIDAS source not found at "
            f"{MIDAS_ROOT}. Clone the official repository and pass --midas-root."
        )
    install_midas_io_stub()
    from modules import models as loaded_midas_models  # noqa: E402

    midas_models = loaded_midas_models


def main():
    args = parse_args()
    load_midas(args.midas_root or os.environ.get("MIDAS_ROOT", ROOT / "third_party" / "midas"))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data = prepare_data(args)
    modality2_key = args.modality2_name.lower()
    matrices = {"rna": data["rna"], modality2_key: data[modality2_key]}
    dataset = PairedMIDASDataset(matrices, data["batch"])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.dry_run,
        num_workers=0,
        pin_memory=True,
    )
    options = model_options(args, data)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = midas_models.Net(options).to(device)
    discriminator = midas_models.Discriminator(options).to(device)
    parameter_count = sum(p.numel() for p in model.parameters()) + sum(
        p.numel() for p in discriminator.parameters()
    )
    print(
        f"MIDAS paper model | dataset={args.dataset_name} | modalities=RNA+{args.modality2_name} "
        f"| cells={len(dataset):,} | features={data['rna'].shape[1]:,}+"
        f"{data[modality2_key].shape[1]:,} | latent={args.latent_dim} "
        f"| parameters={parameter_count / 1e6:.2f}M",
        flush=True,
    )

    if args.dry_run:
        payload = to_device(next(iter(loader)), device)
        with torch.no_grad():
            loss, latent = model(payload)
        print(
            json.dumps(
                {
                    "损失": float(loss.item()),
                    "潜变量形状": {
                        name: list(values.shape) for name, values in latent.items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    history = train_model(model, discriminator, loader, args, device)
    deterministic_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    embeddings = extract_embeddings(
        model,
        deterministic_loader,
        ["rna", modality2_key],
        device,
    )
    np.savez_compressed(
        output / "嵌入.npz",
        integrated=embeddings["joint"],
        rna=embeddings["rna"],
        modality2=embeddings[modality2_key],
        **{modality2_key: embeddings[modality2_key]},
        modality2_name=np.asarray(args.modality2_name),
        cell_id=data["cell_id"],
    )
    torch.save(
        {
            "model": model.state_dict(),
            "discriminator": discriminator.state_dict(),
        },
        output / "模型状态.pt",
    )
    history.to_csv(output / "训练历史.csv", index=False)
    np.savez_compressed(
        output / "特征选择.npz",
        rna_indices=data["rna_selected"],
        modality2_indices=data["modality2_selected"],
    )
    metadata = {
        **vars(args),
        "方法": "MIDAS",
        "实现": "官方论文复现分支网络与损失；项目稀疏 h5ad 数据适配层",
        "官方提交": "reproducibility branch",
        "细胞数": len(dataset),
        "RNA特征数": data["rna"].shape[1],
        "第二模态特征数": data[modality2_key].shape[1],
        "批次数": int(np.max(data["batch"])) + 1,
        "计数尺度声明": data["count_scales"],
        "Human Cortex边界": (
            "source_integer_counts 可用于实验1整合；不视为经验证原始UMI，"
            "不进入实验2原始计数概率主结果。"
            if "source_integer_counts" in data["count_scales"].values()
            else "不适用"
        ),
        "参数量": parameter_count,
        "耗时秒": float(history.iloc[-1]["已耗时秒"]),
    }
    with (output / "运行元数据.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    with (output / "完成状态.json").open("w", encoding="utf-8") as handle:
        json.dump({"状态": "成功", "输出目录": str(output.resolve())}, handle, ensure_ascii=False, indent=2)
    print(f"MIDAS completed: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
