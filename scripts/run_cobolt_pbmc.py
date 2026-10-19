import argparse
import json
import os
import random
import time

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from cobolt.model import Cobolt
from cobolt.utils import MultiomicDataset, SingleData


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna-h5ad", default="dataset/scLAMP_SR_ready/GLUE_PBMC10k/rna.h5ad")
    parser.add_argument("--atac-h5ad", default="dataset/scLAMP_SR_ready/GLUE_PBMC10k/atac.h5ad")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="GLUE PBMC10k")
    parser.add_argument("--dataset-id", default="PBMC10k")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    return parser.parse_args()


def reorder(latent, barcodes, desired):
    positions = {str(barcode): index for index, barcode in enumerate(barcodes)}
    return latent[[positions[barcode] for barcode in desired]]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    started = time.time()

    print(f"Preparing paired {args.dataset_name} raw counts for Cobolt...", flush=True)
    rna = ad.read_h5ad(args.rna_h5ad)
    atac = ad.read_h5ad(args.atac_h5ad)
    if not rna.obs_names.equals(atac.obs_names):
        raise ValueError("RNA and ATAC cell order differs.")
    cell_ids = np.asarray(rna.obs_names, dtype=str)
    desired_barcodes = np.asarray([args.dataset_id + "~" + x for x in cell_ids], dtype=str)

    rna_data = SingleData(
        "GeneExpr",
        args.dataset_id,
        np.asarray(rna.var_names, dtype=str),
        sp.csr_matrix(rna.X, dtype=np.float32),
        cell_ids,
    )
    atac_data = SingleData(
        "ChromAccess",
        args.dataset_id,
        np.asarray(atac.var_names, dtype=str),
        sp.csr_matrix(atac.X, dtype=np.float32),
        cell_ids,
    )
    rna_data.filter_features(upper_quantile=0.99, lower_quantile=0.70)
    atac_data.filter_features(upper_quantile=0.99, lower_quantile=0.70)
    dataset = MultiomicDataset.from_singledata(rna_data, atac_data)
    print(dataset, flush=True)

    model = Cobolt(
        dataset=dataset,
        n_latent=args.latent_dim,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=args.learning_rate,
        batch_size=args.batch_size,
    )
    print(
        f"Training Cobolt: epochs={args.epochs}, "
        f"batch size={args.batch_size}, latent dimensions={args.latent_dim}",
        flush=True,
    )
    model.train(num_epochs=args.epochs)

    print("Extracting joint and modality-specific latent representations...", flush=True)
    integrated, joint_barcodes = model.get_latent([True, True], return_barcode=True)
    rna_latent, rna_barcodes = model.get_latent([True, False], return_barcode=True)
    atac_latent, atac_barcodes = model.get_latent([False, True], return_barcode=True)
    integrated = reorder(integrated, joint_barcodes, desired_barcodes)
    rna_latent = reorder(rna_latent, rna_barcodes, desired_barcodes)
    atac_latent = reorder(atac_latent, atac_barcodes, desired_barcodes)
    np.savez_compressed(
        os.path.join(args.output_dir, "嵌入.npz"),
        integrated=integrated,
        rna=rna_latent,
        atac=atac_latent,
        modality2=atac_latent,
        modality2_name=np.asarray("ATAC"),
        cell_id=cell_ids,
    )
    torch.save(model.model.state_dict(), os.path.join(args.output_dir, "模型状态.pt"))
    metadata = {
        **vars(args),
        "method": "Cobolt",
        "n_cells": rna.n_obs,
        "n_rna_features": int(rna_data.count.shape[1]),
        "n_atac_features": int(atac_data.count.shape[1]),
        "elapsed_seconds": time.time() - started,
    }
    with open(os.path.join(args.output_dir, "运行元数据.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Finished in {metadata['elapsed_seconds'] / 60:.1f} minutes.", flush=True)


if __name__ == "__main__":
    main()
