import argparse
import json
import os
import random
import time

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import tqdm.auto  # Required before importing the compatible torchmetrics release.

import scvi
from scvi.model import MULTIVI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna-h5ad", default="dataset/scLAMP_SR_ready/GLUE_PBMC10k/rna.h5ad")
    parser.add_argument("--atac-h5ad", default="dataset/scLAMP_SR_ready/GLUE_PBMC10k/atac.h5ad")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="GLUE PBMC10k")
    parser.add_argument("--categorical-covariate-key")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-top-genes", type=int, default=3000)
    parser.add_argument("--min-atac-cell-fraction", type=float, default=0.10)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def nonzero_features(x):
    return np.asarray(x.sum(axis=0)).ravel() > 0


def prepare_multiome(rna_path, atac_path, n_top_genes, min_atac_fraction):
    rna = ad.read_h5ad(rna_path)
    atac = ad.read_h5ad(atac_path)
    if not rna.obs_names.equals(atac.obs_names):
        raise ValueError("RNA and ATAC cell order differs.")

    rna = rna[:, nonzero_features(rna.X)].copy()
    atac = atac[:, nonzero_features(atac.X)].copy()

    rna_hvg = rna.copy()
    sc.pp.normalize_total(rna_hvg)
    sc.pp.log1p(rna_hvg)
    sc.pp.highly_variable_genes(rna_hvg, n_top_genes=n_top_genes)
    rna = rna[:, np.asarray(rna_hvg.var["highly_variable"])].copy()

    min_cells = int(np.ceil(atac.n_obs * min_atac_fraction))
    atac = atac[:, np.asarray((atac.X > 0).sum(axis=0)).ravel() >= min_cells].copy()

    rna.var_names = ["RNA:" + str(x) for x in rna.var_names]
    atac.var_names = ["ATAC:" + str(x) for x in atac.var_names]
    x = sp.hstack([rna.X, atac.X], format="csr", dtype=np.float32)
    var = rna.var.iloc[:, 0:0].copy()
    var = var.reindex(rna.var_names.append(atac.var_names))
    multi = ad.AnnData(X=x, obs=rna.obs.copy(), var=var)
    multi.obs["modality"] = "paired"
    multi.obs["modality"] = multi.obs["modality"].astype("category")
    return multi, rna.n_vars, atac.n_vars


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    scvi.settings.seed = args.seed

    started = time.time()
    print(f"Preparing paired {args.dataset_name} data for MultiVI...", flush=True)
    multi, n_genes, n_regions = prepare_multiome(
        args.rna_h5ad,
        args.atac_h5ad,
        args.n_top_genes,
        args.min_atac_cell_fraction,
    )
    print(
        f"Cells: {multi.n_obs:,} | RNA features: {n_genes:,} | "
        f"ATAC features: {n_regions:,}",
        flush=True,
    )

    setup_kwargs = {"batch_key": "modality"}
    if args.categorical_covariate_key:
        if args.categorical_covariate_key not in multi.obs:
            raise KeyError(
                f"Missing categorical covariate: {args.categorical_covariate_key}"
            )
        setup_kwargs["categorical_covariate_keys"] = [
            args.categorical_covariate_key
        ]
    scvi.data.setup_anndata(multi, **setup_kwargs)
    model = MULTIVI(
        multi,
        n_genes=n_genes,
        n_regions=n_regions,
        n_hidden=args.hidden_dim,
        n_latent=args.latent_dim,
        fully_paired=False,
    )
    print(
        f"Training MultiVI: max epochs={args.max_epochs}, "
        f"KL warmup={args.warmup_epochs}, batch size={args.batch_size}",
        flush=True,
    )
    model.train(
        max_epochs=args.max_epochs,
        use_gpu=torch.cuda.is_available(),
        batch_size=args.batch_size,
        n_epochs_kl_warmup=args.warmup_epochs,
        early_stopping=True,
        save_best=True,
        check_val_every_n_epoch=1,
        progress_bar_refresh_rate=1,
    )

    print("Extracting joint and modality-specific latent representations...", flush=True)
    integrated = model.get_latent_representation(modality="joint", give_mean=True)
    rna_latent = model.get_latent_representation(modality="expression", give_mean=True)
    atac_latent = model.get_latent_representation(modality="accessibility", give_mean=True)
    np.savez_compressed(
        os.path.join(args.output_dir, "嵌入.npz"),
        integrated=integrated,
        rna=rna_latent,
        atac=atac_latent,
        modality2=atac_latent,
        modality2_name=np.asarray("ATAC"),
        cell_id=np.asarray(multi.obs_names, dtype=str),
    )
    torch.save(model.module.state_dict(), os.path.join(args.output_dir, "模型状态.pt"))
    metadata = {
        **vars(args),
        "method": "MultiVI",
        "scvi_tools_version": scvi.__version__,
        "n_cells": multi.n_obs,
        "n_genes": n_genes,
        "n_regions": n_regions,
        "epochs_trained": int(model.history["elbo_train"].shape[0]),
        "elapsed_seconds": time.time() - started,
    }
    with open(os.path.join(args.output_dir, "运行元数据.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Finished in {metadata['elapsed_seconds'] / 60:.1f} minutes.", flush=True)


if __name__ == "__main__":
    main()
