import argparse
import contextlib
import json
import random
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
import tqdm.auto  # Import before scvi for the pinned torchmetrics dependency.

import scvi
from scvi.model import TOTALVI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna-h5ad", required=True)
    parser.add_argument("--adt-h5ad", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def nonzero_features(matrix):
    return np.asarray(matrix.sum(axis=0)).ravel() > 0


def prepare_cite_seq(rna_path, adt_path):
    rna = ad.read_h5ad(rna_path)
    adt = ad.read_h5ad(adt_path)
    if not rna.obs_names.equals(adt.obs_names):
        raise ValueError("RNA and ADT cell order differs.")
    rna = rna[:, nonzero_features(rna.X)].copy()
    adt = adt[:, nonzero_features(adt.X)].copy()
    protein = adt.X.toarray() if sp.issparse(adt.X) else np.asarray(adt.X)
    rna.obsm["protein_expression"] = np.asarray(protein, dtype=np.float32)
    rna.uns["protein_names"] = np.asarray(adt.var_names, dtype=str)
    return rna, adt.n_vars


@contextlib.contextmanager
def suppress_windows_mkl_kmeans_warning():
    """Ignore only sklearn's known Windows/MKL thread-count warning."""
    original_warn = warnings.warn

    def filtered_warn(message, *args, **kwargs):
        if "KMeans is known to have a memory leak on Windows with MKL" in str(message):
            return None
        return original_warn(message, *args, **kwargs)

    warnings.warn = filtered_warn
    try:
        yield
    finally:
        warnings.warn = original_warn


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    scvi.settings.seed = args.seed

    started = time.time()
    print(f"Preparing paired {args.dataset_name} data for totalVI...", flush=True)
    cite, n_proteins = prepare_cite_seq(args.rna_h5ad, args.adt_h5ad)
    print(
        f"Cells: {cite.n_obs:,} | RNA features: {cite.n_vars:,} | "
        f"ADT features: {n_proteins:,}",
        flush=True,
    )

    if args.batch_key and args.batch_key not in cite.obs:
        raise KeyError(f"Missing batch column: {args.batch_key}")
    scvi.data.setup_anndata(
        cite,
        batch_key=args.batch_key or None,
        protein_expression_obsm_key="protein_expression",
    )
    with suppress_windows_mkl_kmeans_warning():
        model = TOTALVI(
            cite,
            n_latent=args.latent_dim,
            n_hidden=args.hidden_dim,
            gene_likelihood="nb",
        )
    print(
        f"Training totalVI: max epochs={args.max_epochs}, "
        f"KL warmup={args.warmup_epochs}, batch size={args.batch_size}",
        flush=True,
    )
    model.train(
        max_epochs=args.max_epochs,
        use_gpu=torch.cuda.is_available(),
        batch_size=args.batch_size,
        n_epochs_kl_warmup=args.warmup_epochs,
        early_stopping=True,
        check_val_every_n_epoch=1,
        progress_bar_refresh_rate=1,
    )

    print("Extracting totalVI joint latent representation...", flush=True)
    integrated = model.get_latent_representation(give_mean=True)
    np.savez_compressed(
        output_dir / "嵌入.npz",
        integrated=integrated,
        cell_id=np.asarray(cite.obs_names, dtype=str),
    )
    torch.save(model.module.state_dict(), output_dir / "模型状态.pt")
    metadata = {
        **vars(args),
        "method": "totalVI",
        "scvi_tools_version": scvi.__version__,
        "n_cells": cite.n_obs,
        "n_genes": cite.n_vars,
        "n_proteins": n_proteins,
        "epochs_trained": int(model.history["elbo_train"].shape[0]),
        "elapsed_seconds": time.time() - started,
        "modality_specific_embedding": False,
    }
    with (output_dir / "运行元数据.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"Finished in {metadata['elapsed_seconds'] / 60:.1f} minutes.", flush=True)


if __name__ == "__main__":
    main()
