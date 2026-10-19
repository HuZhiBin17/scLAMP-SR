import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse
import torch
from torch.utils.data import DataLoader, TensorDataset


REQUIRED_OBS = {
    "pair_id",
    "dataset_id",
    "cell_type",
    "label",
    "batch",
    "donor_id",
    "tissue",
    "source_barcode",
}
REQUIRED_VAR = {
    "feature_name",
    "modality",
    "source_feature_name",
    "is_technical_control",
    "use_for_model",
}


def matrix_summary(matrix):
    if not sparse.isspmatrix_csr(matrix):
        raise ValueError("X must be stored as a CSR sparse matrix.")
    if matrix.data.size:
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("X contains non-finite values.")
        if np.any(matrix.data < 0):
            raise ValueError("X contains negative values.")
        if not np.allclose(matrix.data, np.rint(matrix.data)):
            raise ValueError("X contains non-integer values.")

    row_totals = np.asarray(matrix.sum(axis=1)).ravel()
    return {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "nnz": int(matrix.nnz),
        "zero_fraction": float(1.0 - matrix.nnz / np.prod(matrix.shape)),
        "all_zero_cells": int(np.count_nonzero(row_totals == 0)),
        "median_cell_total": float(np.median(row_totals)),
        "max_count": int(matrix.data.max()) if matrix.nnz else 0,
    }


def validate_metadata(adata, modality):
    missing_obs = sorted(REQUIRED_OBS.difference(adata.obs.columns))
    missing_var = sorted(REQUIRED_VAR.difference(adata.var.columns))
    if missing_obs:
        raise ValueError(f"{modality}: missing obs columns: {missing_obs}")
    if missing_var:
        raise ValueError(f"{modality}: missing var columns: {missing_var}")
    if not adata.obs_names.is_unique:
        raise ValueError(f"{modality}: obs_names are not unique.")
    if not adata.var_names.is_unique:
        raise ValueError(f"{modality}: var_names are not unique.")
    if not np.array_equal(
        adata.obs["pair_id"].astype(str).to_numpy(),
        adata.obs_names.astype(str).to_numpy(),
    ):
        raise ValueError(f"{modality}: pair_id does not match obs_names.")


def batch_smoke_test(rna, second, n_cells=100):
    n_cells = min(n_cells, rna.n_obs)
    rna_sample = torch.from_numpy(rna.X[:n_cells].toarray()).float()
    second_sample = torch.from_numpy(second.X[:n_cells].toarray()).float()
    loader = DataLoader(
        TensorDataset(rna_sample, second_sample),
        batch_size=min(32, n_cells),
        shuffle=False,
    )
    rna_batch, second_batch = next(iter(loader))
    return {
        "n_cells_materialized": n_cells,
        "batch_size": int(rna_batch.shape[0]),
        "rna_features": int(rna_batch.shape[1]),
        "second_modality_features": int(second_batch.shape[1]),
        "finite": bool(
            torch.isfinite(rna_batch).all() and torch.isfinite(second_batch).all()
        ),
    }


def validate_pair(dataset_dir):
    rna_path = dataset_dir / "rna.h5ad"
    candidates = [
        ("ATAC", dataset_dir / "atac.h5ad"),
        ("ADT", dataset_dir / "adt.h5ad"),
    ]
    available = [(name, path) for name, path in candidates if path.exists()]
    if not rna_path.exists() or len(available) != 1:
        raise ValueError(
            f"{dataset_dir}: expected rna.h5ad and exactly one of atac.h5ad/adt.h5ad."
        )

    second_name, second_path = available[0]
    rna = ad.read_h5ad(rna_path)
    second = ad.read_h5ad(second_path)
    validate_metadata(rna, "RNA")
    validate_metadata(second, second_name)

    if not rna.obs_names.equals(second.obs_names):
        raise ValueError("Paired modality obs_names or cell order do not match.")

    rna_summary = matrix_summary(rna.X)
    second_summary = matrix_summary(second.X)
    if rna_summary["all_zero_cells"] or second_summary["all_zero_cells"]:
        raise ValueError(
            "All-zero cells remain: "
            f"RNA={rna_summary['all_zero_cells']}, "
            f"{second_name}={second_summary['all_zero_cells']}."
        )

    summary = {
        "dataset_id": dataset_dir.name,
        "status": "passed",
        "modalities": ["RNA", second_name],
        "paired_cells": int(rna.n_obs),
        "rna": rna_summary,
        second_name.lower(): second_summary,
        "cell_type_counts": (
            rna.obs["cell_type"].astype(str).value_counts().sort_index().to_dict()
        ),
        "batch_counts": (
            rna.obs["batch"].astype(str).value_counts().sort_index().to_dict()
        ),
        "donor_counts": (
            rna.obs["donor_id"].astype(str).value_counts().sort_index().to_dict()
        ),
        "tissue_counts": (
            rna.obs["tissue"].astype(str).value_counts().sort_index().to_dict()
        ),
        "batch_smoke_test": batch_smoke_test(rna, second),
    }
    output_path = dataset_dir / "qc_summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate standardized paired scLAMP-SR h5ad datasets."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("dataset") / "scLAMP_SR_ready",
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dirs = (
        [args.root / name for name in args.datasets]
        if args.datasets
        else sorted(path for path in args.root.iterdir() if path.is_dir())
    )
    for dataset_dir in dataset_dirs:
        validate_pair(dataset_dir)


if __name__ == "__main__":
    main()
