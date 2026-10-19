import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse


SOURCE_DIR = (
    Path("dataset") / "GLUE_10x_Multiome_Pbmc10k" / "scAPI_ready"
)
OUTPUT_DIR = Path("dataset") / "scLAMP_SR_ready" / "GLUE_PBMC10k"
FILES = {
    "RNA": SOURCE_DIR / "glue_pbmc10k_rna_scapi.h5ad",
    "ATAC": SOURCE_DIR / "glue_pbmc10k_atac_top20k_scapi.h5ad",
}


def integer_csr(matrix):
    matrix = sparse.csr_matrix(matrix)
    if matrix.data.size:
        if np.any(matrix.data < 0) or not np.allclose(matrix.data, np.rint(matrix.data)):
            raise ValueError("Expected nonnegative integer counts.")
    return sparse.csr_matrix(
        (
            np.rint(matrix.data).astype(np.int32),
            matrix.indices.astype(np.int32, copy=False),
            matrix.indptr.astype(np.int64, copy=False),
        ),
        shape=matrix.shape,
    )


def standardize(adata, modality, source_path=None):
    adata.X = integer_csr(adata.X)
    pair_ids = adata.obs_names.astype(str)
    adata.obs["pair_id"] = pair_ids
    adata.obs["dataset_id"] = "GLUE_PBMC10k"
    adata.obs["source_barcode"] = pair_ids
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
    adata.obs["label"] = adata.obs["label"].astype(str)
    adata.obs["batch"] = (
        adata.obs["orig.ident"].astype(str)
        if "orig.ident" in adata.obs
        else "unknown"
    )
    adata.obs["donor_id"] = "unknown"
    adata.obs["tissue"] = "peripheral blood"

    feature_names = adata.var_names.astype(str)
    adata.var["feature_name"] = feature_names
    adata.var["source_feature_name"] = feature_names
    adata.var["modality"] = modality
    adata.var["is_technical_control"] = False
    adata.var["use_for_model"] = True
    adata.uns["dataset_id"] = "GLUE_PBMC10k"
    adata.uns["modality"] = modality
    adata.uns["count_scale"] = "raw_integer"
    adata.uns["source_file"] = str(Path(source_path or FILES[modality]).resolve())
    return adata


def summarize(adata):
    return {
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "nnz": int(adata.X.nnz),
        "zero_fraction": float(
            1.0 - adata.X.nnz / (adata.n_obs * adata.n_vars)
        ),
        "max_count": int(adata.X.data.max()) if adata.X.nnz else 0,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standardize the GLUE-processed 10x PBMC Multiome pair."
    )
    parser.add_argument("--rna-h5ad", type=Path, default=FILES["RNA"])
    parser.add_argument("--atac-h5ad", type=Path, default=FILES["ATAC"])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    files = {"RNA": args.rna_h5ad, "ATAC": args.atac_h5ad}
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    summaries = {}
    obs_names = None

    for modality, source_path in files.items():
        adata = standardize(ad.read_h5ad(source_path), modality, source_path=source_path)
        if obs_names is None:
            obs_names = adata.obs_names.copy()
        elif not obs_names.equals(adata.obs_names):
            raise ValueError("RNA and ATAC cells or cell order do not match.")

        output_path = output_dir / f"{modality.lower()}.h5ad"
        adata.write_h5ad(output_path, compression="gzip")
        outputs[modality.lower()] = str(output_path.resolve())
        summaries[modality.lower()] = summarize(adata)

    manifest = {
        "dataset_id": "GLUE_PBMC10k",
        "source_files": {
            name.lower(): str(path.resolve()) for name, path in files.items()
        },
        "outputs": outputs,
        "paired_cells": int(len(obs_names)),
        "feature_selection": {
            "atac": "Existing top 20,000 peaks by total count",
            "rna": "Existing scAPI-ready RNA feature set",
        },
        **summaries,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
