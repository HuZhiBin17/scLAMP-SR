import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


DATASETS = {
    "BMNC": {
        "filename": "CITEseq_GSE128639_BMNC_anno.h5",
        "organism": "Homo sapiens",
    },
    "SLN111": {
        "filename": "CITEseq_realdata_spleen_lymph_111_anno.h5",
        "organism": "Mus musculus",
    },
    "SLN208": {
        "filename": "CITEseq_realdata_spleen_lymph_206_anno.h5",
        "organism": "Mus musculus",
    },
}


def decode_strings(values):
    return np.asarray(values).astype("U")


def dense_h5_to_csr(dataset, chunk_rows=512):
    data_parts = []
    index_parts = []
    indptr_parts = [np.array([0], dtype=np.int64)]
    nnz = 0

    for start in range(0, dataset.shape[0], chunk_rows):
        stop = min(start + chunk_rows, dataset.shape[0])
        block = np.asarray(dataset[start:stop])
        if not np.all(np.isfinite(block)):
            raise ValueError(f"Non-finite values found in rows {start}:{stop}.")
        if np.any(block < 0) or not np.allclose(block, np.rint(block)):
            raise ValueError(f"Expected nonnegative integer counts in rows {start}:{stop}.")

        block_csr = sparse.csr_matrix(np.rint(block).astype(np.int32))
        data_parts.append(block_csr.data)
        index_parts.append(block_csr.indices.astype(np.int32, copy=False))
        indptr_parts.append(block_csr.indptr[1:].astype(np.int64, copy=False) + nnz)
        nnz += block_csr.nnz

    data = np.concatenate(data_parts) if data_parts else np.array([], dtype=np.int32)
    indices = np.concatenate(index_parts) if index_parts else np.array([], dtype=np.int32)
    indptr = np.concatenate(indptr_parts)
    return sparse.csr_matrix((data, indices, indptr), shape=dataset.shape)


def make_unique(values):
    counts = {}
    unique = []
    for value in values:
        seen = counts.get(value, 0)
        unique.append(value if seen == 0 else f"{value}-{seen}")
        counts[value] = seen + 1
    return np.asarray(unique, dtype=str)


def make_obs(handle, dataset_id):
    n_cells = handle["X1"].shape[0]
    if "Barcodes" in handle:
        source_barcodes = decode_strings(handle["Barcodes"][:])
    else:
        source_barcodes = np.asarray([f"cell_{idx:06d}" for idx in range(n_cells)])

    if "Batch" in handle:
        batch_values = np.rint(handle["Batch"][:]).astype(np.int64).astype(str)
    else:
        batch_values = np.full(n_cells, "unknown", dtype=object)

    pair_ids = make_unique(
        np.asarray(
            [
                f"{dataset_id}:{batch}:{barcode}"
                for batch, barcode in zip(batch_values, source_barcodes)
            ]
        )
    )
    cell_types = decode_strings(handle["Celltypes"][:])
    label_codes = np.rint(handle["Y"][:]).astype(np.int64)

    obs = pd.DataFrame(index=pd.Index(pair_ids, name="pair_id"))
    obs["pair_id"] = pair_ids
    obs["dataset_id"] = dataset_id
    obs["source_barcode"] = source_barcodes
    obs["cell_type"] = cell_types
    obs["label"] = cell_types
    obs["source_label_code"] = label_codes
    obs["batch"] = batch_values
    obs["donor_id"] = "unknown"
    obs["tissue"] = "bone marrow" if dataset_id == "BMNC" else "spleen/lymph node"
    return obs


def make_var(feature_names, modality):
    feature_names = decode_strings(feature_names)
    unique_names = make_unique(feature_names)
    upper_names = np.char.upper(feature_names)
    is_hto = np.char.startswith(upper_names, "HTO_")
    is_isotype = np.asarray(["ISOTYPE" in value for value in upper_names])

    var = pd.DataFrame(index=pd.Index(unique_names, name="feature_id"))
    var["feature_name"] = feature_names
    var["source_feature_name"] = feature_names
    var["modality"] = modality
    var["is_hto"] = is_hto
    var["is_isotype_control"] = is_isotype
    var["is_technical_control"] = is_hto | is_isotype
    var["use_for_model"] = ~is_hto
    return var


def summarize_matrix(matrix):
    return {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "nnz": int(matrix.nnz),
        "zero_fraction": float(1.0 - matrix.nnz / np.prod(matrix.shape)),
        "max_count": int(matrix.data.max()) if matrix.nnz else 0,
    }


def convert_one(source_path, output_root, dataset_id, organism):
    output_dir = output_root / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_path, "r") as handle:
        if handle["X1"].shape[0] != handle["X2"].shape[0]:
            raise ValueError(f"RNA/ADT cell count mismatch in {source_path}.")

        obs = make_obs(handle, dataset_id)
        genes = decode_strings(handle["Genes"][:])
        adt_key = "ADT" if "ADT" in handle else "ADTS"
        proteins = decode_strings(handle[adt_key][:])

        rna_matrix = dense_h5_to_csr(handle["X1"])
        adt_all_matrix = dense_h5_to_csr(handle["X2"])

    rna_var = make_var(genes, "RNA")
    adt_all_var = make_var(proteins, "ADT")
    adt_model_mask = adt_all_var["use_for_model"].to_numpy()

    common_uns = {
        "dataset_id": dataset_id,
        "source_file": str(source_path.resolve()),
        "source_format": "scMDC h5",
        "organism": organism,
        "count_scale": "raw_integer",
    }

    rna = ad.AnnData(X=rna_matrix, obs=obs.copy(), var=rna_var)
    rna.uns.update(common_uns)
    rna.uns["modality"] = "RNA"

    adt_all = ad.AnnData(X=adt_all_matrix, obs=obs.copy(), var=adt_all_var)
    adt_all.uns.update(common_uns)
    adt_all.uns["modality"] = "ADT"
    adt_all.uns["includes_hto"] = True

    adt_model = adt_all[:, adt_model_mask].copy()
    adt_model.uns["includes_hto"] = False

    rna_path = output_dir / "rna.h5ad"
    adt_path = output_dir / "adt.h5ad"
    adt_all_path = output_dir / "adt_source_complete.h5ad"
    rna.write_h5ad(rna_path, compression="gzip")
    adt_model.write_h5ad(adt_path, compression="gzip")
    adt_all.write_h5ad(adt_all_path, compression="gzip")

    manifest = {
        "dataset_id": dataset_id,
        "organism": organism,
        "source_file": str(source_path.resolve()),
        "outputs": {
            "rna": str(rna_path.resolve()),
            "adt_model_ready": str(adt_path.resolve()),
            "adt_source_complete": str(adt_all_path.resolve()),
        },
        "rna": summarize_matrix(rna_matrix),
        "adt_source_complete": summarize_matrix(adt_all_matrix),
        "adt_model_ready": summarize_matrix(adt_model.X),
        "n_hto_excluded": int(adt_all_var["is_hto"].sum()),
        "n_isotype_controls_retained": int(
            adt_model.var["is_isotype_control"].sum()
        ),
        "cell_type_counts": (
            obs["cell_type"].value_counts().astype(int).sort_index().to_dict()
        ),
        "batch_counts": obs["batch"].value_counts().astype(int).sort_index().to_dict(),
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps(manifest, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert scMDC CITE-seq h5 files to paired scLAMP h5ad files."
    )
    parser.add_argument("--dataset_dir", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("dataset") / "scLAMP_SR_ready",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for dataset_id in args.datasets:
        config = DATASETS[dataset_id]
        source_path = args.dataset_dir / config["filename"]
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        convert_one(source_path, args.output_root, dataset_id, config["organism"])


if __name__ == "__main__":
    main()
