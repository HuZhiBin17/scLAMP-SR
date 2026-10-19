import argparse
import gc
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import rdata
from scipy import sparse


DATASET_ID = "HUMAN_CORTEX_CP"
DEFAULT_RAW_DIR = (
    Path("dataset") / "Human_Cortex_GSE204684" / "raw"
)
DEFAULT_ANNOTATION = Path("dataset") / "human_cortex.h5ad"
DEFAULT_OUTPUT_DIR = (
    Path("dataset") / "scLAMP_SR_ready" / DATASET_ID
)


def elapsed(start):
    return f"{time.time() - start:.1f}s"


def log(message):
    print(message, flush=True)


def load_cortical_plate_obs(annotation_path):
    annotation = ad.read_h5ad(annotation_path, backed="r")
    try:
        tissue = annotation.obs["tissue"].astype(str)
        keep = tissue.eq("cortical plate").to_numpy()
        obs = annotation.obs.iloc[np.flatnonzero(keep)].copy()
        obs_names = annotation.obs_names[keep].astype(str)
    finally:
        annotation.file.close()

    if len(obs) != 14336:
        raise ValueError(
            f"Expected 14,336 cortical plate cells, found {len(obs):,}."
        )

    obs.index = pd.Index(obs_names, name="pair_id")
    obs["ontology_cell_type"] = obs["cell_type"].astype(str)
    obs["cell_type"] = obs["author_cell_type"].astype(str)
    obs["label"] = obs["cell_type"].astype(str)
    obs["pair_id"] = obs.index.astype(str)
    obs["dataset_id"] = DATASET_ID
    obs["batch"] = obs["batch"].astype(str)
    obs["donor_id"] = obs["donor_id"].astype(str)
    obs["tissue"] = "cortical plate"
    obs["source_barcode"] = [
        value.split("_", 1)[1] if "_" in value else value
        for value in obs.index.astype(str)
    ]
    return obs


def convert_r_object_attributes(parsed):
    if parsed.object.info.type.name != "S4":
        raise TypeError(
            f"Expected an S4 sparse matrix, got {parsed.object.info.type}."
        )
    slots = rdata.conversion.convert(parsed.object.attributes)
    required = {"i", "p", "x", "Dim", "Dimnames"}
    missing = sorted(required.difference(slots))
    if missing:
        raise ValueError(f"R sparse matrix is missing slots: {missing}")
    return slots


def normalize_names(values):
    names = []
    for value in values:
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))
    return names


def read_selected_dgc_matrix(rds_path, target_cells):
    start = time.time()
    log(f"[RDS] Parsing {rds_path}")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unknown file type: assumed RDS",
        )
        parsed = rdata.parser.parse_file(rds_path)
    slots = convert_r_object_attributes(parsed)

    shape = tuple(int(value) for value in slots["Dim"])
    row_names = normalize_names(slots["Dimnames"][0])
    column_names = normalize_names(slots["Dimnames"][1])
    if len(row_names) != shape[0] or len(column_names) != shape[1]:
        raise ValueError(
            "RDS dimensions do not agree with Dimnames: "
            f"shape={shape}, rows={len(row_names)}, columns={len(column_names)}."
        )
    if len(set(column_names)) != len(column_names):
        raise ValueError("RDS cell names are not unique.")

    cell_to_index = {name: index for index, name in enumerate(column_names)}
    missing = [name for name in target_cells if name not in cell_to_index]
    if missing:
        raise ValueError(
            f"{len(missing):,} cortical plate cells are absent from RDS; "
            f"first missing={missing[0]!r}."
        )
    selected_columns = np.asarray(
        [cell_to_index[name] for name in target_cells],
        dtype=np.int64,
    )

    source = sparse.csc_matrix(
        (
            np.asarray(slots["x"]),
            np.asarray(slots["i"]),
            np.asarray(slots["p"]),
        ),
        shape=shape,
    )
    selected = source[:, selected_columns].transpose().tocsr()
    selected = integer_csr(selected)

    log(
        f"[RDS] Selected matrix {selected.shape[0]:,} x "
        f"{selected.shape[1]:,}, nnz={selected.nnz:,}, elapsed={elapsed(start)}"
    )
    del source, slots, parsed
    gc.collect()
    return selected, row_names


def integer_csr(matrix):
    matrix = sparse.csr_matrix(matrix)
    if matrix.data.size:
        if np.any(~np.isfinite(matrix.data)):
            raise ValueError("Count matrix contains non-finite values.")
        if np.any(matrix.data < 0):
            raise ValueError("Count matrix contains negative values.")
        if not np.allclose(matrix.data, np.rint(matrix.data)):
            raise ValueError("Count matrix contains non-integer values.")
    return sparse.csr_matrix(
        (
            np.rint(matrix.data).astype(np.int32),
            matrix.indices.astype(np.int32, copy=False),
            matrix.indptr.astype(np.int64, copy=False),
        ),
        shape=matrix.shape,
    )


def make_var(feature_names, modality):
    source_names = pd.Index(feature_names, dtype="object")
    var = pd.DataFrame(index=source_names.copy())
    var.index.name = "features"
    var["feature_name"] = source_names.astype(str)
    var["source_feature_name"] = source_names.astype(str)
    var["modality"] = modality
    var["is_technical_control"] = False
    var["use_for_model"] = True
    return var


def attach_common_uns(adata, modality, source_path):
    adata.uns["dataset_id"] = DATASET_ID
    adata.uns["modality"] = modality
    adata.uns["count_scale"] = "source_integer_counts"
    adata.uns["source_file"] = str(source_path.resolve())
    adata.uns["cell_filter"] = 'tissue == "cortical plate"'


def summarize_matrix(matrix):
    row_totals = np.asarray(matrix.sum(axis=1)).ravel()
    return {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "nnz": int(matrix.nnz),
        "zero_fraction": float(
            1.0 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])
        ),
        "all_zero_cells": int(np.count_nonzero(row_totals == 0)),
        "median_cell_total": float(np.median(row_totals)),
        "max_count": int(matrix.data.max()) if matrix.nnz else 0,
    }


def write_h5ad_atomic(adata, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp.h5ad")
    if temporary.exists():
        temporary.unlink()
    log(f"[WRITE] {output_path}")
    adata.write_h5ad(temporary, compression="gzip")
    temporary.replace(output_path)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_rna(args):
    start = time.time()
    obs = load_cortical_plate_obs(args.annotation)
    source_path = args.raw_dir / "GSE204683_count_matrix.RDS.gz"
    matrix, genes = read_selected_dgc_matrix(source_path, obs.index.tolist())
    var = make_var(genes, "RNA")
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
        adata.var["feature_name"] = adata.var_names.astype(str)
    attach_common_uns(adata, "RNA", source_path)

    output_path = args.output_dir / "rna.h5ad"
    write_h5ad_atomic(adata, output_path)
    summary = summarize_matrix(adata.X)
    summary.update(
        {
            "source": str(source_path.resolve()),
            "output": str(output_path.resolve()),
            "elapsed_seconds": float(time.time() - start),
        }
    )
    write_json(args.output_dir / "conversion_rna_summary.json", summary)
    log(f"[RNA] Complete, elapsed={elapsed(start)}")


def load_peak_metadata(peaks_path, expected_features, rds_feature_names):
    peaks = pd.read_csv(
        peaks_path,
        sep="\t",
        header=None,
        names=["chromosome", "start", "end", "peak_id"],
        compression="gzip",
    )
    if len(peaks) != expected_features:
        raise ValueError(
            f"Peak BED has {len(peaks):,} rows, expected {expected_features:,}."
        )
    peak_ids = peaks["peak_id"].astype(str).tolist()
    if rds_feature_names:
        rds_names = normalize_names(rds_feature_names)
        if rds_names != peak_ids:
            mismatch = next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(rds_names, peak_ids)
                    )
                    if left != right
                ),
                None,
            )
            raise ValueError(
                "RDS peak names do not match BED order; "
                f"first mismatch index={mismatch}."
            )
    peaks.index = pd.Index(peak_ids, name="features")
    peaks["feature_name"] = peak_ids
    peaks["source_feature_name"] = peak_ids
    peaks["modality"] = "ATAC"
    peaks["is_technical_control"] = False
    return peaks


def select_top_features(matrix, n_features):
    totals = np.asarray(matrix.sum(axis=0)).ravel().astype(np.int64)
    n_features = min(int(n_features), matrix.shape[1])
    ranked = np.lexsort((np.arange(matrix.shape[1]), -totals))
    selected_by_rank = ranked[:n_features]
    selection_rank = np.full(matrix.shape[1], -1, dtype=np.int32)
    selection_rank[selected_by_rank] = np.arange(1, n_features + 1)
    selected_in_source_order = np.sort(selected_by_rank)
    return totals, selection_rank, selected_in_source_order


def run_atac(args):
    start = time.time()
    obs = load_cortical_plate_obs(args.annotation)
    source_path = args.raw_dir / "GSE204682_count_matrix.RDS.gz"
    peaks_path = args.raw_dir / "GSE204682_peaks.bed.gz"
    matrix, rds_peak_names = read_selected_dgc_matrix(
        source_path,
        obs.index.tolist(),
    )
    peaks = load_peak_metadata(
        peaks_path,
        matrix.shape[1],
        rds_peak_names,
    )
    totals, selection_rank, selected = select_top_features(
        matrix,
        args.atac_features,
    )
    peaks["total_count_cortical_plate"] = totals
    peaks["selection_rank"] = selection_rank
    peaks["use_for_model"] = selection_rank > 0

    mask_path = args.output_dir / "atac_feature_mask.csv.gz"
    mask_columns = [
        "feature_name",
        "chromosome",
        "start",
        "end",
        "total_count_cortical_plate",
        "selection_rank",
        "use_for_model",
    ]
    peaks.loc[:, mask_columns].to_csv(
        mask_path,
        index=True,
        compression="gzip",
    )

    full_path = args.output_dir / "atac_full.h5ad"
    if not args.skip_full_atac:
        full = ad.AnnData(X=matrix, obs=obs.copy(), var=peaks.copy())
        attach_common_uns(full, "ATAC", source_path)
        full.uns["feature_selection"] = "all cortical plate peaks"
        write_h5ad_atomic(full, full_path)
        del full
        gc.collect()

    selected_matrix = matrix[:, selected].tocsr()
    selected_var = peaks.iloc[selected].copy()
    selected_var["use_for_model"] = True
    selected_adata = ad.AnnData(
        X=selected_matrix,
        obs=obs.copy(),
        var=selected_var,
    )
    attach_common_uns(selected_adata, "ATAC", source_path)
    selected_adata.uns["feature_selection"] = (
        f"top {len(selected):,} peaks by total count across all "
        "14,336 cortical plate cells"
    )
    selected_adata.uns["feature_mask_file"] = str(mask_path.resolve())

    output_path = args.output_dir / "atac.h5ad"
    write_h5ad_atomic(selected_adata, output_path)
    summary = {
        "source": str(source_path.resolve()),
        "peak_bed": str(peaks_path.resolve()),
        "output": str(output_path.resolve()),
        "full_output": (
            str(full_path.resolve()) if not args.skip_full_atac else None
        ),
        "feature_mask": str(mask_path.resolve()),
        "full": summarize_matrix(matrix),
        "selected": summarize_matrix(selected_matrix),
        "selection": {
            "n_features": int(len(selected)),
            "criterion": (
                "total count across all retained cortical plate cells"
            ),
            "tie_breaker": "source feature order",
        },
        "elapsed_seconds": float(time.time() - start),
    }
    write_json(args.output_dir / "conversion_atac_summary.json", summary)
    log(f"[ATAC] Complete, elapsed={elapsed(start)}")


def validate_pair_files(rna, atac):
    if rna.n_obs != 14336 or atac.n_obs != 14336:
        raise ValueError(
            f"Expected 14,336 paired cells, got RNA={rna.n_obs:,}, "
            f"ATAC={atac.n_obs:,}."
        )
    if not rna.obs_names.equals(atac.obs_names):
        raise ValueError("RNA and ATAC cell names or order do not match.")
    for modality, adata in [("RNA", rna), ("ATAC", atac)]:
        if not sparse.isspmatrix_csr(adata.X):
            raise ValueError(f"{modality} X is not CSR.")
        if adata.X.data.size:
            if np.any(adata.X.data < 0):
                raise ValueError(f"{modality} X contains negative counts.")
            if not np.allclose(adata.X.data, np.rint(adata.X.data)):
                raise ValueError(f"{modality} X contains non-integer counts.")


def correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return float(np.corrcoef(left, right)[0, 1])


def set_count_scale_in_place(path):
    with h5py.File(path, "r+") as handle:
        uns = handle["uns"]
        if "count_scale" in uns:
            del uns["count_scale"]
        dataset = uns.create_dataset(
            "count_scale",
            data="source_integer_counts",
        )
        dataset.attrs["encoding-type"] = "string"
        dataset.attrs["encoding-version"] = "0.2.0"


def source_concordance(args, rna, atac):
    annotation = ad.read_h5ad(args.annotation, backed="r")
    try:
        keep = np.flatnonzero(
            annotation.obs["tissue"]
            .astype(str)
            .eq("cortical plate")
            .to_numpy()
        )
        annotation_names = annotation.obs_names[keep].astype(str)
        if not rna.obs_names.equals(annotation_names):
            raise ValueError(
                "Output cells do not match cortical plate annotation order."
            )
        cellxgene_raw = annotation.raw.X[keep]
        cellxgene_totals = np.asarray(
            cellxgene_raw.sum(axis=1)
        ).ravel()
    finally:
        annotation.file.close()

    rna_totals = np.asarray(rna.X.sum(axis=1)).ravel()
    rna_qc_totals = rna.obs["nCount_RNA"].to_numpy()
    rna_difference = np.abs(rna_totals - cellxgene_totals)

    full_atac_path = args.output_dir / "atac_full.h5ad"
    full_atac = (
        ad.read_h5ad(full_atac_path)
        if full_atac_path.exists()
        else None
    )
    if full_atac is not None:
        if not rna.obs_names.equals(full_atac.obs_names):
            raise ValueError(
                "Full ATAC archive does not match paired cell order."
            )
        atac_totals = np.asarray(full_atac.X.sum(axis=1)).ravel()
    else:
        atac_totals = np.asarray(atac.X.sum(axis=1)).ravel()
    atac_qc_totals = atac.obs["nCount_ATAC"].to_numpy()

    return {
        "rna_geo_vs_cellxgene_raw_total_correlation": correlation(
            rna_totals,
            cellxgene_totals,
        ),
        "rna_geo_vs_cellxgene_raw_median_absolute_difference": float(
            np.median(rna_difference)
        ),
        "rna_geo_vs_cellxgene_raw_max_absolute_difference": float(
            np.max(rna_difference)
        ),
        "rna_geo_vs_upstream_nCount_RNA_correlation": correlation(
            rna_totals,
            rna_qc_totals,
        ),
        "atac_geo_vs_upstream_nCount_ATAC_correlation": correlation(
            atac_totals,
            atac_qc_totals,
        ),
        "interpretation": (
            "RNA GEO RDS agrees with the CELLxGENE raw layer but not with "
            "upstream per-cell UMI totals. The paper used SCTransform, so "
            "these source-provided integer values must not be described as "
            "verified raw Cell Ranger UMI counts."
        ),
    }


def run_finalize(args):
    start = time.time()
    rna_path = args.output_dir / "rna.h5ad"
    atac_path = args.output_dir / "atac.h5ad"
    for path in [
        rna_path,
        atac_path,
        args.output_dir / "atac_full.h5ad",
    ]:
        if path.exists():
            set_count_scale_in_place(path)
    rna = ad.read_h5ad(rna_path)
    atac = ad.read_h5ad(atac_path)
    validate_pair_files(rna, atac)
    concordance = source_concordance(args, rna, atac)

    rna_summary = json.loads(
        (args.output_dir / "conversion_rna_summary.json").read_text(
            encoding="utf-8"
        )
    )
    atac_summary = json.loads(
        (args.output_dir / "conversion_atac_summary.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = {
        "dataset_id": DATASET_ID,
        "description": "Human cortical plate paired RNA+ATAC",
        "paired_cells": int(rna.n_obs),
        "cell_filter": 'tissue == "cortical plate"',
        "annotation_source": str(args.annotation.resolve()),
        "source_files": {
            "rna": str(
                (
                    args.raw_dir / "GSE204683_count_matrix.RDS.gz"
                ).resolve()
            ),
            "atac": str(
                (
                    args.raw_dir / "GSE204682_count_matrix.RDS.gz"
                ).resolve()
            ),
            "peaks": str(
                (args.raw_dir / "GSE204682_peaks.bed.gz").resolve()
            ),
        },
        "outputs": {
            "rna": str(rna_path.resolve()),
            "atac": str(atac_path.resolve()),
            "atac_full": atac_summary["full_output"],
            "atac_feature_mask": atac_summary["feature_mask"],
        },
        "rna": rna_summary,
        "atac": atac_summary,
        "metadata": {
            "cell_type_field": "author_cell_type",
            "ontology_cell_type_field": "ontology_cell_type",
            "label_field": "author_cell_type",
        },
        "count_scale": "source_integer_counts",
        "count_provenance": {
            "rna": (
                "Nonnegative integer matrix supplied in the GEO RDS and "
                "CELLxGENE raw layer; not verified raw Cell Ranger UMI."
            ),
            "atac": "Nonnegative integer peak-count matrix supplied by GEO.",
            "source_concordance": concordance,
        },
        "paired_cell_order_verified": True,
        "rdata_version": "0.9",
    }
    write_json(args.output_dir / "manifest.json", manifest)
    log(
        f"[FINALIZE] Validated {rna.n_obs:,} paired cells; "
        f"RNA={rna.n_vars:,}, ATAC={atac.n_vars:,}, "
        f"elapsed={elapsed(start)}"
    )


def stage_command(args, stage):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        stage,
        "--raw-dir",
        str(args.raw_dir),
        "--annotation",
        str(args.annotation),
        "--output-dir",
        str(args.output_dir),
        "--atac-features",
        str(args.atac_features),
    ]
    if args.skip_full_atac:
        command.append("--skip-full-atac")
    return command


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert Human Cortex RNA/ATAC RDS matrices to paired "
            "scLAMP-SR h5ad files."
        )
    )
    parser.add_argument(
        "--stage",
        choices=["all", "rna", "atac", "finalize"],
        default="all",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=DEFAULT_ANNOTATION,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--atac-features", type=int, default=20000)
    parser.add_argument("--skip-full-atac", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        for stage in ["rna", "atac", "finalize"]:
            log(f"[ALL] Starting isolated stage: {stage}")
            subprocess.run(stage_command(args, stage), check=True)
        return
    if args.stage == "rna":
        run_rna(args)
    elif args.stage == "atac":
        run_atac(args)
    else:
        run_finalize(args)


if __name__ == "__main__":
    main()
