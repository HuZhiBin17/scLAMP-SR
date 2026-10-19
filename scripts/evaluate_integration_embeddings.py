import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate integrated and modality-specific paired-cell embeddings."
    )
    parser.add_argument("--reference-rna", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--label-column", default="cell_type")
    parser.add_argument("--batch-column", default="batch")
    parser.add_argument("--isolated-max-fraction", type=float, default=0.05)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--silhouette-cells", type=int, default=5000)
    parser.add_argument("--evaluation-seed", type=int, default=0)
    parser.add_argument(
        "--embedding",
        nargs=3,
        action="append",
        metavar=("METHOD", "SEED", "NPZ_PATH"),
        required=True,
        help="Repeat once per method/seed embedding file.",
    )
    return parser.parse_args()


def stratified_sample(labels, target, seed):
    if len(labels) <= target:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        count = max(2, int(round(target * len(indices) / len(labels))))
        selected.extend(
            rng.choice(indices, size=min(count, len(indices)), replace=False)
        )
    selected = np.asarray(selected)
    if len(selected) > target:
        selected = rng.choice(selected, size=target, replace=False)
    return np.sort(selected)


def neighbor_indices(embedding, neighbors):
    embedding = StandardScaler().fit_transform(embedding)
    model = NearestNeighbors(n_neighbors=neighbors + 1).fit(embedding)
    return model.kneighbors(
        embedding,
        n_neighbors=neighbors + 1,
        return_distance=False,
    )[:, 1:]


def normalized_clisi(indices, labels):
    n_labels = len(np.unique(labels))
    if n_labels == 1:
        return 1.0
    inverse_simpson = []
    for values in labels[indices]:
        _, counts = np.unique(values, return_counts=True)
        probabilities = counts / counts.sum()
        inverse_simpson.append(1.0 / np.sum(probabilities**2))
    scaled = (np.asarray(inverse_simpson) - 1.0) / (n_labels - 1.0)
    return float(1.0 - np.mean(scaled))


def graph_connectivity(indices, labels):
    n_cells = len(labels)
    rows = np.repeat(np.arange(n_cells), indices.shape[1])
    columns = indices.ravel()
    graph = csr_matrix(
        (np.ones(len(rows)), (rows, columns)),
        shape=(n_cells, n_cells),
    )
    graph = graph.maximum(graph.T)
    scores = []
    weights = []
    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        subgraph = graph[label_indices][:, label_indices]
        _, components = connected_components(subgraph, directed=False)
        largest = np.bincount(components).max()
        scores.append(largest / len(label_indices))
        weights.append(len(label_indices))
    return float(np.average(scores, weights=weights))


def isolated_label_f1(labels, predicted, maximum_fraction):
    scores = []
    for label in np.unique(labels):
        truth = labels == label
        if truth.mean() > maximum_fraction:
            continue
        best = 0.0
        for cluster in np.unique(predicted):
            best = max(best, f1_score(truth, predicted == cluster, zero_division=0))
        scores.append(best)
    return float(np.mean(scores)) if scores else np.nan


def biological_metrics(
    embedding,
    labels,
    seed,
    neighbors,
    silhouette_cells,
    isolated_max_fraction,
):
    embedding = StandardScaler().fit_transform(embedding)
    predicted = KMeans(
        n_clusters=len(np.unique(labels)),
        n_init=20,
        random_state=seed,
    ).fit_predict(embedding)
    indices = neighbor_indices(embedding, neighbors)
    sample = stratified_sample(labels, silhouette_cells, seed)
    return {
        "ARI": adjusted_rand_score(labels, predicted),
        "NMI": normalized_mutual_info_score(labels, predicted),
        "AMI": adjusted_mutual_info_score(labels, predicted),
        "cell_type_ASW": silhouette_score(embedding[sample], labels[sample]),
        "knn_purity": float(np.mean(labels[indices] == labels[:, None])),
        "cLISI_conservation": normalized_clisi(indices, labels),
        "graph_connectivity": graph_connectivity(indices, labels),
        "isolated_label_F1": isolated_label_f1(
            labels,
            predicted,
            isolated_max_fraction,
        ),
    }


def batch_metrics(embedding, batches, seed, neighbors, silhouette_cells):
    unique_batches, inverse = np.unique(batches, return_inverse=True)
    if len(unique_batches) < 2:
        return {
            "batch_iLISI": np.nan,
            "batch_ASW": np.nan,
            "kBET_acceptance": np.nan,
        }
    embedding = StandardScaler().fit_transform(embedding)
    indices = neighbor_indices(embedding, neighbors)
    inverse_simpson = []
    for values in batches[indices]:
        _, counts = np.unique(values, return_counts=True)
        probabilities = counts / counts.sum()
        inverse_simpson.append(1.0 / np.sum(probabilities**2))
    normalized_ilisi = (
        np.mean(inverse_simpson) - 1.0
    ) / (len(unique_batches) - 1.0)

    rng = np.random.default_rng(seed)
    sample = rng.choice(
        len(embedding),
        size=min(silhouette_cells, len(embedding)),
        replace=False,
    )
    batch_silhouette = silhouette_score(embedding[sample], batches[sample])

    global_proportions = np.bincount(inverse) / len(inverse)
    expected = neighbors * global_proportions
    accepted = []
    for row in indices:
        observed = np.bincount(inverse[row], minlength=len(unique_batches))
        statistic = np.sum((observed - expected) ** 2 / np.maximum(expected, 1e-12))
        accepted.append(chi2.sf(statistic, len(unique_batches) - 1) >= 0.05)
    return {
        "batch_iLISI": float(normalized_ilisi),
        "batch_ASW": float(1.0 - abs(batch_silhouette)),
        "kBET_acceptance": float(np.mean(accepted)),
    }


def batch_metrics_within_labels(
    embedding,
    labels,
    batches,
    seed,
    neighbors,
    silhouette_cells,
):
    """Average batch mixing within biological labels to limit composition confounding."""
    embedding = StandardScaler().fit_transform(embedding)
    rows = []
    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        label_batches = batches[label_indices]
        unique_batches, inverse = np.unique(label_batches, return_inverse=True)
        if len(unique_batches) < 2 or len(label_indices) < 4:
            continue
        local_neighbors = min(neighbors, len(label_indices) - 1)
        if local_neighbors < 2:
            continue
        local_embedding = embedding[label_indices]
        indices = neighbor_indices(local_embedding, local_neighbors)

        inverse_simpson = []
        for values in label_batches[indices]:
            _, counts = np.unique(values, return_counts=True)
            probabilities = counts / counts.sum()
            inverse_simpson.append(1.0 / np.sum(probabilities**2))
        ilisi = (np.mean(inverse_simpson) - 1.0) / (len(unique_batches) - 1.0)

        global_proportions = np.bincount(inverse) / len(inverse)
        expected = local_neighbors * global_proportions
        accepted = []
        for row in indices:
            observed = np.bincount(inverse[row], minlength=len(unique_batches))
            statistic = np.sum(
                (observed - expected) ** 2 / np.maximum(expected, 1e-12)
            )
            accepted.append(chi2.sf(statistic, len(unique_batches) - 1) >= 0.05)

        sample_size = min(silhouette_cells, len(label_indices))
        sample = stratified_sample(label_batches, sample_size, seed + len(rows))
        batch_asw = 1.0 - abs(
            silhouette_score(local_embedding[sample], label_batches[sample])
        )
        rows.append(
            {
                "n_cells": len(label_indices),
                "batch_iLISI_within_cell_type": float(ilisi),
                "batch_ASW_within_cell_type": float(batch_asw),
                "kBET_acceptance_within_cell_type": float(np.mean(accepted)),
            }
        )

    if not rows:
        return {
            "batch_iLISI_within_cell_type": np.nan,
            "batch_ASW_within_cell_type": np.nan,
            "kBET_acceptance_within_cell_type": np.nan,
            "batch_mixing_evaluated_cell_fraction": 0.0,
        }
    frame = pd.DataFrame(rows)
    weights = frame["n_cells"].to_numpy()
    result = {
        metric: float(np.average(frame[metric], weights=weights))
        for metric in (
            "batch_iLISI_within_cell_type",
            "batch_ASW_within_cell_type",
            "kBET_acceptance_within_cell_type",
        )
    }
    result["batch_mixing_evaluated_cell_fraction"] = float(
        weights.sum() / len(labels)
    )
    return result


def cell_type_structure_rows(
    embedding,
    labels,
    method,
    seed,
    dataset,
    neighbors,
):
    embedding = StandardScaler().fit_transform(embedding)
    indices = neighbor_indices(embedding, neighbors)
    n_cells = len(labels)
    graph_rows = np.repeat(np.arange(n_cells), indices.shape[1])
    graph_columns = indices.ravel()
    graph = csr_matrix(
        (np.ones(len(graph_rows)), (graph_rows, graph_columns)),
        shape=(n_cells, n_cells),
    )
    graph = graph.maximum(graph.T)
    rows = []
    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        subgraph = graph[label_indices][:, label_indices]
        n_components, components = connected_components(subgraph, directed=False)
        component_sizes = np.bincount(components)
        minimum_island_size = max(5, int(np.ceil(0.01 * len(label_indices))))
        within_type_neighbors = labels[indices[label_indices]] == label
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "seed": int(seed),
                "cell_type": label,
                "n_cells": len(label_indices),
                "largest_component_fraction": float(
                    component_sizes.max() / len(label_indices)
                ),
                "n_components": int(n_components),
                "n_substantial_islands": int(
                    np.sum(component_sizes >= minimum_island_size)
                ),
                "substantial_island_min_cells": minimum_island_size,
                "within_type_neighbor_fraction": float(
                    np.mean(within_type_neighbors)
                ),
            }
        )
    return rows


def foscttm_and_retrieval(first, second, block_size=256):
    joined = StandardScaler().fit_transform(np.vstack([first, second]))
    first = normalize(joined[: len(first)])
    second = normalize(joined[len(first) :])
    n_cells = len(first)
    first_fraction = np.empty(n_cells, dtype=np.float64)
    second_fraction = np.empty(n_cells, dtype=np.float64)
    first_rank = np.empty(n_cells, dtype=np.int64)
    second_rank = np.empty(n_cells, dtype=np.int64)
    for start in range(0, n_cells, block_size):
        stop = min(start + block_size, n_cells)
        similarity = first[start:stop] @ second.T
        truth = similarity[np.arange(stop - start), np.arange(start, stop)]
        first_rank[start:stop] = np.sum(similarity > truth[:, None], axis=1)
        first_fraction[start:stop] = first_rank[start:stop] / (n_cells - 1)

        reverse = second[start:stop] @ first.T
        reverse_truth = reverse[np.arange(stop - start), np.arange(start, stop)]
        second_rank[start:stop] = np.sum(reverse > reverse_truth[:, None], axis=1)
        second_fraction[start:stop] = second_rank[start:stop] / (n_cells - 1)
    ranks = np.concatenate([first_rank, second_rank])
    return {
        "FOSCTTM": float(np.mean(np.concatenate([first_fraction, second_fraction]))),
        "paired_top1": float(np.mean(ranks == 0)),
        "paired_top10": float(np.mean(ranks < 10)),
    }


def cross_modal_biology_metrics(first, second, labels, neighbors=15, block_size=256):
    """Measure biological and local-geometry agreement across paired modalities."""
    joined = StandardScaler().fit_transform(np.vstack([first, second]))
    first = normalize(joined[: len(first)])
    second = normalize(joined[len(first) :])
    n_cells = len(first)
    k = min(neighbors, n_cells - 1)

    predictions_forward = np.empty(n_cells, dtype=labels.dtype)
    predictions_reverse = np.empty(n_cells, dtype=labels.dtype)
    classes, encoded_labels = np.unique(labels, return_inverse=True)
    for start in range(0, n_cells, block_size):
        stop = min(start + block_size, n_cells)
        similarity = first[start:stop] @ second.T
        similarity[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        forward = np.argpartition(similarity, -k, axis=1)[:, -k:]
        reverse_similarity = second[start:stop] @ first.T
        reverse_similarity[
            np.arange(stop - start), np.arange(start, stop)
        ] = -np.inf
        reverse = np.argpartition(reverse_similarity, -k, axis=1)[:, -k:]
        for offset, indices in enumerate(forward):
            counts = np.bincount(encoded_labels[indices], minlength=len(classes))
            predictions_forward[start + offset] = classes[counts.argmax()]
        for offset, indices in enumerate(reverse):
            counts = np.bincount(encoded_labels[indices], minlength=len(classes))
            predictions_reverse[start + offset] = classes[counts.argmax()]

    first_neighbors = neighbor_indices(first, k)
    second_neighbors = neighbor_indices(second, k)
    jaccard = []
    for left, right in zip(first_neighbors, second_neighbors):
        intersection = len(np.intersect1d(left, right, assume_unique=False))
        jaccard.append(intersection / (2 * k - intersection))

    centroid_cosines = []
    for label in classes:
        mask = labels == label
        first_centroid = first[mask].mean(axis=0)
        second_centroid = second[mask].mean(axis=0)
        denominator = np.linalg.norm(first_centroid) * np.linalg.norm(second_centroid)
        if denominator > 0:
            centroid_cosines.append(float(first_centroid @ second_centroid / denominator))

    return {
        "cross_modal_label_transfer_balanced_accuracy": float(
            np.mean(
                [
                    balanced_accuracy_score(labels, predictions_forward),
                    balanced_accuracy_score(labels, predictions_reverse),
                ]
            )
        ),
        "paired_neighborhood_jaccard": float(np.mean(jaccard)),
        "cell_type_centroid_cosine": float(np.mean(centroid_cosines)),
    }


def modality_metrics(
    rna,
    modality2,
    modality2_name,
    labels,
    seed,
    neighbors,
    silhouette_cells,
):
    joined = StandardScaler().fit_transform(np.vstack([rna, modality2]))
    modality = np.asarray(["RNA"] * len(rna) + [modality2_name] * len(modality2))
    indices = neighbor_indices(joined, neighbors)
    inverse_simpson = []
    for values in modality[indices]:
        _, counts = np.unique(values, return_counts=True)
        probabilities = counts / counts.sum()
        inverse_simpson.append(1.0 / np.sum(probabilities**2))
    rng = np.random.default_rng(seed)
    sample = rng.choice(
        len(joined),
        size=min(silhouette_cells, len(joined)),
        replace=False,
    )
    modality_asw = silhouette_score(joined[sample], modality[sample])
    return {
        "modality_iLISI": float(np.mean(np.asarray(inverse_simpson) - 1.0)),
        "modality_ASW": float(1.0 - abs(modality_asw)),
        **foscttm_and_retrieval(rna, modality2),
        **cross_modal_biology_metrics(rna, modality2, labels),
    }


def load_modality_embeddings(payload, path):
    if "rna" not in payload.files:
        return None
    if "modality2" in payload.files:
        modality2 = payload["modality2"]
    elif "atac" in payload.files:
        modality2 = payload["atac"]
    elif "adt" in payload.files:
        modality2 = payload["adt"]
    else:
        return None
    if "modality2_name" in payload.files:
        name = str(np.asarray(payload["modality2_name"]).item())
    elif "adt" in payload.files:
        name = "ADT"
    else:
        name = "ATAC"
    rna = np.asarray(payload["rna"], dtype=np.float64)
    modality2 = np.asarray(modality2, dtype=np.float64)
    if rna.shape != modality2.shape:
        raise ValueError(f"Modality-specific embedding shapes differ in {path}")
    return rna, modality2, name


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = ad.read_h5ad(args.reference_rna, backed="r")
    if args.label_column not in reference.obs:
        raise KeyError(f"Missing label column: {args.label_column}")
    if args.batch_column not in reference.obs:
        raise KeyError(f"Missing batch column: {args.batch_column}")
    reference_ids = np.asarray(reference.obs_names, dtype=str)
    labels_by_id = dict(
        zip(reference_ids, np.asarray(reference.obs[args.label_column], dtype=str))
    )
    batches_by_id = dict(
        zip(reference_ids, np.asarray(reference.obs[args.batch_column], dtype=str))
    )
    reference.file.close()

    rows = []
    structure_rows = []
    for method, seed_text, path_text in args.embedding:
        path = Path(path_text)
        payload = np.load(path)
        cell_ids = np.asarray(payload["cell_id"], dtype=str)
        if len(np.unique(cell_ids)) != len(cell_ids):
            raise ValueError(f"Duplicate cell IDs in {path}")
        missing = sorted(set(cell_ids) - set(labels_by_id))
        if missing:
            raise ValueError(f"{len(missing)} embedding cells are absent from {args.reference_rna}")
        labels = np.asarray([labels_by_id[cell_id] for cell_id in cell_ids], dtype=str)
        batches = np.asarray([batches_by_id[cell_id] for cell_id in cell_ids], dtype=str)
        integrated = np.asarray(payload["integrated"], dtype=np.float64)
        if integrated.shape[0] != len(cell_ids):
            raise ValueError(f"Integrated embedding row count differs in {path}")
        metrics = biological_metrics(
            integrated,
            labels,
            args.evaluation_seed,
            args.neighbors,
            args.silhouette_cells,
            args.isolated_max_fraction,
        )
        structure_rows.extend(
            cell_type_structure_rows(
                integrated,
                labels,
                method,
                seed_text,
                args.dataset_name,
                args.neighbors,
            )
        )
        metrics.update(
            batch_metrics(
                integrated,
                batches,
                args.evaluation_seed,
                args.neighbors,
                args.silhouette_cells,
            )
        )
        metrics.update(
            batch_metrics_within_labels(
                integrated,
                labels,
                batches,
                args.evaluation_seed,
                args.neighbors,
                args.silhouette_cells,
            )
        )
        modality_payload = load_modality_embeddings(payload, path)
        if modality_payload is not None:
            metrics.update(
                modality_metrics(
                    *modality_payload,
                    labels,
                    args.evaluation_seed,
                    args.neighbors,
                    args.silhouette_cells,
                )
            )
        rows.append(
            {
                "dataset": args.dataset_name,
                "method": method,
                "seed": int(seed_text),
                "n_cells": len(cell_ids),
                "latent_dim": integrated.shape[1],
                "embedding_path": str(path.resolve()),
                **metrics,
            }
        )
        print(method, seed_text, json.dumps(metrics, sort_keys=True), flush=True)

    result = pd.DataFrame(rows).sort_values(["method", "seed"])
    result.to_csv(output_dir / "整合指标_按随机种子.csv", index=False)
    pd.DataFrame(structure_rows).to_csv(
        output_dir / "结构指标_按细胞类型.csv",
        index=False,
    )
    numeric = [
        column
        for column in result.columns
        if column not in {"dataset", "method", "seed", "embedding_path"}
    ]
    summary = result.groupby("method")[numeric].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "n_seeds", result.groupby("method").size())
    summary.reset_index().to_csv(output_dir / "整合指标汇总.csv", index=False)
    evaluation_config = {
        **vars(args),
        "metric_notes": {
            "isolated_label_F1": (
                "Mean best one-vs-cluster F1 for labels no larger than "
                "isolated_max_fraction."
            ),
            "kBET_acceptance": (
                "All-cell local chi-square acceptance rate against global batch "
                "proportions; scIB/R kBET is not installed."
            ),
            "batch_metrics_within_cell_type": (
                "Cell-count-weighted batch mixing computed separately within each "
                "cell type represented in at least two batches."
            ),
            "cross_modal_label_transfer_balanced_accuracy": (
                "Mean bidirectional balanced accuracy from 15 cross-modal nearest "
                "neighbors after excluding each cell's exact paired counterpart."
            ),
            "paired_neighborhood_jaccard": (
                "Mean Jaccard overlap between corresponding cells' 15-neighbor "
                "within-modality neighborhoods."
            ),
            "cell_type_centroid_cosine": (
                "Mean cosine similarity between matched RNA and modality-2 cell-type "
                "centroids after joint standardization."
            ),
        },
    }
    with (output_dir / "评价配置.json").open("w", encoding="utf-8") as handle:
        json.dump(evaluation_config, handle, ensure_ascii=False, indent=2)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
