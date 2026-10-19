"""Analyze label-free CD14 monocyte states with RNA and promoter-proximal ATAC."""

import argparse
import json
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


PROGRAMS = {
    "即时早期反应": ("JUN", "FOSB"),
    "炎症型单核细胞": ("S100A8", "S100A9", "LYZ"),
    "抗原呈递": ("HLA-DRA", "HLA-DPA1", "CD74"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rna-h5ad",
        default="dataset/scLAMP_SR_ready/GLUE_PBMC10k/rna.h5ad",
    )
    parser.add_argument(
        "--atac-h5ad",
        default="dataset/scLAMP_SR_ready/GLUE_PBMC10k/atac.h5ad",
    )
    parser.add_argument(
        "--sclamp-embedding",
        default=(
            "results/scLAMP-SR_效率与结构/"
            "CleanE200_W50_CE002_KSoft090_R30_Start100/seed_0/embeddings.npz"
        ),
    )
    parser.add_argument(
        "--multivi-embedding",
        default=(
            "results/PBMC_多组学整合基准_试验/"
            "MultiVI/seed_0/embeddings.npz"
        ),
    )
    parser.add_argument(
        "--state-assignments",
        default=(
            "results/scLAMP-SR_效率与结构/"
            "CleanE200_W50_CE002_KSoft090_R30_Start100/seed_0/"
            "pseudo_state_assignments.csv"
        ),
    )
    parser.add_argument(
        "--state-lambda-contrast",
        default=(
            "results/scLAMP-SR_效率与结构/"
            "CleanE200_W50_CE002_KSoft090_R30_Start100/seed_0/"
            "state_lambda_contrast_by_gene.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/PBMC_CD14多模态状态生物学验证",
    )
    parser.add_argument("--promoter-window", type=int, default=50000)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--minimum-state-cells", type=int, default=50)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def dense_vector(matrix):
    return np.asarray(matrix).ravel()


def zscore_columns(values):
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return np.divide(
        values - mean,
        std,
        out=np.zeros_like(values, dtype=np.float64),
        where=std > 0,
    )


def rna_program_scores(rna, cell_mask):
    genes = [gene for markers in PROGRAMS.values() for gene in markers]
    names = rna.var["feature_name"].astype(str).to_numpy()
    positions = {gene: int(np.flatnonzero(names == gene)[0]) for gene in genes}
    counts = sp.csr_matrix(rna.X[cell_mask][:, list(positions.values())], dtype=np.float64)
    totals = dense_vector(rna.X[cell_mask].sum(axis=1)).astype(np.float64)
    normalized = counts.multiply(
        np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)[:, None]
    ).toarray()
    normalized = np.log1p(normalized)
    standardized = zscore_columns(normalized)
    gene_to_column = {gene: index for index, gene in enumerate(positions)}
    result = {}
    for program, markers in PROGRAMS.items():
        result[program] = standardized[
            :, [gene_to_column[gene] for gene in markers]
        ].mean(axis=1)
    return pd.DataFrame(result), positions


def promoter_links(rna, atac, promoter_window):
    genes = [gene for markers in PROGRAMS.values() for gene in markers]
    rna_names = rna.var["feature_name"].astype(str).to_numpy()
    peak_chrom = atac.var["chrom"].astype(str).to_numpy()
    peak_centers = (
        atac.var["chromStart"].to_numpy(dtype=np.int64)
        + atac.var["chromEnd"].to_numpy(dtype=np.int64)
    ) // 2
    links = []
    by_gene = {}
    for gene in genes:
        row = rna.var.iloc[int(np.flatnonzero(rna_names == gene)[0])]
        tss = int(row["chromEnd"] if str(row["strand"]) == "-" else row["chromStart"])
        same_chromosome = np.flatnonzero(peak_chrom == str(row["chrom"]))
        distance = np.abs(peak_centers[same_chromosome] - tss)
        selected = same_chromosome[distance <= promoter_window]
        if len(selected) == 0:
            selected = same_chromosome[np.argsort(distance)[:3]]
        by_gene[gene] = selected
        for peak_index in selected:
            links.append(
                {
                    "基因": gene,
                    "峰": str(atac.var_names[peak_index]),
                    "染色体": str(row["chrom"]),
                    "基因TSS": tss,
                    "峰中心": int(peak_centers[peak_index]),
                    "距TSS绝对距离": int(abs(peak_centers[peak_index] - tss)),
                }
            )
    return by_gene, pd.DataFrame(links)


def atac_program_scores(atac, cell_mask, links):
    tfidf = TfidfTransformer(
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    ).fit_transform(sp.csr_matrix(atac.X, dtype=np.float64))
    gene_scores = {}
    for gene, peak_indices in links.items():
        gene_scores[gene] = dense_vector(
            tfidf[cell_mask][:, peak_indices].mean(axis=1)
        )
    gene_frame = pd.DataFrame(gene_scores)
    standardized = pd.DataFrame(
        zscore_columns(gene_frame.to_numpy()),
        columns=gene_frame.columns,
    )
    result = {}
    for program, markers in PROGRAMS.items():
        result[program] = standardized.loc[:, list(markers)].mean(axis=1)
    return pd.DataFrame(result)


def load_embedding(path, expected_ids):
    with np.load(path) as payload:
        ids = np.asarray(payload["cell_id"], dtype=str)
        integrated = np.asarray(payload["integrated"], dtype=np.float64)
    positions = {cell_id: index for index, cell_id in enumerate(ids)}
    return integrated[[positions[cell_id] for cell_id in expected_ids]]


def neighborhood_preservation(method, embedding, scores, neighbors):
    values = StandardScaler().fit_transform(embedding)
    k = min(neighbors + 1, len(values))
    indices = (
        NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=1)
        .fit(values)
        .kneighbors(return_distance=False)[:, 1:]
    )
    rows = []
    for layer in ("RNA", "ATAC"):
        for program in PROGRAMS:
            score = scores[f"{layer}_{program}"].to_numpy(dtype=np.float64)
            neighbor_mean = score[indices].mean(axis=1)
            rows.append(
                {
                    "方法": method,
                    "分子层": layer,
                    "程序": program,
                    "细胞数": len(score),
                    "近邻数": indices.shape[1],
                    "自身与近邻均值Spearman": float(spearmanr(score, neighbor_mean).statistic),
                    "近邻平均绝对差": float(np.mean(np.abs(score[:, None] - score[indices]))),
                }
            )
    return rows


def bootstrap_contrast(target, other, draws, rng):
    observed = float(np.mean(target) - np.mean(other))
    samples = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        samples[index] = (
            rng.choice(target, size=len(target), replace=True).mean()
            - rng.choice(other, size=len(other), replace=True).mean()
        )
    return observed, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def state_program_contrasts(scores, assignments, lambda_frame, args):
    rng = np.random.default_rng(args.seed)
    counts = assignments["pseudo_state"].value_counts()
    retained = counts[counts >= args.minimum_state_cells].index.to_numpy(dtype=int)
    rows = []
    target_states = {}
    for program, genes in PROGRAMS.items():
        gene_states = lambda_frame.loc[lambda_frame["gene"].isin(genes), "top_state"].astype(int)
        target_state = Counter(gene_states).most_common(1)[0][0]
        target_states[program] = target_state
        for layer in ("RNA", "ATAC"):
            values = scores[f"{layer}_{program}"].to_numpy(dtype=np.float64)
            target = values[assignments["pseudo_state"].to_numpy(dtype=int) == target_state]
            other = values[
                (assignments["pseudo_state"].to_numpy(dtype=int) != target_state)
                & assignments["pseudo_state"].isin(retained).to_numpy()
            ]
            contrast, lower, upper = bootstrap_contrast(
                target,
                other,
                args.bootstrap_draws,
                rng,
            )
            rows.append(
                {
                    "程序": program,
                    "分子层": layer,
                    "目标伪状态": target_state,
                    "目标状态细胞数": len(target),
                    "其他保留状态细胞数": len(other),
                    "目标减其他均值得分": contrast,
                    "自助法95%下限": lower,
                    "自助法95%上限": upper,
                }
            )
    return target_states, pd.DataFrame(rows), retained


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rna = ad.read_h5ad(args.rna_h5ad)
    atac = ad.read_h5ad(args.atac_h5ad)
    if not rna.obs_names.equals(atac.obs_names):
        raise ValueError("RNA and ATAC cell order differs.")
    cd14_mask = rna.obs["cell_type"].astype(str).eq("CD14 Mono").to_numpy()
    cell_ids = np.asarray(rna.obs_names[cd14_mask], dtype=str)

    rna_scores, _ = rna_program_scores(rna, cd14_mask)
    links, link_frame = promoter_links(rna, atac, args.promoter_window)
    atac_scores = atac_program_scores(atac, cd14_mask, links)
    scores = pd.DataFrame({"细胞": cell_ids})
    for program in PROGRAMS:
        scores[f"RNA_{program}"] = rna_scores[program].to_numpy()
        scores[f"ATAC_{program}"] = atac_scores[program].to_numpy()

    assignments = pd.read_csv(args.state_assignments, dtype={"cell": str})
    assignments = assignments.set_index("cell").loc[cell_ids].reset_index()
    lambda_frame = pd.read_csv(args.state_lambda_contrast)
    marker_genes = [gene for markers in PROGRAMS.values() for gene in markers]
    marker_lambda = lambda_frame.loc[lambda_frame["gene"].isin(marker_genes)].copy()
    target_states, contrast_frame, retained_states = state_program_contrasts(
        scores,
        assignments,
        marker_lambda,
        args,
    )

    neighborhood_rows = []
    for method, path in (
        ("scLAMP-SR", args.sclamp_embedding),
        ("MultiVI", args.multivi_embedding),
    ):
        embedding = load_embedding(path, cell_ids)
        neighborhood_rows.extend(
            neighborhood_preservation(method, embedding, scores, args.neighbors)
        )
    neighborhood = pd.DataFrame(neighborhood_rows)

    state_means = scores.copy()
    state_means["伪状态"] = assignments["pseudo_state"].to_numpy(dtype=int)
    state_means = (
        state_means.loc[state_means["伪状态"].isin(retained_states)]
        .groupby("伪状态", as_index=False)
        .agg({**{column: "mean" for column in scores.columns if column != "细胞"}, "细胞": "size"})
        .rename(columns={"细胞": "细胞数"})
    )

    cross_modal_rows = []
    for program in PROGRAMS:
        correlation = spearmanr(
            scores[f"RNA_{program}"],
            scores[f"ATAC_{program}"],
        ).statistic
        cross_modal_rows.append({"程序": program, "RNA_ATAC细胞内Spearman": float(correlation)})
    cross_modal = pd.DataFrame(cross_modal_rows)

    scores["伪状态"] = assignments["pseudo_state"].to_numpy(dtype=int)
    scores.to_csv(output / "CD14细胞多模态程序得分.csv", index=False)
    link_frame.to_csv(output / "标志基因邻近ATAC峰.csv", index=False)
    marker_lambda.to_csv(output / "标志基因状态混合权重.csv", index=False)
    contrast_frame.to_csv(output / "程序目标状态对比.csv", index=False)
    neighborhood.to_csv(output / "程序潜空间近邻保持.csv", index=False)
    state_means.to_csv(output / "伪状态多模态程序均值.csv", index=False)
    cross_modal.to_csv(output / "RNA_ATAC程序一致性.csv", index=False)

    summary = {
        "负二项参数化": "pytorch_mean_dispersion_v1",
        "scLAMP嵌入": str(Path(args.sclamp_embedding)),
        "伪状态分配": str(Path(args.state_assignments)),
        "状态混合权重": str(Path(args.state_lambda_contrast)),
        "CD14细胞数": len(cell_ids),
        "保留伪状态": [int(value) for value in retained_states],
        "程序目标伪状态": {key: int(value) for key, value in target_states.items()},
        "程序基因": {key: list(value) for key, value in PROGRAMS.items()},
        "启动子邻域窗口bp": args.promoter_window,
        "各基因邻近峰数": {gene: int(len(indices)) for gene, indices in links.items()},
        "RNA_ATAC细胞内Spearman": {
            row["程序"]: row["RNA_ATAC细胞内Spearman"]
            for row in cross_modal.to_dict("records")
        },
        "scLAMP近邻相关高于MultiVI": {
            "比较数": int(len(neighborhood) // 2),
            "占优数": int(
                sum(
                    neighborhood.loc[
                        (neighborhood["方法"] == "scLAMP-SR")
                        & (neighborhood["程序"] == program)
                        & (neighborhood["分子层"] == layer),
                        "自身与近邻均值Spearman",
                    ].iloc[0]
                    > neighborhood.loc[
                        (neighborhood["方法"] == "MultiVI")
                        & (neighborhood["程序"] == program)
                        & (neighborhood["分子层"] == layer),
                        "自身与近邻均值Spearman",
                    ].iloc[0]
                    for program in PROGRAMS
                    for layer in ("RNA", "ATAC")
                )
            ),
        },
        "生物学参照": [
            {
                "程序": "即时早期反应",
                "解释": (
                    "JUN/FOSB 可表示快速单核细胞激活，也可能受采血后处理或解离应激影响；"
                    "当前结果应称为即时早期激活/应激轴，而不是稳定细胞亚型。"
                ),
                "文献": [
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC8169801/",
                    "https://pubmed.ncbi.nlm.nih.gov/31623682/",
                ],
            },
            {
                "程序": "炎症型单核细胞",
                "解释": "S100A8/S100A9 是经典髓系炎症程序，并在 CD14 单核细胞内部呈显著异质性。",
                "文献": [
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC6073713/",
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10278311/",
                ],
            },
            {
                "程序": "抗原呈递",
                "解释": "HLA-DRA/HLA-DPA1/CD74 对应 MHC-II 抗原加工与呈递增强的单核细胞状态。",
                "文献": [
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7568339/",
                    "https://pmc.ncbi.nlm.nih.gov/articles/PMC9934388/",
                ],
            },
        ],
        "解释边界": [
            "单数据集、单模型随机种子的探索性生物学分析。",
            "自助法区间以细胞为统计单位，不代表跨供体生物重复。",
            "邻近TSS峰提供正交染色质支持，但不等同于已验证的增强子-基因因果链接。",
            "炎症程序的逐细胞 RNA-ATAC 相关较弱，当前证据主要支持状态层面的方向一致。",
        ],
    }
    with (output / "分析汇总.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(contrast_frame.to_string(index=False), flush=True)
    print(neighborhood.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
