#!/usr/bin/env python
"""Validate and summarize a formal held-out probability benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("fixed", "global", "feature", "continuous_prior", "state_resolved")
METHOD_OUTPUT_NAMES = {
    "fixed": "固定权重",
    "global": "全局权重",
    "feature": "特征级权重",
    "continuous_prior": "连续先验",
    "state_resolved": "状态分辨模型",
}
DATASET_MODALITIES = {"GLUE_PBMC10k": "ATAC", "SLN111": "ADT"}
METRICS = (
    "nll_per_entry",
    "zero_rate_mae",
    "mean_mae",
    "variance_mae",
    "coverage_90",
    "interval_width_90",
)
EXPECTED_CONFIG = {
    "epochs": 200,
    "warmup_epochs": 50,
    "latent_dim": 32,
    "batch_size": 64,
    "coverage_draws": 20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_root",
        type=Path,
        default=Path("results/PBMC_留出概率基准_正式版"),
    )
    parser.add_argument("--dataset", default="GLUE_PBMC10k")
    parser.add_argument("--modality2", default="")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--allow_incomplete", action="store_true")
    args = parser.parse_args()
    if not args.modality2:
        try:
            args.modality2 = DATASET_MODALITIES[args.dataset]
        except KeyError as error:
            raise SystemExit("--modality2 is required for an unknown dataset") from error
    return args


def read_formal_runs(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    records: list[dict] = []
    errors: list[str] = []
    dataset_root = args.result_root / args.dataset

    for method in METHODS:
        for seed in args.seeds:
            preferred_dir = (
                dataset_root / METHOD_OUTPUT_NAMES[method] / f"随机种子_{seed}"
            )
            legacy_dir = dataset_root / method / f"seed_{seed}"
            run_dir = preferred_dir if preferred_dir.is_dir() else legacy_dir
            status_path = first_existing(
                run_dir / "终端状态.txt",
                run_dir / "terminal_status.txt",
            )
            config_path = first_existing(
                run_dir / "运行配置.json",
                run_dir / "run_config.json",
            )
            metrics_path = first_existing(
                run_dir / "测试指标.csv",
                run_dir / "test_metrics.csv",
            )
            history_path = first_existing(
                run_dir / "训练历史.csv",
                run_dir / "training_history.csv",
            )
            required = (status_path, config_path, metrics_path, history_path)
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                errors.append(f"{method}/随机种子_{seed}: missing {', '.join(missing)}")
                continue

            status = status_path.read_text(encoding="utf-8-sig").strip()
            if status != "success":
                errors.append(f"{method}/随机种子_{seed}: status={status!r}")
                continue

            config = json.loads(config_path.read_text(encoding="utf-8"))
            for key, expected in EXPECTED_CONFIG.items():
                if config.get(key) != expected:
                    errors.append(
                        f"{method}/随机种子_{seed}: config {key}={config.get(key)!r}, "
                        f"expected {expected!r}"
                    )
            if config.get("method") != method or config.get("seed") != seed:
                errors.append(f"{method}/随机种子_{seed}: method or seed mismatch in 运行配置.json")

            metrics = pd.read_csv(metrics_path)
            history = pd.read_csv(history_path)
            expected_modalities = {"RNA", args.modality2}
            if set(metrics["modality"]) != expected_modalities:
                errors.append(
                    f"{method}/随机种子_{seed}: expected metric rows {sorted(expected_modalities)}"
                )
                continue
            if len(history) != EXPECTED_CONFIG["epochs"]:
                errors.append(
                    f"{method}/随机种子_{seed}: history has {len(history)} epochs, "
                    f"expected {EXPECTED_CONFIG['epochs']}"
                )
                continue

            final_loss = float(history.iloc[-1]["loss"])
            runtime_minutes = float(history.iloc[-1]["elapsed_seconds"]) / 60.0
            for row in metrics.to_dict("records"):
                records.append(
                    {
                        **row,
                        "final_loss": final_loss,
                        "runtime_minutes": runtime_minutes,
                        "run_dir": str(run_dir),
                    }
                )

    return pd.DataFrame.from_records(records), errors


def first_existing(preferred: Path, legacy: Path) -> Path:
    return preferred if preferred.exists() else legacy


def aggregate(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = detail.groupby(["method", "modality"], sort=False)
    rows = []
    for (method, modality), frame in grouped:
        row = {
            "method": method,
            "modality": modality,
            "n_seeds": frame["seed"].nunique(),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = frame[metric].mean()
            row[f"{metric}_std"] = frame[metric].std(ddof=1)
        row["runtime_minutes_mean"] = frame["runtime_minutes"].mean()
        row["runtime_minutes_std"] = frame["runtime_minutes"].std(ddof=1)
        row["final_loss_mean"] = frame["final_loss"].mean()
        row["final_loss_std"] = frame["final_loss"].std(ddof=1)
        rows.append(row)

    joint = detail[["method", "seed", "joint_nll", "runtime_minutes", "final_loss"]].drop_duplicates()
    joint_rows = []
    for method, frame in joint.groupby("method", sort=False):
        joint_rows.append(
            {
                "method": method,
                "n_seeds": frame["seed"].nunique(),
                "joint_nll_mean": frame["joint_nll"].mean(),
                "joint_nll_std": frame["joint_nll"].std(ddof=1),
                "runtime_minutes_mean": frame["runtime_minutes"].mean(),
                "runtime_minutes_std": frame["runtime_minutes"].std(ddof=1),
                "final_loss_mean": frame["final_loss"].mean(),
                "final_loss_std": frame["final_loss"].std(ddof=1),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(joint_rows)


def paired_differences(detail: pd.DataFrame) -> pd.DataFrame:
    joint = detail[["method", "seed", "joint_nll"]].drop_duplicates()
    wide = joint.pivot(index="seed", columns="method", values="joint_nll")
    rows = []

    comparisons = [("fixed", method) for method in METHODS if method != "fixed"]
    comparisons.extend(
        (method, "state_resolved")
        for method in METHODS
        if method not in {"fixed", "state_resolved"}
    )
    for reference, candidate in comparisons:
        for seed in wide.index:
            rows.append(
                {
                    "reference": reference,
                    "candidate": candidate,
                    "seed": seed,
                    "joint_nll_improvement": wide.loc[seed, reference] - wide.loc[seed, candidate],
                }
            )
    return pd.DataFrame(rows)


def exact_two_sided_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return float("nan")
    extreme = max(wins, losses)
    upper_tail = sum(math.comb(n, value) for value in range(extreme, n + 1)) / (2**n)
    return min(1.0, 2.0 * upper_tail)


def summarize_paired(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (reference, candidate), frame in paired.groupby(["reference", "candidate"], sort=False):
        values = frame["joint_nll_improvement"].to_numpy()
        non_ties = values[np.abs(values) > 1e-12]
        wins = int((non_ties > 0).sum())
        losses = int((non_ties < 0).sum())
        rows.append(
            {
                "reference": reference,
                "candidate": candidate,
                "n_seeds": len(values),
                "mean_improvement": values.mean(),
                "std_improvement": values.std(ddof=1),
                "median_improvement": np.median(values),
                "wins": wins,
                "losses": losses,
                "ties": len(values) - len(non_ties),
                "two_sided_exact_sign_p": exact_two_sided_sign_p(wins, losses),
            }
        )
    return pd.DataFrame(rows)


def mean_std(mean: float, std: float, digits: int = 6) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def write_report(
    path: Path,
    dataset: str,
    modality2: str,
    seeds: list[int],
    modality_summary: pd.DataFrame,
    joint_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
) -> None:
    ranked = joint_summary.sort_values("joint_nll_mean")
    best = ranked.iloc[0]
    state = joint_summary[joint_summary["method"] == "state_resolved"].iloc[0]
    state_vs = paired_summary[paired_summary["candidate"] == "state_resolved"]

    lines = [
        f"# {dataset} 留出数据概率拟合与校准实验报告",
        "",
        f"- 数据集：`{dataset}`（RNA+{modality2}）",
        f"- 正式随机种子：{', '.join(map(str, seeds))}",
        "- 固定配置：200 轮、50 轮预热、32 维潜变量、批大小 64",
        "- 数据划分：按细胞类型分层的训练/验证/测试划分为 70%/15%/15%；标签仅用于划分，不参与模型训练",
        "- 测试解码：使用确定性的共享后验均值；90% 后验预测区间额外采样潜变量与观测计数",
        "",
        "## 联合负对数似然",
        "",
        "| 方法 | 联合负对数似然（均值 ± 标准差） | 训练时间/分钟（均值 ± 标准差） |",
        "| --- | ---: | ---: |",
    ]
    for row in ranked.itertuples(index=False):
        lines.append(
            f"| `{row.method}` | {mean_std(row.joint_nll_mean, row.joint_nll_std)} "
            f"| {mean_std(row.runtime_minutes_mean, row.runtime_minutes_std, 2)} |"
        )

    lines.extend(
        [
            "",
            "## 分模态结果",
            "",
            "| 方法 | 模态 | 负对数似然 | 零比例误差 | 均值误差 | 方差误差 | 90% 覆盖率 | 区间宽度 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in modality_summary.itertuples(index=False):
        lines.append(
            f"| `{row.method}` | {row.modality} "
            f"| {mean_std(row.nll_per_entry_mean, row.nll_per_entry_std)} "
            f"| {mean_std(row.zero_rate_mae_mean, row.zero_rate_mae_std)} "
            f"| {mean_std(row.mean_mae_mean, row.mean_mae_std)} "
            f"| {mean_std(row.variance_mae_mean, row.variance_mae_std)} "
            f"| {mean_std(row.coverage_90_mean, row.coverage_90_std)} "
            f"| {mean_std(row.interval_width_90_mean, row.interval_width_90_std)} |"
        )

    lines.extend(
        [
            "",
            "## 成对比较",
            "",
            "下表的正值表示状态分辨模型具有更低的联合负对数似然。",
            "",
            "| 参照方法 | 联合负对数似然改善（均值 ± 标准差） | 胜/负 | 双侧精确符号检验 P 值 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in state_vs.itertuples(index=False):
        lines.append(
            f"| `{row.reference}` | {mean_std(row.mean_improvement, row.std_improvement)} "
            f"| {row.wins}/{row.losses} | {row.two_sided_exact_sign_p:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            f"- 当前联合负对数似然最低的方法是 `{best.method}`："
            f"{mean_std(best.joint_nll_mean, best.joint_nll_std)}。",
            f"- `state_resolved` 的联合负对数似然为 "
            f"{mean_std(state.joint_nll_mean, state.joint_nll_std)}；"
            "是否支持状态残差改善概率拟合，应结合上述成对差值及各模态校准指标判断。",
            "- 覆盖率高于 0.90 不等于越高越好；明显高于目标值通常表示预测区间偏宽或离散计数导致的保守覆盖。",
            "- 五个随机种子的精确符号检验统计功效有限；本实验应报告效应量、方向一致性和原始种子点，"
            "不把未达到显著性的微小均值差异表述为确定优势。",
            f"- 该报告只总结 `{dataset}`；跨数据集结论必须与其他数据集的独立结果联合判断。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    detail, errors = read_formal_runs(args)
    if errors:
        print("Formal-run validation issues:")
        for error in errors:
            print(f"- {error}")
        if not args.allow_incomplete:
            raise SystemExit(1)
    if detail.empty:
        raise SystemExit("No valid formal runs found.")

    expected_rows = len(METHODS) * len(args.seeds) * 2
    if not args.allow_incomplete and len(detail) != expected_rows:
        raise SystemExit(f"Expected {expected_rows} metric rows, found {len(detail)}.")

    modality_summary, joint_summary = aggregate(detail)
    paired = paired_differences(detail)
    paired_summary = summarize_paired(paired)
    output = args.result_root / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    detail.sort_values(["method", "seed", "modality"]).to_csv(
        output / "全部随机种子指标.csv", index=False
    )
    modality_summary.to_csv(output / "方法模态指标汇总.csv", index=False)
    joint_summary.to_csv(output / "方法联合指标汇总.csv", index=False)
    paired.to_csv(output / "配对联合负对数似然差异.csv", index=False)
    paired_summary.to_csv(output / "配对联合负对数似然汇总.csv", index=False)
    write_report(
        output / "留出概率基准报告.md",
        args.dataset,
        args.modality2,
        args.seeds,
        modality_summary,
        joint_summary,
        paired_summary,
    )
    print(f"Wrote validated summaries to {output.resolve()}")


if __name__ == "__main__":
    main()
