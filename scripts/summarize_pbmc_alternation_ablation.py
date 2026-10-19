"""Compare Experiment-3 alternating and shared-only missing-modality results."""

import argparse
import json
from pathlib import Path

import pandas as pd


PATHS = (("RNA", "ATAC→RNA"), ("ATAC", "RNA→ATAC"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alternating-dir",
        default=(
            "results/PBMC_实验3_缺失模态预测_"
            "单次z采样_修正版/随机种子_0"
        ),
    )
    parser.add_argument(
        "--shared-only-dir",
        default=(
            "results/PBMC_实验3_去除交替训练_"
            "缺失模态预测/随机种子_0"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/PBMC_实验3_交替训练消融比较",
    )
    return parser.parse_args()


def load_result(directory):
    directory = Path(directory)
    metrics = pd.read_csv(directory / "缺失模态预测_总体指标.csv").set_index(
        ["模态", "预测路径"]
    )
    structure = pd.read_csv(directory / "特征相关结构保留.csv").set_index(
        ["模态", "预测路径"]
    )
    classification = pd.read_csv(directory / "补全数据下游分类.csv").set_index(
        ["模态", "输入"]
    )
    with (directory / "试点汇总.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    return metrics, structure, classification, summary


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    alternating = load_result(args.alternating_dir)
    shared_only = load_result(args.shared_only_dir)
    rows = []
    for modality, path in PATHS:
        for metric, source, higher_is_better in (
            ("标准化均方误差", 0, False),
            ("逐细胞Spearman", 0, True),
            ("逐特征Spearman", 0, True),
            ("特征相关结构Spearman", 1, True),
            ("补全数据平衡分类准确率", 2, True),
            ("差异效应合并Spearman", 3, True),
        ):
            if source == 0:
                column = {
                    "标准化均方误差": "对数尺度标准化均方误差",
                    "逐细胞Spearman": "逐细胞Spearman均值",
                    "逐特征Spearman": "逐特征Spearman均值",
                }[metric]
                full_value = float(alternating[0].loc[(modality, path), column])
                ablation_value = float(shared_only[0].loc[(modality, path), column])
            elif source == 1:
                full_value = float(
                    alternating[1].loc[(modality, path), "特征相关结构Spearman"]
                )
                ablation_value = float(
                    shared_only[1].loc[(modality, path), "特征相关结构Spearman"]
                )
            elif source == 2:
                full_value = float(alternating[2].loc[(modality, path), "平衡准确率"])
                ablation_value = float(shared_only[2].loc[(modality, path), "平衡准确率"])
            else:
                key = f"{modality}|{path}"
                full_value = float(alternating[3]["差异效应合并Spearman"][key])
                ablation_value = float(shared_only[3]["差异效应合并Spearman"][key])
            advantage = (
                full_value - ablation_value
                if higher_is_better
                else ablation_value - full_value
            )
            rows.append(
                {
                    "模态": modality,
                    "跨模态路径": path,
                    "指标": metric,
                    "完整交替训练": full_value,
                    "仅共享路径训练": ablation_value,
                    "完整模型优势": advantage,
                    "越高越好": higher_is_better,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "交替训练消融指标比较.csv", index=False)
    summary = {
        "比较指标数": len(frame),
        "完整模型占优指标数": int((frame["完整模型优势"] > 0).sum()),
        "仅共享路径占优指标数": int((frame["完整模型优势"] < 0).sum()),
        "方向持平指标数": int((frame["完整模型优势"].abs() <= 1e-12).sum()),
        "解释": (
            "只有完整模型在多数跨模态指标上优于仅共享路径训练，"
            "才支持交替解码输入是跨模态互操作性的来源。"
        ),
    }
    with (output / "消融比较汇总.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
