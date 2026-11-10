"""Controlled loss ablations / 固定其他配置的损失消融。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import write_json

VARIANTS = {
    "full": ("Full objective", 1.0, 0.1, 0.1),
    "no_report": ("No report loss", 1.0, 0.1, 0.0),
    "no_event": ("No event loss", 1.0, 0.0, 0.1),
    "no_outcome": ("No outcome loss", 0.0, 0.1, 0.1),
    "outcome_only": ("Outcome only", 1.0, 0.0, 0.0),
}
COLORS = ["#248d80", "#cb8c53", "#8270b5", "#bd6585", "#758595"]


def plot_results(records, out):
    """Plot paired loss-ablation results / 绘制两组损失消融结果。

    Input / 输入:
        records: run→variant metrics and histories; out: figure directory.
        records：按实验和方案组织的指标与日志；out：图像目录。
    Processing / 处理:
        Compare accuracy/BCE and plot validation traces with selected and final steps.
        比较准确率和 BCE，绘制验证轨迹并标记最佳与终止位置。
    Output / 输出:
        comparison.png and learning_curves.png; returns None.
        保存指标对比图与曲线，无返回值。
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    curves, curve_axes = plt.subplots(1, 2, figsize=(12, 4))
    for column, (run, variants) in enumerate(records.items()):
        labels = [value[0].replace(" ", "\n") for value in VARIANTS.values()]
        for row, metric in enumerate(["accuracy", "bce"]):
            ax = axes[row, column]
            values = [record[metric] for record in variants.values()]
            ax.bar(np.arange(len(values)), values, color=COLORS)
            ax.set_xticks(np.arange(len(values)), labels, fontsize=8)
            ax.set_title(
                f"Run {run.upper()} · {'Accuracy' if metric == 'accuracy' else 'BCE'}"
            )
            ax.set_ylim(0, max(1.0, max(values) * 1.18))
            for i, value in enumerate(values):
                ax.text(i, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)
        ax = curve_axes[column]
        for (key, record), color in zip(variants.items(), COLORS):
            trace = record["trace"]
            ax.plot(
                [r["step"] for r in trace],
                [r["eval_bce"] for r in trace],
                color=color,
                label=f"{VARIANTS[key][0]} (end: {trace[-1]['step']})",
                marker=".",
                markersize=3,
            )
            ax.scatter(
                trace[-1]["step"],
                trace[-1]["eval_bce"],
                marker="x",
                s=45,
                color=color,
                zorder=4,
            )
            selected = next(r for r in trace if r["step"] == record["selected_step"])
            ax.scatter(selected["step"], selected["eval_bce"], color=color, s=25)
        ax.set(
            title=f"Run {run.upper()}",
            xlabel="Fine-tuning step",
            ylabel="Validation BCE",
            xlim=(-2, 103),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Stage 2 loss ablation · image + events")
    curves.suptitle("Loss ablation learning curves")
    curves.text(
        0.5,
        0.015,
        "Dots: logged values | Filled circle: selected checkpoint | x: training end (early stopping or step limit)",
        ha="center",
        fontsize=8,
    )
    for figure, filename in [(fig, "comparison.png"), (curves, "learning_curves.png")]:
        figure.tight_layout(rect=(0, 0.055 if figure is curves else 0, 1, 1))
        figure.savefig(out / filename, dpi=170, bbox_inches="tight")
        plt.close(figure)


def main():
    """Run controlled Stage 2 loss ablations / 执行 Stage 2 损失消融。

    Input / 输入:
        CLI --pretrained-run, --out, --results; cached patient features.
        预训练目录、训练输出目录、结果目录及患者特征缓存。
    Processing / 处理:
        Train five loss variants with image+events for each run; verify other settings and splits match.
        每组联合输入训练五种损失配置，核对其他设置与患者划分一致。
    Output / 输出:
        Per-variant checkpoints/logs and aggregate metrics/figures; returns None.
        输出各方案权重、日志及汇总指标、图像，无返回值。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/ablation"))
    parser.add_argument("--results", type=Path, default=Path("results/ablation"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    args.results.mkdir(parents=True, exist_ok=True)
    records = {}
    for experiment in ["a", "b"]:
        records[experiment] = {}
        reference = None
        for variant, (label, outcome, event, report) in VARIANTS.items():
            run = args.out / f"run_{experiment}" / variant
            run.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "pathai.train",
                "--stage",
                "finetune",
                "--experiment",
                experiment,
                "--modes",
                "both",
                "--pretrained-run",
                str(args.pretrained_run),
                "--out",
                str(run),
                "--outcome-weight",
                str(outcome),
                "--lm-weight",
                str(event),
                "--report-weight",
                str(report),
            ]
            print(f"Run {experiment.upper()} / {label}", flush=True)
            with (run.parent / f"{variant}.log").open("w") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            config = json.loads((run / "config.json").read_text())
            split = json.loads((run / "split.json").read_text())
            invariants = {
                k: v
                for k, v in config.items()
                if k not in {"out", "outcome_weight", "lm_weight", "report_weight"}
            }
            if reference is None:
                reference = (invariants, split)
            elif reference != (invariants, split):
                raise ValueError(
                    "Ablation runs differ beyond the selected loss weights"
                )
            metric = json.loads((run / "metrics.json").read_text())["both"]
            history = json.loads((run / "history.json").read_text())[0]
            records[experiment][variant] = {
                **metric,
                "weights": {"outcome": outcome, "event": event, "report": report},
                "selected_step": history["selected_step"],
                "trace": history["trace"],
            }
            print(metric, flush=True)
    write_json(args.results / "metrics.json", records)
    plot_results(records, args.results)


if __name__ == "__main__":
    main()
