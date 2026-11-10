"""Aggregate patient predictions and plot the single fixed validation split."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .common import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="artifacts/run")
    parser.add_argument(
        "--out",
        type=Path,
        help="Directory for metrics and figures; defaults to the run directory",
    )
    args = parser.parse_args()
    root = Path(args.run)
    out = args.out or root
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((root / "predictions.csv").open()))
    names = {"events": "Events", "image": "Image", "both": "Image + events"}
    result = {}
    for mode in ["events", "image", "both"]:
        selected = [r for r in rows if r["mode"] == mode]
        y = np.array([int(r["label"]) for r in selected])
        p = np.array([float(r["probability"]) for r in selected])
        result[mode] = {
            "accuracy": float(accuracy_score(y, p >= 0.5)),
            "bce": float(log_loss(y, p, labels=[0, 1])),
            "auroc": float(roc_auc_score(y, p)),
        }
    write_json(out / "metrics.json", result)
    history = json.loads((root / "history.json").read_text())
    budget = json.loads((root / "config.json").read_text())["steps"]
    pre = json.loads((root / "pretraining.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for field in ["train_lm", "val_lm"]:
        axes[0, 0].plot([x["step"] for x in pre], [x[field] for x in pre], label=field)
    axes[0, 0].set(
        title="Original GPT: Mayo pretraining",
        xlabel="Step",
        ylabel="Next-event cross entropy",
    )
    for mode in ["events", "image", "both"]:
        record = next(x for x in history if x["mode"] == mode)
        trace = record["trace"]
        for ax, field, title in [
            (axes[0, 1], "train_bce", "TCGA training"),
            (axes[1, 0], "eval_bce", "TCGA validation"),
            (axes[1, 1], "report_loss", "Training report prediction"),
        ]:
            (line,) = ax.plot(
                [x["step"] for x in trace],
                [x[field] for x in trace],
                label=f"{names[mode]} (end: {trace[-1]['step']})",
                marker=".",
                markersize=3,
            )
            ax.scatter(
                trace[-1]["step"],
                trace[-1][field],
                marker="x",
                s=45,
                color=line.get_color(),
                zorder=4,
            )
            ax.set_xlim(-2, budget + 3)
            if field == "eval_bce":
                selected = next(
                    x for x in trace if x["step"] == record["selected_step"]
                )
                ax.scatter(
                    selected["step"], selected[field], color=line.get_color(), zorder=3
                )
            ax.set(
                title=title,
                xlabel="Joint training step",
                ylabel={
                    "train_bce": "Binary cross-entropy",
                    "eval_bce": "Binary cross-entropy",
                    "report_loss": "Cosine distance",
                }[field],
            )
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Training curves")
    fig.text(
        0.5,
        0.015,
        "Dots: logged values | Filled circle: selected checkpoint | x: training end (early stopping or step limit)",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out / "learning_curves.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, metric in zip(axes, ["accuracy", "bce", "auroc"]):
        vals = [result[m][metric] for m in result]
        ax.bar(
            [names[m] for m in result], vals, color=["#697080", "#c28e52", "#2c8b94"]
        )
        ax.set(
            title={"accuracy": "Accuracy", "bce": "BCE", "auroc": "AUROC"}[metric],
            ylim=(0, max(1, max(vals) * 1.15)),
        )
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    fig.suptitle("6 train / 4 validation patients")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=170)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
