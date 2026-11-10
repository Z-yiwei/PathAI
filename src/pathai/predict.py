"""Single-patient outcome inference / 单患者结局预测。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import read_rows, tokens
from .model import Multimodal


def main():
    """Predict one validation patient / 对一位验证患者执行预测。

    Input / 输入:
        CLI --run, --patient, --mode; checkpoint and prepared event/image features.
        命令行实验目录、患者 ID、模式；对应检查点及预处理事件/图像特征。
    Processing / 处理:
        Require validation membership, restore the matching model, encode inputs, and apply sigmoid.
        检查患者属于验证集，恢复对应模型，整理输入并用 sigmoid 得到概率。
    Output / 输出:
        Print JSON with patient ID, mode, probability, and class at threshold 0.5; returns None.
        打印患者 ID、模式、概率及 0.5 阈值分类的 JSON，无返回值。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", required=True)
    parser.add_argument("--mode", choices=["events", "image", "both"], default="both")
    parser.add_argument("--run", default="artifacts/run")
    args = parser.parse_args()
    root = Path(args.run)
    if args.patient not in json.loads((root / "split.json").read_text())["val"]:
        raise ValueError("Demo patient must belong to the validation set")
    ck = torch.load(root / f"{args.mode}.pt", map_location="cpu", weights_only=False)
    model = Multimodal(len(ck["vocab"]) + 1, {})
    model.load_state_dict(ck["model"])
    model.eval()
    t, im = None, None
    if args.mode in ["events", "both"]:
        row = next(
            r
            for r in read_rows("artifacts/processed/patients.jsonl")
            if r["patient_id"] == args.patient
        )
        t = tokens([row], ck["vocab"])
    if args.mode in ["image", "both"]:
        im = F.normalize(
            torch.tensor(
                np.load(f"artifacts/image_features/{args.patient}.npz")[
                    "features"
                ].mean(0)
            )[None],
            dim=1,
        )
    with torch.no_grad():
        p = float(torch.sigmoid(model(t, im, args.mode)[0])[0])
    print(
        json.dumps(
            {
                "patient_id": args.patient,
                "mode": args.mode,
                "probability_death_recorded": p,
                "prediction": int(p >= 0.5),
            }
        )
    )


if __name__ == "__main__":
    main()
