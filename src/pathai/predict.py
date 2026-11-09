"""Run the selected input mode on a patient with its validation-selected checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import read_rows, tokens
from .model import Multimodal


def main():
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
