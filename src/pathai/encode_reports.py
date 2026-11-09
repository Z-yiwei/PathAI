"""Encode source-only reports without rewriting content or using outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np

from .common import write_json
from .report_quality import load_training_report


def contiguous_chunks(token_ids, size=75):
    return [token_ids[i : i + size] for i in range(0, len(token_ids), size)]


def main():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", default="artifacts/processed/patients.jsonl")
    parser.add_argument("--out", default="artifacts/report_features")
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    rows = sorted(
        [json.loads(s) for s in Path(args.patients).read_text().splitlines()],
        key=lambda r: r["patient_id"],
    )
    model = (
        CLIPModel.from_pretrained("artifacts/pretrained/plip", local_files_only=True)
        .eval()
        .requires_grad_(False)
        .cuda()
    )
    processor = CLIPProcessor.from_pretrained(
        "artifacts/pretrained/plip", local_files_only=True
    )
    records = []
    for r in rows:
        text = load_training_report(r) if r.get("report_trainable") else ""
        chunks = contiguous_chunks(
            processor.tokenizer(text, add_special_tokens=False)["input_ids"]
        )
        if chunks:
            with torch.inference_mode():
                sequences = [
                    [processor.tokenizer.bos_token_id]
                    + c
                    + [processor.tokenizer.eos_token_id]
                    for c in chunks
                ]
                inputs = processor.tokenizer.pad(
                    {
                        "input_ids": sequences,
                        "attention_mask": [[1] * len(c) for c in sequences],
                    },
                    padding=True,
                    return_tensors="pt",
                )
                z = model.get_text_features(**{k: v.cuda() for k, v in inputs.items()})
                z = torch.nn.functional.normalize(z, dim=-1)
                weights = torch.tensor(
                    [len(c) for c in chunks], device=z.device, dtype=z.dtype
                )
                z = (z * weights[:, None]).sum(0) / weights.sum()
                vector = torch.nn.functional.normalize(z, dim=0).cpu().numpy()
        else:
            vector = np.zeros(512, dtype=np.float32)
        np.savez_compressed(
            root / f"{r['patient_id']}.npz", features=vector, available=bool(chunks)
        )
        records.append(
            {
                "patient_id": r["patient_id"],
                "available": bool(chunks),
                "text_chunks": len(chunks),
                "pooling": "token_weighted_mean",
                "tokens_covered": sum(map(len, chunks)) if chunks else None,
            }
        )
    write_json(
        root / "encoding.json",
        {
            "pooling": "token_weighted_mean",
            "records": records,
            "patients_path": str(Path(args.patients).resolve()),
            "model": "vinid/plip",
            "dimension": 512,
            "frozen": True,
        },
    )
    print(
        "Encoded",
        len(records),
        "patients;",
        sum(r["available"] for r in records),
        "usable reports",
        flush=True,
    )


if __name__ == "__main__":
    main()
