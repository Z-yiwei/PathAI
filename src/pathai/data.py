"""Read original events and write formatting-only report copies."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import write_json, write_jsonl
from .events import TOKEN_TO_ID, collate
from .report_quality import normalize_text, quality


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def prefix(events):
    if "PROC_RESECTION" not in events:
        raise ValueError("TCGA input requires PROC_RESECTION")
    result = events[: events.index("PROC_RESECTION") + 1]
    if {"DEATH", "EVENT_PROGRESSION"} & set(result):
        raise ValueError("Outcome event in input")
    return result


def prepare(out=Path("artifacts/processed"), data_dir=Path("data")):
    data_dir = Path(data_dir).expanduser().resolve()
    out = Path(out)
    rows, seen = [], set()
    for source, split, file in [
        ("mayo", "train", "train.jsonl"),
        ("mayo", "val", "val.jsonl"),
        ("tcga", "unassigned", "tcga.jsonl"),
    ]:
        for item in read_rows(data_dir / source / file):
            if set(item) != {"patient_id", "events"}:
                raise ValueError("Unexpected JSON fields")
            pid, events = item["patient_id"], item["events"]
            if not isinstance(pid, str) or pid in seen:
                raise ValueError("Invalid/duplicate patient ID")
            if (
                not isinstance(events, list)
                or len(events) < 2
                or any(
                    not isinstance(e, str) or not e or e != e.strip() for e in events
                )
            ):
                raise ValueError("Invalid event sequence")
            seen.add(pid)
            report = data_dir / source / "reports" / f"{pid}.txt"
            slide = data_dir / source / "slides" / f"{pid}.svs"
            if source == "tcga" and (not report.exists() or not slide.exists()):
                raise ValueError("Missing TCGA file")
            clean_path, qc = None, None
            if report.exists():
                text = normalize_text(report.read_text(encoding="utf-8"))
                clean_path = out / "reports" / source / f"{pid}.txt"
                clean_path.parent.mkdir(parents=True, exist_ok=True)
                clean_path.write_text(text)
                qc = quality(text)
            usable = qc is not None and qc["status"] == "pass"
            rows.append(
                {
                    "patient_id": pid,
                    "source": source,
                    "split": split,
                    "events": events,
                    "input_events": prefix(events) if source == "tcga" else events,
                    "label": int("DEATH" in events) if source == "tcga" else None,
                    "clean_report_path": (
                        str(clean_path.resolve()) if clean_path else None
                    ),
                    "report_trainable": usable,
                    "report_quality": qc,
                    "report_validation_status": (
                        "validated" if usable else "unavailable"
                    ),
                    "slide_path": str(slide) if slide.exists() else None,
                }
            )
    vocab = dict(TOKEN_TO_ID)
    for event in sorted({e for r in rows for e in r["events"]} - set(vocab)):
        vocab[event] = len(vocab) + 1
    write_jsonl(out / "patients.jsonl", rows)
    write_json(out / "vocab.json", vocab)
    write_json(
        out / "quality.json",
        {
            "patients": len(rows),
            "tcga_patients": sum(r["source"] == "tcga" for r in rows),
            "reports": sum(r["clean_report_path"] is not None for r in rows),
            "usable_reports": sum(r["report_trainable"] for r in rows),
            "unreadable_reports": [
                r["patient_id"]
                for r in rows
                if r["clean_report_path"] and not r["report_trainable"]
            ],
        },
    )
    return rows, vocab


def tokens(rows, vocab):
    if any(
        len(r["input_events"]) > (63 if r["source"] == "tcga" else 64) for r in rows
    ):
        raise ValueError("Sequence exceeds GPT context length")
    return collate([[vocab[e] for e in r["input_events"]] for r in rows], 64)


def reports(rows):
    values = [
        np.load(Path("artifacts/report_features") / f"{r['patient_id']}.npz")
        for r in rows
    ]
    return torch.tensor(np.stack([v["features"] for v in values])), torch.tensor(
        [bool(v["available"]) for v in values]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/processed"))
    args = parser.parse_args()
    prepare(args.out, args.data_dir)
    print((args.out / "quality.json").read_text())
