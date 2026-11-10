"""Patient preparation and feature loading / 患者数据准备与特征加载。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import write_json, write_jsonl
from .events import TOKEN_TO_ID, collate
from .report_quality import normalize_text, quality


def read_rows(path):
    """Read patient JSONL records / 读取患者 JSONL 记录。

    Input / 输入:
        path: JSONL file.
        path：JSONL 文件路径。
    Processing / 处理:
        Parse each line as JSON.
        逐行解析 JSON。
    Output / 输出:
        List of patient dictionaries.
        患者字典列表。
    """
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def prefix(events):
    """Keep TCGA events through resection / 截取 TCGA 手术前缀。

    Input / 输入:
        events: full ordered event-name list.
        events：完整、有序的事件名称列表。
    Processing / 处理:
        Retain the first resection; reject DEATH/progression in the retained input.
        保留首次手术及之前事件；检查前缀中没有死亡或进展事件。
    Output / 输出:
        Input event list; raises ValueError for missing surgery or outcome leakage.
        输入事件列表；缺少手术或混入结局事件时抛出 ValueError。
    """
    if "PROC_RESECTION" not in events:
        raise ValueError("TCGA input requires PROC_RESECTION")
    result = events[: events.index("PROC_RESECTION") + 1]
    if {"DEATH", "EVENT_PROGRESSION"} & set(result):
        raise ValueError("Outcome event in input")
    return result


def prepare(out=Path("artifacts/processed"), data_dir=Path("data")):
    """Validate and prepare patient records / 检查并整理患者数据。

    Input / 输入:
        data_dir: Mayo/TCGA raw-data root; out: processed-output directory.
        data_dir：Mayo/TCGA 原始数据根目录；out：预处理输出目录。
    Processing / 处理:
        Validate IDs/events and paired files, copy normalized reports, derive TCGA labels/prefixes, and extend the vocabulary.
        检查 ID、事件与配对文件，保存规范化报告副本，生成 TCGA 标签与前缀并扩展词表。
    Output / 输出:
        (rows, vocab); writes patients.jsonl, vocab.json, quality.json, and report copies.
        返回 (患者列表, 词表)，保存患者记录、词表、质量统计及报告副本；不改原始数据。
    """
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
    # Append new events without changing IDs inherited from Mayo pretraining.
    # 新事件追加到词表，保持 Mayo 预训练事件的 ID 不变。
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
    """Convert prepared events into a model batch / 将事件转换为模型 batch。

    Input / 输入:
        rows: prepared patients; vocab: event-name-to-ID mapping.
        rows：预处理患者记录；vocab：事件名到 ID 的映射。
    Processing / 处理:
        Check context limits, map input_events to IDs, then right-pad via collate.
        检查长度，转换 input_events，再通过 collate 右侧补零。
    Output / 输出:
        Integer tensor [B,T]; TCGA reserves one position for the image prefix.
        整数张量 [B,T]；TCGA 为图像前缀预留一个位置。
    """
    if any(
        len(r["input_events"]) > (63 if r["source"] == "tcga" else 64) for r in rows
    ):
        raise ValueError("Sequence exceeds GPT context length")
    return collate([[vocab[e] for e in r["input_events"]] for r in rows], 64)


def reports(rows):
    """Load cached report supervision / 加载报告监督缓存。

    Input / 输入:
        rows: patients whose IDs match artifacts/report_features/*.npz.
        rows：患者记录，ID 对应 artifacts/report_features 下的 NPZ。
    Processing / 处理:
        Stack target vectors and availability flags in the same patient order.
        按患者顺序堆叠目标向量与可用标记。
    Output / 输出:
        Float targets [B,512] and Boolean availability [B].
        浮点目标 [B,512] 和布尔可用掩码 [B]。
    """
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
