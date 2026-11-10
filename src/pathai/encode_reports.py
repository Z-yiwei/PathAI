"""Frozen report encoding / 冻结编码器生成报告监督目标。"""

import argparse
import json
from pathlib import Path

import numpy as np

from .common import write_json
from .report_quality import load_training_report


def contiguous_chunks(token_ids, size=75):
    """Split token IDs without dropping text / 连续分块且保留全部文本。

    Input / 输入:
        token_ids: full token list; size: content-token limit, default 75.
        完整 token ID 列表；size：每块内容 token 上限，默认 75。
    Processing / 处理:
        Slice contiguous non-overlapping chunks, keeping the shorter final chunk.
        连续无重叠分块，保留不足上限的最后一块。
    Output / 输出:
        List of token-ID lists; empty input returns an empty list.
        token ID 子列表集合；空输入返回空列表。
    """
    return [token_ids[i : i + size] for i in range(0, len(token_ids), size)]


def main():
    """Cache PLIP report targets / 缓存 PLIP 报告目标。

    Input / 输入:
        CLI --patients and --out; readable report copies and local frozen PLIP weights.
        命令行 --patients、--out；报告副本和本地冻结 PLIP 权重，需要 CUDA。
    Processing / 处理:
        Encode complete reports in 75-token chunks with BOS/EOS; normalize, weight by text-token count, and normalize again.
        完整报告按 75 token 分块并加 BOS/EOS，编码归一化后按 token 数加权，再次归一化。
    Output / 输出:
        Patient NPZ files with features [512] and available; encoding.json; returns None.
        保存每人 [512] 特征、available 标记和 encoding.json，无返回值。
    """
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
                # Weight by content length so the final short chunk contributes proportionally.
                # 按内容 token 数加权，使最后一个短块的贡献与其长度成比例。
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
