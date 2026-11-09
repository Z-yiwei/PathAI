"""Mayo GPT pretraining followed by joint Mayo/TCGA training; one fixed 6/4 patient split."""

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch.nn import functional as F

from .common import write_json
from .data import read_rows, reports, tokens
from .model import Multimodal, report_loss
from .original import TOKEN_TO_ID, ToyAesclea, ToyConfig, collate, make_batch


def pretrain(rows, out):
    torch.manual_seed(0)
    rng = random.Random(0)
    train = [
        [TOKEN_TO_ID[e] for e in r["events"]]
        for r in rows
        if r["source"] == "mayo" and r["split"] == "train"
    ]
    val = [
        [TOKEN_TO_ID[e] for e in r["events"]]
        for r in rows
        if r["source"] == "mayo" and r["split"] == "val"
    ]
    model = ToyAesclea(ToyConfig(vocab_size=len(TOKEN_TO_ID) + 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    trace = []
    for step in range(301):
        if step:
            batch = make_batch(train, 32, rng, 64)
            loss = model.lm_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if step % 10 == 0 or step == 1:
            with torch.no_grad():
                trace.append(
                    {
                        "step": step,
                        "train_lm": float(model.lm_loss(collate(train, 64))),
                        "val_lm": float(model.lm_loss(collate(val, 64))),
                    }
                )
    torch.save(model.state_dict(), out / "gpt_pretrained.pt")
    write_json(out / "pretraining.json", trace)
    print("Original GPT pretraining:", trace[0], trace[-1], flush=True)
    return model.state_dict()


def metrics(y, p):
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "bce": float(log_loss(y, p, labels=[0, 1])),
        "auroc": float(roc_auc_score(y, p)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/run")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="TCGA split, new parameter initialization and joint-training batches",
    )
    parser.add_argument(
        "--pretrained-run",
        type=Path,
        help="Reuse the fixed seed-0 Mayo pretraining from an existing run",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gpt-lr", type=float, default=0.0001)
    parser.add_argument("--report-weight", type=float, default=0.1)
    parser.add_argument("--lm-weight", type=float, default=0.1)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    rows = read_rows("artifacts/processed/patients.jsonl")
    vocab = json.loads(Path("artifacts/processed/vocab.json").read_text())
    if args.pretrained_run is None:
        pretrained = pretrain(rows, out)
    else:
        pretrained = torch.load(
            args.pretrained_run / "gpt_pretrained.pt",
            map_location="cpu",
            weights_only=True,
        )
        torch.save(pretrained, out / "gpt_pretrained.pt")
        write_json(
            out / "pretraining.json",
            json.loads((args.pretrained_run / "pretraining.json").read_text()),
        )
    tcga = sorted(
        [r for r in rows if r["source"] == "tcga"], key=lambda r: r["patient_id"]
    )
    mayo = [r for r in rows if r["source"] == "mayo" and r["split"] == "train"]
    mayo_val = [r for r in rows if r["source"] == "mayo" and r["split"] == "val"]
    t = tokens(tcga, vocab)
    m = tokens(mayo, vocab)
    mv = tokens(mayo_val, vocab)
    r, available = reports(tcga)
    mr, ma = reports(mayo)
    y = torch.tensor([p["label"] for p in tcga], dtype=torch.float32)
    image = F.normalize(
        torch.tensor(
            np.stack(
                [
                    np.load(f"artifacts/image_features/{p['patient_id']}.npz")[
                        "features"
                    ].mean(0)
                    for p in tcga
                ]
            )
        ),
        dim=1,
    )
    split_rng = random.Random(args.seed)
    validation = []
    for label in [0, 1]:
        indices = [i for i in range(len(tcga)) if int(y[i]) == label]
        split_rng.shuffle(indices)
        validation.extend(indices[:2])
    te = np.array(sorted(validation))
    tr = np.array([i for i in range(len(tcga)) if i not in validation])
    write_json(
        out / "config.json",
        {
            **vars(args),
            "pretrained_run": str(args.pretrained_run) if args.pretrained_run else None,
            "pretrain_seed": 0,
            "split": "6 train / 4 validation",
            "pretrain_steps": 300,
            "checkpoint_rule": "minimum validation BCE; check every 10 steps, patience 20 steps",
            "gpt_source": "../aesclepius-jr/model.py",
            "train_source": "../aesclepius-jr/train.py",
            "mayo_train": [p["patient_id"] for p in mayo],
            "mayo_val": [p["patient_id"] for p in mayo_val],
        },
    )
    write_json(
        out / "split.json",
        {
            "train": [tcga[i]["patient_id"] for i in tr],
            "val": [tcga[i]["patient_id"] for i in te],
        },
    )
    summaries = {}
    prediction_rows = []
    history = []
    for mode in ["events", "image", "both"]:
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        model = Multimodal(len(vocab) + 1, pretrained)
        optimizer = torch.optim.AdamW(
            [
                {"params": model.gpt.parameters(), "lr": args.gpt_lr},
                {
                    "params": list(model.image_projection.parameters())
                    + list(model.outcome.parameters())
                    + list(model.report.parameters()),
                    "lr": args.lr,
                },
            ],
            weight_decay=0.01,
        )
        im = image
        trace = []
        initial_gradients = {}
        best = None
        best_loss = float("inf")
        best_step = 0
        for step in range(args.steps + 1):
            if step:
                model.train()
                idx = rng.choice(len(mayo), 32, replace=False)
                logits, pred = model(t[tr], im[tr], mode)
                class_weight = torch.where(
                    y[tr] > 0,
                    len(tr) / (2 * y[tr].sum()),
                    len(tr) / (2 * (len(tr) - y[tr].sum())),
                )
                bce = F.binary_cross_entropy_with_logits(
                    logits, y[tr], weight=class_weight
                )
                lm = model.gpt.lm_loss(m[idx]) + model.event_loss(
                    t[tr], im[tr] if mode == "both" else None
                )
                _, mreport = model(m[idx], None, "events")
                report = report_loss(pred, r[tr], available[tr]) + report_loss(
                    mreport, mr[idx], ma[idx]
                )
                loss = bce + args.lm_weight * lm + args.report_weight * report
                optimizer.zero_grad()
                loss.backward()
                if step == 1:
                    for name, parameter in [
                        ("gpt_attention", model.gpt.blocks[0].attn.qkv.weight),
                        ("image_projection", model.image_projection.weight),
                        ("report_head", model.report.weight),
                    ]:
                        initial_gradients[name] = (
                            float(parameter.grad.norm())
                            if parameter.grad is not None
                            else 0.0
                        )
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if step % 10 == 0 or step == 1:
                model.eval()
                with torch.no_grad():
                    train_logits, train_report = model(t[tr], im[tr], mode)
                    eval_logits, _ = model(t[te], im[te], mode)
                    trace.append(
                        {
                            "step": step,
                            "train_bce": float(
                                F.binary_cross_entropy_with_logits(train_logits, y[tr])
                            ),
                            "eval_bce": float(
                                F.binary_cross_entropy_with_logits(eval_logits, y[te])
                            ),
                            "report_loss": float(
                                report_loss(train_report, r[tr], available[tr])
                            ),
                            "mayo_val_lm": float(model.gpt.lm_loss(mv)),
                        }
                    )
                if step and trace[-1]["eval_bce"] < best_loss - 1e-6:
                    best_loss = trace[-1]["eval_bce"]
                    best_step = step
                    best = copy.deepcopy(model.state_dict())
                if step and step - best_step >= 20:
                    break
        model.load_state_dict(best)
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(t[te], im[te], mode)[0]).numpy()
        torch.save(
            {
                "model": model.state_dict(),
                "mode": mode,
                "vocab": vocab,
                "selected_step": best_step,
                "train_ids": [tcga[i]["patient_id"] for i in tr],
            },
            out / f"{mode}.pt",
        )
        history.append(
            {
                "mode": mode,
                "initial_gradients": initial_gradients,
                "selected_step": best_step,
                "trace": trace,
            }
        )
        prediction_rows.extend(
            {
                "mode": mode,
                "patient_id": tcga[i]["patient_id"],
                "label": int(y[i]),
                "probability": float(prob),
            }
            for i, prob in zip(te, p)
        )
        print(mode, "selected step", best_step, "validation BCE", best_loss, flush=True)
        summaries[mode] = metrics(y[te].numpy(), p)
        print(mode, summaries[mode], flush=True)
    write_json(out / "metrics.json", summaries)
    write_json(out / "history.json", history)
    with (out / "predictions.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
