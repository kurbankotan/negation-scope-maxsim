"""Control C1: full fine-tuning of a transformer cross-encoder on CondaQA (3-class), the upper-bound reference.

Input: "<question> [SEP] <passage>" (question first; truncation "longest_first" shortens the passage, the longer member, and never raises when a question is long).
Precision (since 0.1.2): weights are loaded explicitly in float32 (transformers >= 5 otherwise loads the checkpoint's
stored half precision, which made every step of the v1.1 run NaN); training runs in float32 (TF32 matmuls on Ampere) unless
`bf16=True`; any non-finite loss aborts the run with FloatingPointError instead of silently producing constant predictions.
Standard recipe:
AdamW lr 2e-5, weight decay 0.01, batch 16, 3 epochs, linear warm-up 6%, selection on dev accuracy,
test scored once. Predictions are archived in the same record format as the module runs so that
`cacsa2.analysis` can read them (encoder = "C1-fullft", aggregator = model short name).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from .condaqa import LABELS, Row, unseen_cue_flags


def _records(rows: Sequence[Row], probs: torch.Tensor, unseen: dict) -> List[dict]:
    out = []
    for r, p in zip(rows, probs):
        out.append({"sample_id": r.sample_id, "passage_id": r.passage_id, "question_id": r.question_id,
                    "edit_id": r.edit_id, "label": r.label, "pred": LABELS[int(p.argmax())],
                    "probs": [round(float(x), 6) for x in p], "cue": r.cue, "unseen_cue": bool(unseen.get(r.sample_id, False))})
    return out


def finetune_run(model_name: str, train_rows: Sequence[Row], dev_rows: Sequence[Row], test_rows: Sequence[Row],
                 seed: int, out_dir: str | Path, lr: float = 2e-5, epochs: int = 3, batch_size: int = 16,
                 max_length: int = 512, warmup_frac: float = 0.06, device: Optional[str] = None,
                 revision: Optional[str] = None, bf16: bool = False, verbose: bool = True,
                 state_dir: str | Path = "/content/fullft_states", keep_state: bool = False) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(LABELS), revision=revision,
                                                                   dtype=torch.float32)
    except TypeError:  # older transformers versions only know `torch_dtype`
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(LABELS), revision=revision,
                                                                   torch_dtype=torch.float32)
    model = model.float().to(device)
    param_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    if param_dtypes != ["torch.float32"]:
        raise RuntimeError(f"fine-tuning requires float32 weights, got {param_dtypes}")
    train_rows = [r for r in train_rows if r.eligible]
    dev_rows = [r for r in dev_rows if r.eligible]
    test_rows = [r for r in test_rows if r.eligible]
    unseen = unseen_cue_flags(train_rows, list(dev_rows) + list(test_rows))

    def encode(rows):
        return tok([r.question for r in rows], [r.passage for r in rows], truncation="longest_first",
                   max_length=max_length, padding=True, return_tensors="pt")

    @torch.no_grad()
    def predict(rows):
        model.eval()
        probs = []
        order = sorted(range(len(rows)), key=lambda i: len(rows[i].passage))
        for i in range(0, len(order), batch_size * 2):
            idx = order[i:i + batch_size * 2]
            enc = {k: v.to(device) for k, v in encode([rows[j] for j in idx]).items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(bf16 and device == "cuda")):
                logits = model(**enc).logits.float()
            probs.append((idx, F.softmax(logits, dim=-1).cpu()))
        out = torch.zeros(len(rows), len(LABELS))
        for idx, p in probs:
            out[idx] = p
        return out

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(train_rows) + batch_size - 1) // batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(warmup_frac * steps), steps)
    rng = random.Random(seed)
    best = {"dev_acc": -1.0, "epoch": None, "state": None}
    history = []
    t0 = time.time()
    labels_t = torch.tensor([r.label_index for r in train_rows])
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_rows))); rng.shuffle(order)
        tot = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            enc = {k: v.to(device) for k, v in encode([train_rows[j] for j in idx]).items()}
            y = labels_t[idx].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(bf16 and device == "cuda")):
                logits = model(**enc).logits.float()
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss {loss.item()} at epoch {epoch}, batch starting {i}")
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(idx)
        dev_probs = predict(dev_rows)
        dev_acc = 100.0 * sum(LABELS[int(p.argmax())] == r.label for r, p in zip(dev_rows, dev_probs)) / len(dev_rows)
        history.append({"epoch": epoch, "train_loss": tot / len(order), "dev_acc": dev_acc, "elapsed_s": round(time.time() - t0, 1)})
        if verbose:
            print(f"[fullft {model_name.split('/')[-1]} seed {seed}] epoch {epoch} train_loss {tot/len(order):.4f} dev {dev_acc:.2f}")
        if dev_acc > best["dev_acc"]:
            best = {"dev_acc": dev_acc, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    dev_rec = _records(dev_rows, predict(dev_rows), unseen)
    test_rec = _records(test_rows, predict(test_rows), unseen)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"config": {"model_name": model_name, "revision": revision, "lr": lr, "epochs": epochs, "batch_size": batch_size,
                          "max_length": max_length, "seed": seed, "aggregator": model_name.split("/")[-1] + "-ft"},
               "parameter_counts": {"total": sum(p.numel() for p in model.parameters())},
               "selected_epoch": best["epoch"], "epochs_run": len(history), "history": history,
               "dev": {"accuracy": best["dev_acc"]},
               "test": {"accuracy": 100.0 * sum(r["pred"] == r["label"] for r in test_rec) / len(test_rec)},
               "train_time_s": round(time.time() - t0, 1)}
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    for name, recs in (("dev_predictions.jsonl", dev_rec), ("test_predictions.jsonl", test_rec)):
        with open(out_dir / name, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    # full-model weights (~0.7 GB) are not redistributed and are kept off Google Drive (quota and trash);
    # they are written to local disk, hashed, and removed unless keep_state=True
    state_dir = Path(state_dir); state_dir.mkdir(parents=True, exist_ok=True)
    sd_path = state_dir / f"{model_name.split('/')[-1]}_seed{seed}.pt"
    torch.save(best["state"], sd_path)
    from .provenance import sha256_file
    summary["model_state_sha256"] = sha256_file(sd_path)
    summary["model_state_bytes"] = sd_path.stat().st_size
    if not keep_state:
        sd_path.unlink()
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
