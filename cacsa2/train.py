"""Training, selection and prediction archiving for one (encoder, aggregator, seed) run.

Protocol (manifest section 5): AdamW, lr 1e-3, weight decay 0.01, batch 16, up to 20 epochs,
cross-entropy over candidate scores, early stopping on dev accuracy (patience 5), checkpoint
selection on dev only, test scored once with the selected checkpoint.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .condaqa import LABELS, Row, render_candidates, context_text, eligible_rows, unseen_cue_flags
from .encoders import EncodedStore
from .model import CACSAModule
from .scope import ScopeResolver


@dataclass
class TrainConfig:
    aggregator: str = "max"
    lr: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 16
    epochs: int = 20
    patience: int = 5
    proj: int = 128
    hidden: int = 256
    dropout: float = 0.1
    seed: int = 13
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_clip: float = 1.0
    max_ctx_tokens: Optional[int] = None     # optional hard cap on cached context tokens (None = keep all)
    label_set: Tuple[str, ...] = LABELS
    amp: bool = True                         # bf16 autocast on CUDA for the module forward pass (loss in fp32)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label_set"] = list(self.label_set)
        return d


@dataclass
class Item:
    row: Row
    ctx: torch.Tensor            # (n,d) fp16
    cands: List[torch.Tensor]    # K x (m_k,d) fp16
    polarity: torch.Tensor       # (n,) uint8
    label: int
    unseen_cue: bool = False


def build_items(rows: Sequence[Row], store: EncodedStore, resolver: Optional[ScopeResolver],
                train_rows_for_cues: Optional[Sequence[Row]] = None, label_set: Tuple[str, ...] = LABELS,
                max_ctx_tokens: Optional[int] = None, polarity_cache: Optional[dict] = None, progress: bool = False) -> List[Item]:
    """Encode (through the store) and assemble items for eligible rows. Polarity masks come from the resolver
    (all zeros when resolver is None, which is what A1-A3 and A5 use)."""
    rows = [r for r in rows if r.label in label_set]
    unseen = unseen_cue_flags(train_rows_for_cues, rows) if train_rows_for_cues is not None else {}
    contexts = [context_text(r.question, r.passage) for r in rows]
    cands = render_candidates(labels=label_set)
    store.get_or_encode(contexts, progress=progress)
    store.get_or_encode(cands, progress=progress)
    if resolver is not None:
        resolver.prefetch(contexts, progress=progress)     # batched spaCy parse; cached by text across encoders
    polarity_cache = polarity_cache if polarity_cache is not None else {}
    items: List[Item] = []
    for idx, r in enumerate(rows):
        text = context_text(r.question, r.passage)
        ctx = store.get(text)
        states = ctx.states
        offsets = ctx.offsets
        if max_ctx_tokens is not None and states.shape[0] > max_ctx_tokens:
            states, offsets = states[:max_ctx_tokens], offsets[:max_ctx_tokens]
        if resolver is not None:
            key = (text, tuple(offsets))                      # offsets differ between encoders
            pol = polarity_cache.get(key)
            if pol is None:
                pol = torch.tensor(resolver.polarity_mask(text, offsets), dtype=torch.uint8)
                polarity_cache[key] = pol
        else:
            pol = torch.zeros(len(offsets), dtype=torch.uint8)
        cand_states = [store.get(c).states for c in cands]
        items.append(Item(row=r, ctx=states, cands=cand_states, polarity=pol, label=label_set.index(r.label),
                          unseen_cue=bool(unseen.get(r.sample_id, False))))
        if progress and idx % 1000 == 0:
            print(f"  items {idx}/{len(rows)}", end="\r")
    if progress:
        print()
    return items


def collate(batch: Sequence[Item], device: str) -> dict:
    B = len(batch)
    K = len(batch[0].cands)
    d = batch[0].ctx.shape[1]
    n = max(it.ctx.shape[0] for it in batch)
    m = max(c.shape[0] for it in batch for c in it.cands)
    ctx = torch.zeros(B, n, d)
    ctx_mask = torch.zeros(B, n, dtype=torch.bool)
    pol = torch.zeros(B, n)
    cands = torch.zeros(B, K, m, d)
    cand_mask = torch.zeros(B, K, m, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)
    for b, it in enumerate(batch):
        nn_ = it.ctx.shape[0]
        ctx[b, :nn_] = it.ctx.float()
        ctx_mask[b, :nn_] = True
        pol[b, :nn_] = it.polarity.float()
        for k, c in enumerate(it.cands):
            cands[b, k, : c.shape[0]] = c.float()
            cand_mask[b, k, : c.shape[0]] = True
        labels[b] = it.label
    return {"ctx": ctx.to(device), "ctx_mask": ctx_mask.to(device), "polarity": pol.to(device),
            "cands": cands.to(device), "cand_mask": cand_mask.to(device), "labels": labels.to(device)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


@torch.no_grad()
def predict(model: CACSAModule, items: Sequence[Item], cfg: TrainConfig, label_set: Tuple[str, ...] = LABELS) -> Tuple[List[dict], float]:
    model.eval()
    records: List[dict] = []
    total_loss = 0.0
    items = sorted(items, key=lambda it: it.ctx.shape[0])
    use_amp = cfg.amp and str(cfg.device).startswith("cuda")
    for i in range(0, len(items), cfg.batch_size):
        batch = items[i:i + cfg.batch_size]
        tb = collate(batch, cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            scores = model(tb["ctx"], tb["ctx_mask"], tb["cands"], tb["cand_mask"], polarity=tb["polarity"])
        scores = scores.float()
        total_loss += F.cross_entropy(scores, tb["labels"], reduction="sum").item()
        probs = F.softmax(scores, dim=1).cpu()
        for it, p, s in zip(batch, probs, scores.cpu()):
            r = it.row
            records.append({"sample_id": r.sample_id, "passage_id": r.passage_id, "question_id": r.question_id,
                            "edit_id": r.edit_id, "label": r.label, "pred": label_set[int(p.argmax())],
                            "probs": [round(float(x), 6) for x in p], "scores": [round(float(x), 6) for x in s],
                            "cue": r.cue, "unseen_cue": it.unseen_cue})
    return records, total_loss / max(1, len(items))


def train_run(cfg: TrainConfig, dim: int, train_items: Sequence[Item], dev_items: Sequence[Item],
              test_items: Optional[Sequence[Item]] = None, out_dir: Optional[str | Path] = None,
              label_set: Tuple[str, ...] = LABELS, verbose: bool = True, model_factory=None) -> dict:
    """Train one model; select on dev accuracy (tie -> lower dev loss); score test once.

    `model_factory(dim, cfg) -> nn.Module` overrides the default CACSAModule (used for the head-only control);
    the module must expose forward(ctx, ctx_mask, cands, cand_mask, polarity) -> (B,K) and parameter_counts()."""
    set_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_amp = cfg.amp and str(cfg.device).startswith("cuda")
    if model_factory is None:
        model = CACSAModule(dim, cfg.aggregator, proj=cfg.proj, hidden=cfg.hidden, dropout=cfg.dropout).to(cfg.device)
    else:
        model = model_factory(dim, cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = random.Random(cfg.seed)
    order = list(range(len(train_items)))
    history: List[dict] = []

    def batches_for_epoch() -> List[List[int]]:
        """Shuffle, then sort within chunks of 50 batches by context length to limit padding; shuffle batch order."""
        rng.shuffle(order)
        chunk = cfg.batch_size * 50
        out: List[List[int]] = []
        for i in range(0, len(order), chunk):
            block = sorted(order[i:i + chunk], key=lambda j: train_items[j].ctx.shape[0])
            out.extend(block[k:k + cfg.batch_size] for k in range(0, len(block), cfg.batch_size))
        rng.shuffle(out)
        return out
    best = {"dev_acc": -1.0, "dev_loss": float("inf"), "epoch": None, "state": None}
    bad_epochs = 0
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tot_loss, tot_correct = 0.0, 0
        for idxs in batches_for_epoch():
            batch = [train_items[j] for j in idxs]
            tb = collate(batch, cfg.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                scores = model(tb["ctx"], tb["ctx_mask"], tb["cands"], tb["cand_mask"], polarity=tb["polarity"])
            loss = F.cross_entropy(scores.float(), tb["labels"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot_loss += loss.item() * len(batch)
            tot_correct += (scores.argmax(1) == tb["labels"]).sum().item()
        dev_records, dev_loss = predict(model, dev_items, cfg, label_set)
        dev_acc = 100.0 * sum(r["pred"] == r["label"] for r in dev_records) / len(dev_records)
        is_cacsa = hasattr(model, "scorer") and hasattr(model, "agg")
        rec = {"epoch": epoch, "train_loss": tot_loss / len(order), "train_acc": 100.0 * tot_correct / len(order),
               "dev_loss": dev_loss, "dev_acc": dev_acc,
               "lambda": (model.scorer.lam.item() if is_cacsa else None),
               "beta": (model.agg.beta.item() if is_cacsa and model.agg.beta is not None else None),
               "tau": (model.agg.tau.item() if is_cacsa and model.agg.tau is not None else None),
               "elapsed_s": round(time.time() - t0, 1)}
        history.append(rec)
        if verbose:
            print(f"[{cfg.aggregator} seed {cfg.seed}] epoch {epoch:2d} train {rec['train_acc']:.1f} dev {dev_acc:.2f} "
                  f"(loss {dev_loss:.4f})" + (f" lam {rec['lambda']:.3f}" if rec["lambda"] is not None else "")
                  + (f" beta {rec['beta']:.3f}" if rec["beta"] is not None else ""))
        improved = (dev_acc > best["dev_acc"]) or (dev_acc == best["dev_acc"] and dev_loss < best["dev_loss"])
        if improved:
            best = {"dev_acc": dev_acc, "dev_loss": dev_loss, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break
    model.load_state_dict(best["state"])
    dev_records, dev_loss = predict(model, dev_items, cfg, label_set)
    result = {"config": cfg.to_dict(), "dim": dim, "parameter_counts": model.parameter_counts(),
              "selected_epoch": best["epoch"], "epochs_run": len(history), "history": history,
              "dev": {"accuracy": best["dev_acc"], "loss": dev_loss, "records": dev_records},
              "train_time_s": round(time.time() - t0, 1)}
    if test_items is not None:
        test_records, test_loss = predict(model, test_items, cfg, label_set)
        result["test"] = {"accuracy": 100.0 * sum(r["pred"] == r["label"] for r in test_records) / len(test_records),
                          "loss": test_loss, "records": test_records}
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best["state"], out_dir / "module_state.pt")
        slim = {k: v for k, v in result.items() if k not in ("dev", "test")}
        slim["dev"] = {k: v for k, v in result["dev"].items() if k != "records"}
        if "test" in result:
            slim["test"] = {k: v for k, v in result["test"].items() if k != "records"}
        (out_dir / "run_summary.json").write_text(json.dumps(slim, indent=2))
        _write_records(out_dir / "dev_predictions.jsonl", result["dev"]["records"])
        if "test" in result:
            _write_records(out_dir / "test_predictions.jsonl", result["test"]["records"])
    result["model"] = model
    return result


def _write_records(path: Path, records: Sequence[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


@torch.no_grad()
def candidate_scores(model: CACSAModule, cfg: TrainConfig, ctx_states: torch.Tensor, cand_states: Sequence[torch.Tensor],
                     polarity: Optional[torch.Tensor] = None) -> List[float]:
    """Raw candidate scores for one context (used by the Proposition-1 cue-deletion test)."""
    model.eval()
    it = Item(row=None, ctx=ctx_states, cands=list(cand_states),
              polarity=(polarity if polarity is not None else torch.zeros(ctx_states.shape[0], dtype=torch.uint8)), label=0)
    tb = collate([it], cfg.device)
    s = model(tb["ctx"], tb["ctx_mask"], tb["cands"], tb["cand_mask"], polarity=tb["polarity"])
    return [float(x) for x in s[0].cpu()]
