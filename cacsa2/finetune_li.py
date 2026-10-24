"""Fine-tuning of released late-interaction checkpoints on minimal pairs (revision step 3, October 2026).

A Stanford-format ColBERT checkpoint (colbert-ir/colbertv2.0, answerdotai/answerai-colbert-small-v1) is trained end to
end with its own scoring (query = answer option or NevIR query; document = "<context> <question>" or NevIR document;
score = sum over query vectors of the maximum cosine over document vectors) on one of two training sets:
    negpairs  NegPairs v1 train (two options per item; cross-entropy over the two option scores), model selection
              on v1 dev accuracy;
    nevir     NevIR train (two documents per query; cross-entropy over the two document scores for both queries of an
              instance), model selection on NevIR validation paired accuracy.
After training, the model is scored exactly like the released systems (public_systems: ColBERTNative settings, float32)
on NegPairs-Perm, NegPairs-Fresh and NevIR test, and on BEIR SciFact (nDCG@10 by brute-force MaxSim over the corpus) as a
standard-retrieval check; the released checkpoint is scored the same way as the baseline. Weights are float32; the
encoder and the projection are trained; AdamW, linear warm-up and decay, gradient clipping at 1.0, patience on the
selection metric. Every run records its configuration, history, selected epoch and the SHA-256 of the selected state.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .negpairs import LABELS2, NegRow
from .public_systems import ColBERTCheckpoint, Scorer, negpairs_document


class ColBERTTrainer:
    """Wraps a public_systems.ColBERTCheckpoint for training: the same tokenization, markers, [MASK] expansion,
    punctuation masking and projection, with gradients."""

    def __init__(self, scorer: ColBERTCheckpoint, lr: float = 1e-5, weight_decay: float = 0.01, warmup_frac: float = 0.06,
                 total_steps: int = 1, grad_clip: float = 1.0, train_projection: bool = True):
        from transformers import get_linear_schedule_with_warmup
        self.sc = scorer
        self.nat = scorer.native
        self.model = self.nat.model
        self.device = self.nat.device
        self.model.float()
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.W = torch.nn.Parameter(self.nat.W.detach().float().clone())
        self.W.requires_grad_(train_projection)
        params = [p for p in self.model.parameters()] + ([self.W] if train_projection else [])
        self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.sched = get_linear_schedule_with_warmup(self.opt, int(warmup_frac * total_steps), max(1, total_steps))
        self.grad_clip = grad_clip
        self.skip_ids = torch.tensor(sorted(self.nat.skiplist), dtype=torch.long, device=self.device) if self.nat.skiplist else None

    def queries(self, texts: Sequence[str]) -> torch.Tensor:
        ids, attn, _ = self.nat._ids(texts, self.nat.q_id, self.nat.query_maxlen, pad_with_mask=True)
        if self.nat.attend_to_mask_tokens:
            attn = torch.ones_like(attn)
        h = self.model(input_ids=ids.to(self.device), attention_mask=attn.to(self.device)).last_hidden_state
        return F.normalize(h @ self.W.T, dim=-1)

    def docs(self, texts: Sequence[str]) -> torch.Tensor:
        ids, attn, _ = self.nat._ids(texts, self.nat.d_id, self.nat.doc_maxlen, pad_with_mask=False)
        ids, attn = ids.to(self.device), attn.to(self.device)
        h = self.model(input_ids=ids, attention_mask=attn).last_hidden_state
        d = h @ self.W.T
        keep = attn.bool()
        if self.skip_ids is not None:
            keep &= ~torch.isin(ids, self.skip_ids)
        d = d * keep.unsqueeze(-1).to(d.dtype)
        return F.normalize(d, dim=-1)

    def scores(self, queries: Sequence[str], docs: Sequence[str]) -> torch.Tensor:
        Q, D = self.queries(queries), self.docs(docs)
        return torch.matmul(Q, D.transpose(1, 2)).max(dim=2).values.sum(dim=1)

    def step(self, loss: torch.Tensor):
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in self.opt.param_groups for p in g["params"]], self.grad_clip)
        self.opt.step()
        self.sched.step()

    def sync_eval(self):
        """Copies the trained projection into the evaluation scorer (the model object is shared)."""
        self.nat.W = self.W.detach().clone()
        self.model.eval()

    def state(self) -> Dict[str, torch.Tensor]:
        sd = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        sd["__projection__"] = self.W.detach().cpu().clone()
        return sd

    def load_state(self, sd: Dict[str, torch.Tensor]):
        self.model.load_state_dict({k: v for k, v in sd.items() if k != "__projection__"})
        with torch.no_grad():
            self.W.copy_(sd["__projection__"].to(self.device))
        self.sync_eval()


# ------------------------------------------------------------------------------------------------ training sets
def negpairs_batches(rows: Sequence[NegRow], batch: int, rng: random.Random) -> List[List[NegRow]]:
    order = list(rows)
    rng.shuffle(order)
    return [order[i:i + batch] for i in range(0, len(order), batch)]


def negpairs_loss(tr: ColBERTTrainer, rows: Sequence[NegRow]) -> torch.Tensor:
    qs = [o for r in rows for o in r.options]
    ds = [negpairs_document(r.context, r.question) for r in rows for _ in r.options]
    s = tr.scores(qs, ds).view(len(rows), 2)
    y = torch.tensor([LABELS2.index(r.label) for r in rows], device=s.device)
    return F.cross_entropy(s, y)


def nevir_loss(tr: ColBERTTrainer, instances: Sequence[dict]) -> torch.Tensor:
    qs = [ins[q] for ins in instances for q in ("q1", "q2") for _ in ("doc1", "doc2")]
    ds = [ins[d] for ins in instances for _ in ("q1", "q2") for d in ("doc1", "doc2")]
    s = tr.scores(qs, ds).view(len(instances) * 2, 2)
    y = torch.tensor([0, 1] * len(instances), device=s.device)                  # q1 -> doc1, q2 -> doc2
    return F.cross_entropy(s, y)


@torch.no_grad()
def negpairs_accuracy(sc: Scorer, rows: Sequence[NegRow], batch_size: int = 32) -> float:
    vals = sc.score_negpairs([(o, r.context, r.question) for r in rows for o in r.options], batch_size=batch_size)
    ok = sum((vals[2 * k] >= vals[2 * k + 1]) == (r.label == "A") for k, r in enumerate(rows))
    return 100.0 * ok / len(rows)


@torch.no_grad()
def nevir_paired(sc: Scorer, instances: Sequence[dict], batch_size: int = 32) -> float:
    from .nevir import four_combos, paired_accuracy, to_score_dict
    combos = four_combos(instances)
    vals = sc.score_nevir([(c[2], c[3]) for c in combos], batch_size=batch_size)
    return paired_accuracy(to_score_dict(combos, vals), instances)["paired_accuracy"]


def finetune(scorer: ColBERTCheckpoint, train_set: str, train_data, dev_data, seed: int, lr: float = 1e-5, epochs: int = 5,
             batch: int = 16, patience: int = 2, warmup_frac: float = 0.06, verbose: bool = True, max_steps: Optional[int] = None) -> dict:
    """Trains `scorer` in place on `train_set` ('negpairs' rows or 'nevir' instances); returns the summary with the
    selected state loaded into the scorer. `dev_data` drives the selection (v1 dev accuracy or validation paired acc.)."""
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    rng = random.Random(seed)
    n_batches = math.ceil(len(train_data) / batch)
    total = epochs * n_batches if max_steps is None else min(max_steps, epochs * n_batches)
    tr = ColBERTTrainer(scorer, lr=lr, warmup_frac=warmup_frac, total_steps=total)
    loss_fn = negpairs_loss if train_set == "negpairs" else nevir_loss
    metric = (lambda: negpairs_accuracy(scorer, dev_data)) if train_set == "negpairs" else (lambda: nevir_paired(scorer, dev_data))
    tr.sync_eval()
    dev0 = metric()
    best = {"dev": dev0, "epoch": 0, "state": tr.state()}
    history = [{"epoch": 0, "train_loss": None, "dev": dev0}]
    t0, steps, bad = time.time(), 0, 0
    for epoch in range(1, epochs + 1):
        tr.model.train()
        tot, n = 0.0, 0
        for b in negpairs_batches(train_data, batch, rng):
            loss = loss_fn(tr, b)
            tr.step(loss)
            tot += float(loss.detach()) * len(b); n += len(b); steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        tr.sync_eval()
        dev = metric()
        history.append({"epoch": epoch, "train_loss": tot / max(1, n), "dev": dev, "elapsed_s": round(time.time() - t0, 1)})
        if verbose:
            print(f"[{scorer.name} {train_set} seed {seed}] epoch {epoch} loss {tot / max(1, n):.4f} dev {dev:.2f}", flush=True)
        if dev > best["dev"]:
            best = {"dev": dev, "epoch": epoch, "state": tr.state()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
        if max_steps is not None and steps >= max_steps:
            break
    tr.load_state(best["state"])
    import hashlib
    h = hashlib.sha256()
    for k in sorted(best["state"]):
        h.update(k.encode()); h.update(best["state"][k].numpy().tobytes())
    return {"train_set": train_set, "seed": seed, "lr": lr, "epochs": epochs, "patience": patience, "batch": batch, "warmup_frac": warmup_frac,
            "n_train": len(train_data), "n_dev": len(dev_data), "steps": steps, "selected_epoch": best["epoch"], "selected_dev": best["dev"],
            "dev_before": dev0, "history": history, "train_time_s": round(time.time() - t0, 1), "state_sha256": h.hexdigest(),
            "parameters": sum(p.numel() for p in tr.model.parameters()) + tr.W.numel()}


# ------------------------------------------------------------------------------------------------ SciFact (BEIR) check
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def load_scifact(cache_dir: str | Path = "/content/beir", url: str = SCIFACT_URL) -> Tuple[List[dict], List[dict], Dict[str, Dict[str, int]], dict]:
    """BEIR SciFact (corpus.jsonl, queries.jsonl, qrels/test.tsv) from the BEIR distribution zip (downloaded once into
    `cache_dir`; its SHA-256 is recorded). Returns (test queries, corpus with 'title text', qrels, provenance)."""
    import hashlib
    import io
    import urllib.request
    import zipfile
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zpath = cache_dir / "scifact.zip"
    if not zpath.exists():
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                zpath.write_bytes(r.read())
        except Exception as e:                                   # fallback: the hub copies (BeIR/scifact, BeIR/scifact-qrels)
            return _load_scifact_hub(repr(e)[:200])
    sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        def read(suffix):
            name = next(n for n in names if n.endswith(suffix))
            return z.read(name).decode("utf-8")
        corpus = [json.loads(l) for l in read("corpus.jsonl").splitlines() if l.strip()]
        queries = [json.loads(l) for l in read("queries.jsonl").splitlines() if l.strip()]
        qrels_txt = read("qrels/test.tsv").splitlines()
    rel: Dict[str, Dict[str, int]] = {}
    for line in qrels_txt[1:]:
        qid, did, score = line.split("\t")
        rel.setdefault(str(qid), {})[str(did)] = int(score)
    docs = [{"id": str(r["_id"]), "text": (f"{r.get('title', '')} {r['text']}".strip())} for r in corpus]
    qs = [{"id": str(r["_id"]), "text": r["text"]} for r in queries if str(r["_id"]) in rel]
    return qs, docs, rel, {"url": url, "zip_sha256": sha, "n_corpus": len(docs), "n_test_queries": len(qs), "n_qrels": sum(len(v) for v in rel.values())}


def _load_scifact_hub(reason: str):
    from datasets import load_dataset
    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels = load_dataset("BeIR/scifact-qrels", split="test")
    rel: Dict[str, Dict[str, int]] = {}
    for r in qrels:
        rel.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    docs = [{"id": str(r["_id"]), "text": (f"{r.get('title') or ''} {r['text']}".strip())} for r in corpus]
    qs = [{"id": str(r["_id"]), "text": r["text"]} for r in queries if str(r["_id"]) in rel]
    return qs, docs, rel, {"url": "hub: BeIR/scifact, BeIR/scifact-qrels", "zip_sha256": None, "fallback_reason": reason, "n_corpus": len(docs),
                           "n_test_queries": len(qs), "n_qrels": sum(len(v) for v in rel.values())}


def ndcg_at_k(ranked: Sequence[str], rel: Dict[str, int], k: int = 10) -> float:
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


@torch.no_grad()
def scifact_ndcg(scorer: ColBERTCheckpoint, queries: Sequence[dict], docs: Sequence[dict], rel: Dict[str, Dict[str, int]],
                 k: int = 10, batch_docs: int = 64, batch_queries: int = 32) -> dict:
    """Brute-force MaxSim retrieval over the corpus with the checkpoint's own settings; nDCG@k over the test queries."""
    nat = scorer.native
    D_list, n_tok = [], []
    for i in range(0, len(docs), batch_docs):
        D, _ = nat.encode_docs([d["text"] for d in docs[i:i + batch_docs]])
        D = D.half()
        for b in range(D.shape[0]):
            D_list.append(D[b]); n_tok.append(int(D.shape[1]))
    maxlen = max(n_tok)
    Dpad = torch.zeros(len(D_list), maxlen, D_list[0].shape[-1], dtype=torch.float16, device=nat.device)
    for j, d in enumerate(D_list):
        Dpad[j, : d.shape[0]] = d
    ids = [d["id"] for d in docs]
    nd = []
    per_query = {}
    for i in range(0, len(queries), batch_queries):
        Q = nat.encode_queries([q["text"] for q in queries[i:i + batch_queries]]).half()       # (b, q, dim)
        scores = torch.zeros(Q.shape[0], len(docs), device=nat.device)
        for j in range(0, len(docs), 512):
            chunk = Dpad[j:j + 512]                                                              # (c, n, dim)
            s = torch.einsum("bqd,cnd->bcqn", Q, chunk).max(dim=3).values.sum(dim=2)            # (b, c)
            scores[:, j:j + 512] = s.float()
        top = scores.topk(k, dim=1).indices.cpu().tolist()
        for b, q in enumerate(queries[i:i + batch_queries]):
            v = ndcg_at_k([ids[t] for t in top[b]], rel[q["id"]], k)
            per_query[q["id"]] = v
            nd.append(v)
    return {"ndcg@10": 100.0 * sum(nd) / len(nd), "n_queries": len(nd), "n_docs": len(docs), "per_query": per_query}
