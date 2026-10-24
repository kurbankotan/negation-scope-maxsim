"""NevIR external test (manifest v3.0, sections 4.8, 7.4, 8.2 R7).

Instances carry two queries and two documents; q1 is relevant to doc1 and q2 to doc2. Paired accuracy as in the
benchmark's evaluation code: an instance is correct if s(q1, doc1) > s(q1, doc2) and s(q2, doc2) > s(q2, doc1),
with strict comparisons (ties are errors). For the static scorers a margin of 1e-9 replaces the strict comparison.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

import torch

from .exact import bag, nested_instances


def load_test(dataset: str = "orionweller/NevIR", split: str = "test", revision: Optional[str] = None) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split, revision=revision)
    return [{"id": str(r["id"]), "q1": r["q1"], "q2": r["q2"], "doc1": r["doc1"], "doc2": r["doc2"]} for r in ds]


def paired_accuracy(scores: Dict[str, Dict[str, float]], instances: Sequence[dict], margin: float = 0.0) -> dict:
    """scores[id] = {"q1d1", "q1d2", "q2d1", "q2d2"}. Returns paired and per-query accuracy and the per-instance
    correctness list (for the bootstrap)."""
    corr, per_q = [], []
    for ins in instances:
        s = scores[ins["id"]]
        a = s["q1d1"] > s["q1d2"] + margin
        b = s["q2d2"] > s["q2d1"] + margin
        corr.append(int(a and b))
        per_q.extend([int(a), int(b)])
    n = len(instances)
    return {"n": n, "paired_accuracy": (100.0 * sum(corr) / n) if n else None,
            "per_query_accuracy": (100.0 * sum(per_q) / len(per_q)) if per_q else None, "correct": corr, "margin": margin}


def bootstrap_paired(correct: Sequence[int], n_boot: int = 10000, seed: int = 20261003, conf: float = 0.95) -> dict:
    import numpy as np
    x = np.asarray(correct, dtype=float)
    if len(x) == 0:
        return {"point": None, "low": None, "high": None, "n": 0}
    rng = np.random.default_rng(seed)
    reps = []
    for start in range(0, n_boot, 1000):
        m = min(1000, n_boot - start)
        ii = rng.integers(0, len(x), size=(m, len(x)))
        reps.append(100.0 * x[ii].mean(axis=1))
    reps = np.concatenate(reps)
    lo, hi = np.percentile(reps, [100 * (1 - conf) / 2, 100 * (1 + conf) / 2])
    return {"point": float(100.0 * x.mean()), "low": float(lo), "high": float(hi), "n": int(len(x)), "n_boot": n_boot}


def evaluate_scorer(scores: Dict[str, Dict[str, float]], instances: Sequence[dict], nested_ids: Sequence[str],
                    margin: float = 0.0, n_boot: int = 10000) -> dict:
    nested = set(nested_ids)
    out = {}
    for name, sub in (("all", list(instances)), ("nested", [i for i in instances if i["id"] in nested]),
                      ("complement", [i for i in instances if i["id"] not in nested])):
        acc = paired_accuracy(scores, sub, margin)
        out[name] = {k: v for k, v in acc.items() if k != "correct"}
        out[name]["bootstrap"] = bootstrap_paired(acc["correct"], n_boot)
    return out


@torch.no_grad()
def maxsim_scores_gpu(ctx_states: Sequence[torch.Tensor], qry_states: Sequence[torch.Tensor], device: str = "cuda") -> List[float]:
    """M0 for cached states: mean over query tokens of the maximum over document tokens of the dot product (cosine for
    unit vectors); float32 on the GPU, one pair at a time (documents differ in length)."""
    out = []
    for d, q in zip(ctx_states, qry_states):
        D = d.to(device, dtype=torch.float32)
        Q = q.to(device, dtype=torch.float32)
        out.append(float(torch.matmul(Q, D.T).max(dim=1).values.mean()))
    return out


def four_combos(instances: Sequence[dict]) -> List[tuple]:
    """(id, key, query text, document text) for the four query-document combinations of every instance."""
    out = []
    for ins in instances:
        for key, q, d in (("q1d1", "q1", "doc1"), ("q1d2", "q1", "doc2"), ("q2d1", "q2", "doc1"), ("q2d2", "q2", "doc2")):
            out.append((ins["id"], key, ins[q], ins[d]))
    return out


def to_score_dict(combos: Sequence[tuple], values: Sequence[float]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for (iid, key, _, _), v in zip(combos, values):
        out.setdefault(iid, {})[key] = float(v)
    return out
