"""Exact rules of manifest v3.0 (sections 8.2 and 10) in float64 on the CPU, and the pilot checks of section 11.

R5 (Proposition B, insertion pairs), R6 (Proposition A, permutation pairs) and R7 (Corollary B.1, nested NevIR
documents) are evaluated on the static encoders E0 and E3 with the archived module states; the states are regenerated
in float64, no autocast, no TF32, tolerance 1e-9. Batches are padded; padding cannot change a row's own scores
(masked positions get -1e4 before a maximum, and exp(-1e4 / tau) underflows to exactly zero in float64).
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .encoders import word_tokenize
from .model import make_module
from .negpairs import ARMS, NegRow
from .train import Item, collate

TOL = 1e-9
MONOTONE = ("max", "lse", "m1")                       # Proposition B
SYMMETRIC = ("max", "mean", "lse", "signed_max_gate", "m1", "head_only")   # Proposition A (unmasked rules)


def load_condition(state_path: str | Path, dim: int, kind: str) -> torch.nn.Module:
    m = make_module(kind, dim)
    m.load_state_dict(torch.load(state_path, map_location="cpu"))
    return m.to(dtype=torch.float64).eval()


@torch.no_grad()
def scores_f64(model: torch.nn.Module, items: Sequence[Item], batch_size: int = 64) -> List[List[float]]:
    """Option scores of each item in float64 on the CPU (items carry fp16/32 states; converted here)."""
    model = model.to(dtype=torch.float64).eval()
    out: List[List[float]] = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        tb = collate(batch, "cpu")
        tb = {k: (v.to(torch.float64) if v.is_floating_point() else v) for k, v in tb.items()}
        s = model(tb["ctx"], tb["ctx_mask"], tb["cands"], tb["cand_mask"], polarity=tb["polarity"])
        out.extend([[float(x) for x in row] for row in s])
    return out


def bag(text: str) -> Counter:
    return Counter(t.lower() for t in word_tokenize(text)[0])


def insertion_pairs(rows: Sequence[NegRow]) -> List[Tuple[NegRow, NegRow]]:
    """(O, X), X in {N, S} of the same scenario, with Bag(O) a proper sub-multiset of Bag(X) on `<context> <question>`."""
    by = {}
    for r in rows:
        by.setdefault(r.passage_id, {})[r.arm] = r
    out = []
    for v in by.values():
        if "O" not in v:
            continue
        bo = bag(v["O"].text)
        for arm in ("N", "S"):
            if arm in v:
                bx = bag(v[arm].text)
                if not (bo - bx) and sum(bx.values()) > sum(bo.values()):
                    out.append((v["O"], v[arm]))
    return out


def permutation_pairs(rows: Sequence[NegRow]) -> List[Tuple[NegRow, NegRow]]:
    """(N, S) of every scenario with Bag(N) = Bag(S) (all NegPairs-Perm scenarios by construction; asserted)."""
    by = {}
    for r in rows:
        by.setdefault(r.passage_id, {})[r.arm] = r
    out = []
    for sid, v in sorted(by.items()):
        if "N" in v and "S" in v and bag(v["N"].text) == bag(v["S"].text):
            out.append((v["N"], v["S"]))
    return out


def check_r5(model, item_of, pairs: Sequence[Tuple[NegRow, NegRow]], tol: float = TOL) -> dict:
    """No option score under the longer context X may be lower than under O by more than tol."""
    a = scores_f64(model, [item_of(o) for o, _ in pairs])
    b = scores_f64(model, [item_of(x) for _, x in pairs])
    n = v = 0
    worst = float("-inf")
    for so, sx in zip(a, b):
        for yo, yx in zip(so, sx):
            n += 1
            d = yo - yx                       # positive = the longer context scored lower
            worst = max(worst, d)
            v += d > tol
    return {"pairs": len(pairs), "n_scores": n, "violations": v, "largest_decrease": (worst if n else None), "tolerance": tol}


def check_r6(model, item_of, pairs: Sequence[Tuple[NegRow, NegRow]], tol: float = TOL) -> dict:
    """Every option scores the same under N and S within tol; decisions agree where both margins exceed tol."""
    a = scores_f64(model, [item_of(n) for n, _ in pairs])
    b = scores_f64(model, [item_of(s) for _, s in pairs])
    out = {"pairs": len(pairs), "n_scores": 0, "score_violations": 0, "largest_abs_diff": 0.0, "near_ties": 0,
           "decision_pairs_checked": 0, "decision_disagreements": 0, "flip_O_N_minus_flip_O_S": None, "tolerance": tol}
    for sn, ss in zip(a, b):
        for yn, ys in zip(sn, ss):
            out["n_scores"] += 1
            d = abs(yn - ys)
            out["largest_abs_diff"] = max(out["largest_abs_diff"], d)
            out["score_violations"] += d > tol
        mn, ms = abs(sn[0] - sn[1]), abs(ss[0] - ss[1])
        if mn <= tol or ms <= tol:
            out["near_ties"] += 1
            continue
        out["decision_pairs_checked"] += 1
        out["decision_disagreements"] += (sn[0] > sn[1]) != (ss[0] > ss[1])
    return out


def nested_instances(instances: Sequence[dict]) -> List[dict]:
    """NevIR instances whose documents are nested as lower-cased E0 token bags; adds `contained`."""
    out = []
    for ins in instances:
        b1, b2 = bag(ins["doc1"]), bag(ins["doc2"])
        if b1 == b2:
            out.append({**ins, "contained": "equal"})
        elif not (b1 - b2):
            out.append({**ins, "contained": "doc1"})
        elif not (b2 - b1):
            out.append({**ins, "contained": "doc2"})
    return out


# ------------------------------------------------------------------------------------ pilot checks (section 11)
def shuffled_item(it: Item, rng: random.Random) -> Item:
    idx = list(range(it.ctx.shape[0]))
    rng.shuffle(idx)
    return Item(row=it.row, ctx=it.ctx[idx], cands=it.cands, polarity=it.polarity[idx], label=it.label, unseen_cue=it.unseen_cue)


def pilot_exact_checks(dim: int, dev_items: Sequence[Item], dev_rows: Sequence[NegRow], item_of, seeds: Sequence[int],
                       tol: float = TOL, item_of_scope=None) -> dict:
    """Random initializations of every unmasked rule: identical scores for every dev context and a random permutation
    of its tokens (Proposition A); no decrease on dev insertion pairs for max, lse and m1 (Proposition B); the
    violations of mean and of signed_max with the resolver's masks (`item_of_scope`) on insertion pairs are reported
    (expected)."""
    pairs = insertion_pairs(dev_rows)
    out = {"dev_items": len(dev_items), "dev_insertion_pairs": len(pairs), "permutation": {}, "insertion": {}}
    for kind in SYMMETRIC:
        worst, viol, n = 0.0, 0, 0
        for seed in seeds:
            torch.manual_seed(seed)
            m = make_module(kind, dim).to(dtype=torch.float64).eval()
            rng = random.Random(seed)
            a = scores_f64(m, dev_items)
            b = scores_f64(m, [shuffled_item(it, rng) for it in dev_items])
            for sa, sb in zip(a, b):
                for x, y in zip(sa, sb):
                    n += 1
                    worst = max(worst, abs(x - y))
                    viol += abs(x - y) > tol
        out["permutation"][kind] = {"n_scores": n, "violations": viol, "largest_abs_diff": worst}
    for kind in ("max", "lse", "m1", "mean", "signed_max"):
        tot = {"pairs": len(pairs), "n_scores": 0, "violations": 0, "largest_decrease": float("-inf")}
        for seed in seeds:
            torch.manual_seed(seed)
            m = make_module(kind, dim).to(dtype=torch.float64).eval()
            r = check_r5(m, item_of_scope if (kind == "signed_max" and item_of_scope is not None) else item_of, pairs, tol)
            tot["n_scores"] += r["n_scores"]
            tot["violations"] += r["violations"]
            if r["largest_decrease"] is not None:
                tot["largest_decrease"] = max(tot["largest_decrease"], r["largest_decrease"])
        tot["expected_zero"] = kind in MONOTONE
        if tot["largest_decrease"] == float("-inf"):
            tot["largest_decrease"] = None
        out["insertion"][kind] = tot
    out["passed"] = (all(v["violations"] == 0 for v in out["permutation"].values())
                     and all(v["violations"] == 0 for k, v in out["insertion"].items() if k in MONOTONE))
    return out
