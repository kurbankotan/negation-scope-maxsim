"""Estimands M1-M7 computed from archived per-row predictions.

A prediction record is a dict with at least:
  sample_id, passage_id, question_id, edit_id, label, pred (label string), probs (list of 3 floats),
  cue, unseen_cue (bool)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .condaqa import LABELS, EDIT_NAMES


def _acc(records: Sequence[dict]) -> Optional[float]:
    if not records:
        return None
    return 100.0 * sum(r["pred"] == r["label"] for r in records) / len(records)


def accuracy(records: Sequence[dict]) -> Optional[float]:
    return _acc(records)


def accuracy_by_edit(records: Sequence[dict]) -> Dict[str, Optional[float]]:
    by = defaultdict(list)
    for r in records:
        by[int(r["edit_id"])].append(r)
    return {EDIT_NAMES[e]: _acc(by[e]) for e in sorted(by)}


def accuracy_yes_no_only(records: Sequence[dict]) -> Optional[float]:
    return _acc([r for r in records if r["label"] in ("YES", "NO")])


def _groups(records: Sequence[dict]) -> Dict[Tuple[str, str], Dict[int, dict]]:
    g: Dict[Tuple[str, str], Dict[int, dict]] = defaultdict(dict)
    for r in records:
        g[(r["passage_id"], r["question_id"])].setdefault(int(r["edit_id"]), r)
    return g


def group_consistency(records: Sequence[dict]) -> Dict[str, Optional[float]]:
    """Fraction of complete 4-edit groups in which every edit is answered correctly (CondaQA metric)."""
    g = _groups(records)
    complete = [v for v in g.values() if set(v) == {0, 1, 2, 3}]
    if not complete:
        return {"n_groups": 0, "consistency": None}
    ok = sum(all(v[e]["pred"] == v[e]["label"] for e in (0, 1, 2, 3)) for v in complete)
    return {"n_groups": len(complete), "consistency": 100.0 * ok / len(complete)}


def pair_flip_rate(records: Sequence[dict], edit_id: int, label_flipping: Optional[bool] = None) -> Dict[str, Optional[float]]:
    """M3: fraction of (original, edit) pairs whose predictions differ; also both-correct rate."""
    g = _groups(records)
    n = flips = both = 0
    for v in g.values():
        if 0 in v and edit_id in v:
            a, b = v[0], v[edit_id]
            lf = a["label"] != b["label"]
            if label_flipping is not None and lf != label_flipping:
                continue
            n += 1
            flips += a["pred"] != b["pred"]
            both += (a["pred"] == a["label"]) and (b["pred"] == b["label"])
    return {"n_pairs": n, "flip_rate": (100.0 * flips / n if n else None), "both_correct": (100.0 * both / n if n else None)}


def polarity_and_scope_cost(records: Sequence[dict]) -> Dict[str, Optional[float]]:
    """M4: acc(paraphrase) - acc(affirmative) and acc(paraphrase) - acc(scope)."""
    a = accuracy_by_edit(records)
    p, s, f = a.get("paraphrase"), a.get("scope"), a.get("affirmative")
    return {
        "polarity_cost": (p - f) if (p is not None and f is not None) else None,
        "scope_cost": (p - s) if (p is not None and s is not None) else None,
        "edit_cost": (a.get("original") - p) if (a.get("original") is not None and p is not None) else None,
    }


def unseen_cue_gap(records: Sequence[dict]) -> Dict[str, Optional[float]]:
    seen = [r for r in records if not r.get("unseen_cue")]
    unseen = [r for r in records if r.get("unseen_cue")]
    a_s, a_u = _acc(seen), _acc(unseen)
    return {"n_seen": len(seen), "n_unseen": len(unseen), "acc_seen": a_s, "acc_unseen": a_u,
            "gap": (a_s - a_u) if (a_s is not None and a_u is not None) else None}


def margin_profile(records: Sequence[dict]) -> Dict[str, dict]:
    """M6: softmax margin (top-1 minus top-2 probability) for right and wrong predictions, by edit type."""
    out: Dict[str, dict] = {}
    by = defaultdict(lambda: {"right": [], "wrong": []})
    for r in records:
        probs = sorted(r["probs"], reverse=True)
        m = probs[0] - (probs[1] if len(probs) > 1 else 0.0)
        by[EDIT_NAMES[int(r["edit_id"])]]["right" if r["pred"] == r["label"] else "wrong"].append(m)
    for edit, d in by.items():
        out[edit] = {}
        for k, xs in d.items():
            out[edit][k] = {"n": len(xs), "mean_margin": (sum(xs) / len(xs) if xs else None),
                            "frac_margin_above_0.9": (sum(x > 0.9 for x in xs) / len(xs) if xs else None)}
    return out


def proposition1_violations(score_pairs: Iterable[Tuple[Sequence[float], Sequence[float]]], tol: float = 1e-5) -> dict:
    """M7: for (scores_full, scores_cue_deleted) pairs, count candidates whose score INCREASED after deleting
    the cue. Under a context-independent encoder with a token-addition-monotone aggregator this must be 0."""
    n = viol = 0
    worst = 0.0
    for full, deleted in score_pairs:
        for a, b in zip(full, deleted):
            n += 1
            d = float(b) - float(a)
            if d > tol:
                viol += 1
            worst = max(worst, d)
    return {"n_scores": n, "violations": viol, "violation_rate": (viol / n if n else None), "max_increase": worst}


def full_report(records: Sequence[dict]) -> dict:
    return {
        "n": len(records),
        "accuracy": accuracy(records),
        "accuracy_yes_no_only": accuracy_yes_no_only(records),
        "accuracy_by_edit": accuracy_by_edit(records),
        "group_consistency": group_consistency(records),
        "flip_original_vs_paraphrase_all": pair_flip_rate(records, 1),
        "flip_original_vs_scope_label_flipping": pair_flip_rate(records, 2, label_flipping=True),
        "flip_original_vs_affirmative_label_flipping": pair_flip_rate(records, 3, label_flipping=True),
        "costs": polarity_and_scope_cost(records),
        "unseen_cue": unseen_cue_gap(records),
        "margins": margin_profile(records),
    }
