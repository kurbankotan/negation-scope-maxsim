"""Aggregate archived runs into the manifest's tables and evaluate the prespecified rules R1-R5.

Run directory layout (written by train.train_run):
    <root>/<encoder>/<aggregator>/seed<seed>/{run_summary.json, dev_predictions.jsonl, test_predictions.jsonl}
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .metrics import full_report, accuracy, accuracy_by_edit, group_consistency, pair_flip_rate
from .stats import paired_t_interval, cluster_bootstrap, mcnemar_exact, discordant_counts

RULE_THRESHOLDS = {  # manifest section 8 (accepted by the author on 2026-10-02)
    "R1_max_flip_rate_below": 25.0,
    "R1_paraphrase_flip_rate_below": 10.0,
    "R2_remedy_gain_points": 10.0,
    "R2_no_regression_lower_bound": -2.0,
    "R3_scope_gain_points": 5.0,
}


def read_records(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_runs(root: str | Path, split: str = "test") -> Dict[Tuple[str, str, int], dict]:
    """(encoder, aggregator, seed) -> {'summary': ..., 'records': [...]}"""
    root = Path(root)
    runs = {}
    for summ in root.glob("*/*/seed*/run_summary.json"):
        seed_dir = summ.parent
        enc, agg = seed_dir.parent.parent.name, seed_dir.parent.name
        seed = int(seed_dir.name.replace("seed", ""))
        pred = seed_dir / f"{split}_predictions.jsonl"
        if not pred.exists():
            continue
        runs[(enc, agg, seed)] = {"summary": json.loads(summ.read_text()), "records": read_records(pred)}
    return runs


def per_run_metrics(runs: Dict[Tuple[str, str, int], dict]) -> List[dict]:
    rows = []
    for (enc, agg, seed), r in sorted(runs.items()):
        rep = full_report(r["records"])
        rows.append({
            "encoder": enc, "aggregator": agg, "seed": seed,
            "selected_epoch": r["summary"].get("selected_epoch"),
            "params_total": r["summary"].get("parameter_counts", {}).get("total"),
            "acc": rep["accuracy"], "acc_yes_no": rep["accuracy_yes_no_only"],
            **{f"acc_{k}": v for k, v in rep["accuracy_by_edit"].items()},
            "consistency": rep["group_consistency"]["consistency"],
            "flip_paraphrase": rep["flip_original_vs_paraphrase_all"]["flip_rate"],
            "flip_scope": rep["flip_original_vs_scope_label_flipping"]["flip_rate"],
            "flip_affirmative": rep["flip_original_vs_affirmative_label_flipping"]["flip_rate"],
            "polarity_cost": rep["costs"]["polarity_cost"], "scope_cost": rep["costs"]["scope_cost"],
            "acc_seen_cue": rep["unseen_cue"]["acc_seen"], "acc_unseen_cue": rep["unseen_cue"]["acc_unseen"],
            "unseen_cue_gap": rep["unseen_cue"]["gap"],
        })
    return rows


def seed_mean_table(rows: Sequence[dict]) -> List[dict]:
    by = defaultdict(list)
    for r in rows:
        by[(r["encoder"], r["aggregator"])].append(r)
    out = []
    metrics = [k for k in rows[0] if k not in ("encoder", "aggregator", "seed", "selected_epoch", "params_total")] if rows else []
    for (enc, agg), rs in sorted(by.items()):
        rec = {"encoder": enc, "aggregator": agg, "n_seeds": len(rs)}
        for m in metrics:
            vals = [r[m] for r in rs if r[m] is not None]
            rec[m] = (sum(vals) / len(vals)) if vals else None
            rec[m + "_sd"] = (paired_t_interval(vals)["sd"] if len(vals) > 1 else None)
        out.append(rec)
    return out


def paired_contrast(rows: Sequence[dict], encoder: str, agg_a: str, agg_b: str, metric: str) -> dict:
    """Seed-paired difference metric(agg_a) - metric(agg_b) with a 95% t interval."""
    a = {r["seed"]: r[metric] for r in rows if r["encoder"] == encoder and r["aggregator"] == agg_a}
    b = {r["seed"]: r[metric] for r in rows if r["encoder"] == encoder and r["aggregator"] == agg_b}
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds if a[s] is not None and b[s] is not None]
    ci = paired_t_interval(diffs) if diffs else {"n": 0}
    return {"encoder": encoder, "contrast": f"{agg_a} - {agg_b}", "metric": metric, "seeds": seeds, "diffs": diffs, **ci}


def cluster_interval_for_contrast(runs: Dict[Tuple[str, str, int], dict], encoder: str, agg_a: str, agg_b: str,
                                  metric: str = "acc_affirmative", n_boot: int = 10000, seed: int = 20261002) -> dict:
    """Passage-cluster percentile bootstrap of the seed-mean accuracy difference (agg_a - agg_b).

    All seeds and edits of a passage are resampled together. Every seed scores the same rows, so the
    seed-mean accuracy of a resample equals the pooled accuracy over seeds; the computation is therefore
    vectorized over per-passage counts. `metric` selects the rows: acc (all), acc_original, acc_paraphrase,
    acc_scope, acc_affirmative.
    """
    import numpy as np

    edit_of = {"acc": None, "acc_original": 0, "acc_paraphrase": 1, "acc_scope": 2, "acc_affirmative": 3}
    if metric not in edit_of:
        raise ValueError(f"cluster bootstrap supports {sorted(edit_of)}; got {metric}")
    want = edit_of[metric]
    seeds = sorted(set(s for (e, g, s) in runs if e == encoder and g == agg_a) & set(s for (e, g, s) in runs if e == encoder and g == agg_b))
    if not seeds:
        return {"point": None, "low": None, "high": None, "n_clusters": 0, "n_boot": n_boot, "seed": seed, "seeds": []}
    passages = sorted({r["passage_id"] for s_ in seeds for g in (agg_a, agg_b) for r in runs[(encoder, g, s_)]["records"]})
    index = {p: k for k, p in enumerate(passages)}
    C = len(passages)
    correct = {g: np.zeros(C) for g in (agg_a, agg_b)}
    total = {g: np.zeros(C) for g in (agg_a, agg_b)}
    for g in (agg_a, agg_b):
        for s_ in seeds:
            for r in runs[(encoder, g, s_)]["records"]:
                if want is not None and int(r["edit_id"]) != want:
                    continue
                k = index[r["passage_id"]]
                total[g][k] += 1
                correct[g][k] += (r["pred"] == r["label"])
    point = 100.0 * (correct[agg_a].sum() / total[agg_a].sum() - correct[agg_b].sum() / total[agg_b].sum())
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    chunk = 1000
    for start_ in range(0, n_boot, chunk):
        idx = rng.integers(0, C, size=(min(chunk, n_boot - start_), C))
        ta, tb = total[agg_a][idx].sum(1), total[agg_b][idx].sum(1)
        ca, cb = correct[agg_a][idx].sum(1), correct[agg_b][idx].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            reps[start_:start_ + idx.shape[0]] = 100.0 * (ca / ta - cb / tb)
    reps = reps[np.isfinite(reps)]
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"point": float(point), "low": float(lo), "high": float(hi), "n_clusters": C, "n_boot": n_boot,
            "n_valid_replicates": int(reps.size), "seed": seed, "seeds": seeds}


def group_consistency_pooled(recs: Sequence[dict]) -> Optional[float]:
    """Consistency computed per seed then averaged (records carry `_seed`)."""
    by_seed = defaultdict(list)
    for r in recs:
        by_seed[r.get("_seed", 0)].append(r)
    vals = [group_consistency(v)["consistency"] for v in by_seed.values()]
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def evaluate_rules(rows: Sequence[dict], runs: Dict[Tuple[str, str, int], dict], contextual_encoders: Sequence[str],
                   thresholds: dict = RULE_THRESHOLDS, n_boot: int = 10000) -> dict:
    """Evaluate R1-R3 on point estimates (manifest section 8) and attach intervals."""
    out: Dict[str, dict] = {}
    # R1: polarity blindness of MAX for every contextual encoder
    r1 = {}
    for enc in contextual_encoders:
        mx = [r for r in rows if r["encoder"] == enc and r["aggregator"] == "max"]
        flips = [r["flip_affirmative"] for r in mx]
        para = [r["flip_paraphrase"] for r in mx]
        r1[enc] = {"flip_affirmative_by_seed": flips, "flip_paraphrase_by_seed": para,
                   "met": bool(flips) and all(f is not None and f < thresholds["R1_max_flip_rate_below"] for f in flips)
                   and all(p is not None and p < thresholds["R1_paraphrase_flip_rate_below"] for p in para)}
    out["R1"] = r1
    # R2: remedy (signed_max vs max) on affirmative accuracy and consistency; no regression on originals
    r2 = {}
    for enc in contextual_encoders:
        aff = paired_contrast(rows, enc, "signed_max", "max", "acc_affirmative")
        con = paired_contrast(rows, enc, "signed_max", "max", "consistency")
        orig = paired_contrast(rows, enc, "signed_max", "max", "acc_original")
        aff_cl = cluster_interval_for_contrast(runs, enc, "signed_max", "max", "acc_affirmative", n_boot=n_boot)
        met = (aff.get("n", 0) > 0 and aff["mean"] >= thresholds["R2_remedy_gain_points"] and aff["positive_count"] == aff["n"] and aff["low"] > 0
               and con.get("n", 0) > 0 and con["mean"] >= thresholds["R2_remedy_gain_points"] and con["positive_count"] == con["n"] and con["low"] > 0
               and orig.get("n", 0) > 0 and orig["low"] >= thresholds["R2_no_regression_lower_bound"])
        r2[enc] = {"affirmative": aff, "affirmative_cluster_ci": aff_cl, "consistency": con, "original_no_regression": orig, "met": bool(met)}
    out["R2"] = r2
    # R3: scope specificity (signed_max vs mean) on scope-edit accuracy
    r3 = {}
    for enc in contextual_encoders:
        sc = paired_contrast(rows, enc, "signed_max", "mean", "acc_scope")
        met = sc.get("n", 0) > 0 and sc["mean"] >= thresholds["R3_scope_gain_points"] and sc["positive_count"] == sc["n"] and sc["low"] > 0
        r3[enc] = {"scope": sc, "met": bool(met)}
    out["R3"] = r3
    # R4 descriptive: gate vs parser scope
    r4 = {}
    for enc in contextual_encoders:
        r4[enc] = {m: paired_contrast(rows, enc, "signed_max_gate", "signed_max", m) for m in ("acc", "acc_affirmative", "acc_scope", "consistency")}
    out["R4_descriptive"] = r4
    return out


def mcnemar_table(runs: Dict[Tuple[str, str, int], dict], encoder: str, agg_a: str, agg_b: str) -> List[dict]:
    out = []
    seeds = sorted({s for (e, g, s) in runs if e == encoder and g in (agg_a, agg_b)})
    for s in seeds:
        ka, kb = (encoder, agg_a, s), (encoder, agg_b, s)
        if ka in runs and kb in runs:
            b, c = discordant_counts(runs[ka]["records"], runs[kb]["records"])
            out.append({"encoder": encoder, "seed": s, "contrast": f"{agg_a} vs {agg_b}", **mcnemar_exact(b, c)})
    return out


def markdown_table(rows: Sequence[dict], columns: Sequence[str], digits: int = 2) -> str:
    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.{digits}f}"
        return str(v)
    head = "| " + " | ".join(columns) + " |\n|" + "|".join("---" for _ in columns) + "|\n"
    body = "".join("| " + " | ".join(fmt(r.get(c)) for c in columns) + " |\n" for r in rows)
    return head + body
