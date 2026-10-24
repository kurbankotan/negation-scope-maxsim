"""Analysis of manifest v3.0: permutation estimands, the dev gate, the statistical rules R2, R3 and R8 with their
four conditions (threshold, seed count, seed-level t interval and scenario-cluster bootstrap at level 1 - 0.05/k),
descriptive blocks, and tables. The v2.0 functions of `np_analysis` are reused for the standard metrics."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .negpairs import ARMS, NegRow, report, scenario_flips
from .np_analysis import _seed_mean, contrast, markdown, per_run, seed_means
from .stats import paired_t_interval

SETS = ("v1test", "fresh", "perm")
STATIC = ("E0-hash", "E3-glove")
PERM_METRICS = ("flip_O_N", "flip_O_S", "flip_O_P", "SS", "joint_NS_acc", "acc_O", "acc_P", "acc_N", "acc_S", "n_scenarios")


def _rate(vals: Sequence[int]) -> Optional[float]:
    return (100.0 * sum(vals) / len(vals)) if vals else None


def perm_block(records: Sequence[dict], rows_by_id: Dict[str, NegRow]) -> dict:
    """Flip rates, scope sensitivity, joint (N, S) accuracy and arm accuracies over the scenarios in `records`."""
    scen = defaultdict(dict)
    for r in records:
        row = rows_by_id[r["sample_id"]]
        scen[row.passage_id][row.arm] = r
    full = [v for v in scen.values() if all(a in v for a in ARMS)]
    out = {"n_scenarios": len(full)}
    for arm in ("N", "S", "P"):
        out[f"flip_O_{arm}"] = _rate([int(v[arm]["pred"] != v["O"]["pred"]) for v in full])
    out["SS"] = (out["flip_O_N"] - out["flip_O_S"]) if full else None
    out["joint_NS_acc"] = _rate([int(v["N"]["pred"] == v["N"]["label"] and v["S"]["pred"] == v["S"]["label"]) for v in full])
    for arm in ARMS:
        out[f"acc_{arm}"] = _rate([int(v[arm]["pred"] == v[arm]["label"]) for v in full])
    return out


def perm_report(records: Sequence[dict], rows_by_id: Dict[str, NegRow]) -> dict:
    """Blocks for all / seen / unseen / family, each split by clause order, with the order-balanced estimand
    SS_perm_bal = (SS_first + SS_second) / 2 (manifest v3.0, section 7.2)."""
    def sel(pred):
        return [r for r in records if pred(rows_by_id[r["sample_id"]])]
    groups = {"all": lambda row: True, "seen": lambda row: row.seen_family, "unseen": lambda row: not row.seen_family}
    for fam in sorted({rows_by_id[r["sample_id"]].family for r in records}):
        groups[f"family.{fam}"] = (lambda f: (lambda row: row.family == f))(fam)
    out = {}
    for name, pred in groups.items():
        recs = sel(pred)
        blk = {"pooled": perm_block(recs, rows_by_id)}
        for order, flag in (("first", True), ("second", False)):
            blk[order] = perm_block([r for r in recs if bool(rows_by_id[r["sample_id"]].perm["decisive_first"]) == flag], rows_by_id)
        a, b = blk["first"]["SS"], blk["second"]["SS"]
        blk["SS_bal"] = ((a + b) / 2) if (a is not None and b is not None) else None
        out[name] = blk
    return out


def flatten_perm(rep: dict) -> dict:
    flat = {}
    for name, blk in rep.items():
        for sub in ("pooled", "first", "second"):
            for m in PERM_METRICS:
                flat[f"{name}.{sub}.{m}"] = blk[sub].get(m)
        flat[f"{name}.SS_bal"] = blk["SS_bal"]
    return flat


def filter_runs(runs: Dict[Tuple[str, str, int], List[dict]], rows_by_id: Dict[str, NegRow],
                scenario_filter: Optional[Dict[str, Iterable[str]]]) -> Dict[Tuple[str, str, int], List[dict]]:
    """Keep, per encoder, only the records of the listed scenarios (Perm: equal token multisets under the tokenizer)."""
    if not scenario_filter:
        return runs
    out = {}
    for (enc, agg, seed), recs in runs.items():
        keep = scenario_filter.get(enc)
        if keep is None:
            out[(enc, agg, seed)] = recs
        else:
            keep = set(keep)
            out[(enc, agg, seed)] = [r for r in recs if rows_by_id[r["sample_id"]].passage_id in keep]
    return out


def per_run_perm(runs, rows_by_id: Dict[str, NegRow]) -> List[dict]:
    rows = []
    for (enc, agg, seed), recs in sorted(runs.items()):
        flat = {"encoder": enc, "aggregator": agg, "seed": seed}
        flat.update(flatten_perm(perm_report(recs, rows_by_id)))
        rows.append(flat)
    return rows


# ------------------------------------------------------------------------------------------- bootstrap (level 1 - a)
def _flip_table(runs, rows_by_id, encoder, aggs, seeds, kind, ok):
    per, rows_of = {}, {}
    for a in aggs:
        for sd in seeds:
            recs = runs[(encoder, a, sd)]
            fl = scenario_flips(recs, rows_by_id)
            row_of = {rows_by_id[r["sample_id"]].passage_id: rows_by_id[r["sample_id"]] for r in recs}
            for sid, v in fl.items():
                if not ok(sid, row_of[sid]) or "N" not in v or "S" not in v:
                    continue
                val = (v["N"] - v["S"]) if kind == "SS" else v[kind]
                d = per.setdefault(sid, {x: [0.0, 0] for x in aggs})
                d[a][0] += val
                d[a][1] += 1
                rows_of[sid] = row_of[sid]
    return per, rows_of


def cluster_bootstrap_flip(runs, rows_by_id: Dict[str, NegRow], encoder: str, agg_a: str, agg_b: Optional[str] = None,
                           kind: str = "SS", subset: str = "unseen", scenario_ids: Optional[Sequence[str]] = None,
                           n_boot: int = 10000, seed: int = 20261003, conf: float = 0.95,
                           by_order: bool = False) -> dict:
    """Scenario-cluster percentile bootstrap at confidence `conf` of a flip estimand (pooled over the shared seeds):
    kind "SS" (flip_N - flip_S), "S" or "N"; the difference agg_a - agg_b when agg_b is given. With by_order=True the
    scenarios are resampled within each clause order and the statistic is the mean over the two orders (SS_perm_bal)."""
    import numpy as np

    aggs = [agg_a] + ([agg_b] if agg_b else [])
    seed_sets = [set(s for (e, g, s) in runs if e == encoder and g == a) for a in aggs]
    seeds = sorted(set.intersection(*seed_sets)) if seed_sets else []
    if not seeds:
        return {"point": None, "low": None, "high": None, "n_clusters": 0}
    keep = set(scenario_ids) if scenario_ids is not None else None
    ok = lambda sid, row: (subset == "all" or (subset == "seen") == row.seen_family) and (keep is None or sid in keep)
    per, row_of = _flip_table(runs, rows_by_id, encoder, aggs, seeds, kind, ok)
    if not per:
        return {"point": None, "low": None, "high": None, "n_clusters": 0}
    strata = [sorted(per)]
    if by_order:
        strata = [sorted(s for s in per if bool(row_of[s].perm["decisive_first"]) == flag) for flag in (True, False)]
        if any(not s for s in strata):
            return {"point": None, "low": None, "high": None, "n_clusters": [len(s) for s in strata]}
    arrays = [({a: np.array([per[sid][a][0] for sid in scen]) for a in aggs},
               {a: np.array([per[sid][a][1] for sid in scen]) for a in aggs}) for scen in strata]
    rng = np.random.default_rng(seed)

    def stat(idx_list=None):
        vals = []
        for k, (num, den) in enumerate(arrays):
            part = []
            for a in aggs:
                if idx_list is None:
                    part.append(100.0 * num[a].sum() / max(1, den[a].sum()))
                else:
                    ii = idx_list[k]
                    part.append(100.0 * num[a][ii].sum(axis=-1) / np.maximum(1, den[a][ii].sum(axis=-1)))
            vals.append(part[0] - part[1] if agg_b else part[0])
        return sum(vals) / len(vals)

    point = float(stat())
    reps = []
    for start in range(0, n_boot, 1000):
        m = min(1000, n_boot - start)
        reps.append(stat([rng.integers(0, len(s), size=(m, len(s))) for s in strata]))
    reps = np.concatenate(reps)
    lo, hi = np.percentile(reps, [100 * (1 - conf) / 2, 100 * (1 + conf) / 2])
    return {"point": point, "low": float(lo), "high": float(hi), "n_clusters": [len(s) for s in strata], "n_boot": n_boot,
            "seeds": seeds, "kind": kind, "subset": subset, "conf": conf, "by_order": by_order,
            "contrast": f"{agg_a} - {agg_b}" if agg_b else agg_a}


# ------------------------------------------------------------------------------------------- gate and rules
def dev_gate(dev_runs, rows_by_id_dev: Dict[str, NegRow], encoders: Sequence[str], contextual: Sequence[str], thr: dict) -> dict:
    """G (per encoder): A1 seed-mean SS on the v1 dev split >= gate_SS_seen_dev. Fixes the encoders evaluated by R2,
    R3 and R8 and the family sizes k (manifest v3.0, 8.3)."""
    out = {"threshold": thr["gate_SS_seen_dev"], "encoders": {}}
    for enc in encoders:
        vals = {s: report(recs, rows_by_id_dev)["all"]["SS"] for (e, a, s), recs in dev_runs.items() if e == enc and a == "max"}
        seeds = sorted(vals)
        m = (sum(vals[s] for s in seeds) / len(seeds)) if seeds else None
        out["encoders"][enc] = {"A1_dev_SS_by_seed": [vals[s] for s in seeds], "seeds": seeds, "A1_seed_mean_dev_SS": m,
                                "passed": bool(m is not None and m >= thr["gate_SS_seen_dev"])}
    passed = [e for e in encoders if out["encoders"][e]["passed"]]
    out["passed"] = passed
    out["k_R2_R3"] = len(passed)
    out["k_R8"] = len([e for e in passed if e in contextual])
    return out


def _conditions(vals_by_seed: Dict[int, float], threshold: float, k: int, thr: dict, boot: dict) -> dict:
    """(a) seed mean >= threshold; (b) positive in >= min_positive_seeds of n_seeds_required seeds; (c) two-sided t
    interval at level 1 - 0.05/k above zero; (d) cluster bootstrap interval at the same level above zero."""
    seeds = sorted(vals_by_seed)
    vals = [vals_by_seed[s] for s in seeds]
    conf = 1 - thr["family_alpha"] / max(1, k)
    ti = paired_t_interval(vals, conf=conf) if vals else {"n": 0}
    n_req = thr["n_seeds_required"]
    cond = {"a_mean_ge_threshold": bool(vals and ti["mean"] >= threshold),
            "b_positive_seeds": bool(len(vals) == n_req and ti["positive_count"] >= thr["min_positive_seeds"]),
            "c_t_interval_above_zero": bool(len(vals) >= 2 and ti["low"] > 0),
            "d_cluster_interval_above_zero": bool(boot.get("low") is not None and boot["low"] > 0)}
    return {"seeds": seeds, "values": vals, "t_interval": ti, "cluster_interval": boot, "conf": conf, "k": k,
            "threshold": threshold, "conditions": cond, "met": all(cond.values())}


def _by_seed(rows, enc, agg, key) -> Dict[int, float]:
    return {r["seed"]: r[key] for r in rows if r["encoder"] == enc and r["aggregator"] == agg and r.get(key) is not None}


def _diff_by_seed(rows, enc, agg_a, agg_b, key) -> Dict[int, float]:
    a, b = _by_seed(rows, enc, agg_a, key), _by_seed(rows, enc, agg_b, key)
    return {s: a[s] - b[s] for s in sorted(set(a) & set(b))}


def evaluate_rules_v3(gate: dict, rows_fresh: Sequence[dict], runs_fresh, fresh_by_id: Dict[str, NegRow],
                      rows_perm: Sequence[dict], runs_perm, perm_by_id: Dict[str, NegRow], encoders: Sequence[str],
                      contextual: Sequence[str], thr: dict, s_same_fresh: Sequence[str], n_unseen_fresh: int,
                      perm_seen_counts: Dict[str, Dict[str, int]], n_boot: int = 10000) -> dict:
    """R2, R3 (NegPairs-Fresh) and R8 (NegPairs-Perm) of manifest v3.0, section 8.3, for the encoders that pass G.
    `perm_seen_counts[enc]` = {"first": n, "second": n} seen-family Perm scenarios per clause order after the
    tokenizer restriction. R1 is reported descriptively."""
    out = {"thresholds": thr, "gate": gate, "R2": {}, "R3": {}, "R8": {}, "R1_descriptive": {}}
    k23, k8 = max(1, gate["k_R2_R3"]), max(1, gate["k_R8"])
    conf23, conf8 = 1 - thr["family_alpha"] / k23, 1 - thr["family_alpha"] / k8
    n_same = len(s_same_fresh)
    for enc in encoders:
        passed = enc in gate["passed"]
        # R2: A4 - A1 unseen-family SS on Fresh
        ev = passed and n_unseen_fresh >= thr["min_scenarios_R2"]
        d = _diff_by_seed(rows_fresh, enc, "signed_max", "max", "unseen.SS")
        boot = cluster_bootstrap_flip(runs_fresh, fresh_by_id, enc, "signed_max", "max", "SS", "unseen", None, n_boot, conf=conf23) if ev else {}
        res = _conditions(d, thr["R2_min_gain_unseen_SS"], k23, thr, boot) if d else {"met": False}
        out["R2"][enc] = {"evaluated": bool(ev), "gate_passed": passed, "n_unseen_fresh_scenarios": n_unseen_fresh,
                          "contrast": "A4 - A1, unseen-family SS, NegPairs-Fresh", **res, "met": bool(ev and res.get("met"))}
        # R3: A6 - A4 flip_O_S on S-same Fresh scenarios
        ev = passed and n_same >= thr["min_scenarios_R3"]
        d = _diff_by_seed(rows_fresh, enc, "signed_max_sentence", "signed_max", "S_same_scen.flip_O_S")
        boot = cluster_bootstrap_flip(runs_fresh, fresh_by_id, enc, "signed_max_sentence", "signed_max", "S", "all", s_same_fresh, n_boot, conf=conf23) if ev else {}
        res = _conditions(d, thr["R3_min_gain_flipS"], k23, thr, boot) if d else {"met": False}
        out["R3"][enc] = {"evaluated": bool(ev), "gate_passed": passed, "n_S_same_scenarios": n_same,
                          "contrast": "A6 - A4, flip_O_S on S-same, NegPairs-Fresh", **res, "met": bool(ev and res.get("met"))}
        # R8: A1 SS_perm_bal on seen-family Perm scenarios (contextual encoders)
        if enc in contextual:
            cnt = perm_seen_counts.get(enc, {})
            ev = passed and all(cnt.get(o, 0) >= thr["min_scenarios_R8_per_order"] for o in ("first", "second"))
            v = _by_seed(rows_perm, enc, "max", "seen.SS_bal")
            boot = cluster_bootstrap_flip(runs_perm, perm_by_id, enc, "max", None, "SS", "seen", None, n_boot, conf=conf8, by_order=True) if ev else {}
            res = _conditions(v, thr["R8_min_SS_perm_bal_seen"], k8, thr, boot) if v else {"met": False}
            out["R8"][enc] = {"evaluated": bool(ev), "gate_passed": passed, "n_seen_perm_by_order": cnt,
                              "quantity": "A1 SS_perm_bal, seen families, NegPairs-Perm", **res, "met": bool(ev and res.get("met"))}
        if enc in STATIC:
            v = _by_seed(rows_fresh, enc, "max", "unseen.SS")
            out["R1_descriptive"][enc] = {"A1_unseen_SS_fresh_by_seed": [v[s] for s in sorted(v)],
                                          "seed_mean": (sum(v.values()) / len(v)) if v else None,
                                          "v2_threshold": thr["R1_max_unseen_SS_descriptive"],
                                          "cluster_interval": cluster_bootstrap_flip(runs_fresh, fresh_by_id, enc, "max", None, "SS", "unseen", None, n_boot)}
    return out


def descriptive_blocks(rows_fresh, runs_fresh, fresh_by_id, rows_perm, runs_perm, perm_by_id, rows_v1, encoders, n_boot=10000) -> dict:
    """Bridge, R4 (A5), D1, MaxSim (M0 is scored once and appears with seed 0), A4 and D1 on Perm under the static
    encoders, same-cue analysis, per-family Perm (manifest v3.0, 8.4)."""
    out = {}
    for enc in encoders:
        b = {"fresh": {"A1_seen_SS": _seed_mean(rows_fresh, enc, "max", "seen.SS"), "A1_unseen_SS": _seed_mean(rows_fresh, enc, "max", "unseen.SS"),
                       "A1_unseen_SS_cluster": cluster_bootstrap_flip(runs_fresh, fresh_by_id, enc, "max", None, "SS", "unseen", None, n_boot)},
             "perm": {g: {"A1_SS_pooled": _seed_mean(rows_perm, enc, "max", f"{g}.pooled.SS"), "A1_SS_bal": _seed_mean(rows_perm, enc, "max", f"{g}.SS_bal"),
                          "A1_joint_NS_acc": _seed_mean(rows_perm, enc, "max", f"{g}.pooled.joint_NS_acc")}
                      for g in ("all", "seen", "unseen", "family.syntactic", "family.quantifier", "family.affixal", "family.lexical")},
             "v1test": {"A1_seen_SS": _seed_mean(rows_v1, enc, "max", "seen.SS"), "A1_unseen_SS": _seed_mean(rows_v1, enc, "max", "unseen.SS")}}
        b["R4_A5"] = {k: contrast(rows_fresh, enc, a, c, key) for k, (a, c, key) in {
            "A5_minus_A1_unseen_SS": ("signed_max_gate", "max", "unseen.SS"),
            "A5_minus_A4_unseen_SS": ("signed_max_gate", "signed_max", "unseen.SS"),
            "A5_minus_A1_seen_SS": ("signed_max_gate", "max", "seen.SS")}.items()}
        b["D1"] = {k: contrast(rows_fresh, enc, a, c, key) for k, (a, c, key) in {
            "D1_minus_A1_unseen_SS": ("signed_max_oracle", "max", "unseen.SS"),
            "D1_minus_A4_unseen_SS": ("signed_max_oracle", "signed_max", "unseen.SS"),
            "D1_minus_A1_seen_SS": ("signed_max_oracle", "max", "seen.SS")}.items()}
        b["MaxSim"] = {"M1_minus_A1_seen_SS_fresh": contrast(rows_fresh, enc, "m1", "max", "seen.SS"),
                       "M1_minus_A1_unseen_SS_fresh": contrast(rows_fresh, enc, "m1", "max", "unseen.SS"),
                       "M1_seen_SS_bal_perm": _seed_mean(rows_perm, enc, "m1", "seen.SS_bal"),
                       "M0_seen_SS_fresh": _seed_mean(rows_fresh, enc, "m0", "seen.SS"),
                       "M0_unseen_SS_fresh": _seed_mean(rows_fresh, enc, "m0", "unseen.SS"),
                       "M0_seen_SS_bal_perm": _seed_mean(rows_perm, enc, "m0", "seen.SS_bal")}
        b["perm_signed"] = {agg: {"seen_SS_bal": _seed_mean(rows_perm, enc, agg, "seen.SS_bal"),
                                  "seen_joint_NS_acc": _seed_mean(rows_perm, enc, agg, "seen.pooled.joint_NS_acc"),
                                  "unseen_SS_bal": _seed_mean(rows_perm, enc, agg, "unseen.SS_bal")}
                            for agg in ("signed_max", "signed_max_oracle", "signed_max_sentence", "signed_max_gate")}
        b["samecue_not_fresh"] = {agg: {"flip_O_N": _seed_mean(rows_fresh, enc, agg, "samecue_not.flip_O_N"),
                                        "flip_O_S": _seed_mean(rows_fresh, enc, agg, "samecue_not.flip_O_S"),
                                        "n_N": _seed_mean(rows_fresh, enc, agg, "samecue_not.n_N"),
                                        "n_S": _seed_mean(rows_fresh, enc, agg, "samecue_not.n_S")}
                                  for agg in ("max", "lse", "mean", "signed_max", "signed_max_gate", "m1")}
        out[enc] = b
    return out


def rule_summary(rules: dict) -> dict:
    return {"G": {e: v["passed"] for e, v in rules["gate"]["encoders"].items()},
            "k_R2_R3": rules["gate"]["k_R2_R3"], "k_R8": rules["gate"]["k_R8"],
            **{r: {e: ("met" if v["met"] else ("not met" if v["evaluated"] else "not evaluable")) for e, v in rules[r].items()}
               for r in ("R2", "R3", "R8")}}


PERM_TABLE_COLS = ["encoder", "aggregator", "n_seeds", "seen.SS_bal", "seen.pooled.SS", "seen.first.SS", "seen.second.SS",
                   "seen.pooled.joint_NS_acc", "unseen.SS_bal", "unseen.pooled.joint_NS_acc", "all.pooled.flip_O_P"]
FLIP_TABLE_COLS = ["encoder", "aggregator", "n_seeds", "seen.SS", "unseen.SS", "seen.flip_O_N", "seen.flip_O_S", "unseen.flip_O_N",
                   "unseen.flip_O_S", "S_same_scen.flip_O_S", "samecue_not.flip_O_N", "samecue_not.flip_O_S", "all.flip_O_P"]
ACC_TABLE_COLS = ["encoder", "aggregator", "n_seeds", "all.acc", "all.acc_O", "all.acc_P", "all.acc_S", "all.acc_N", "unseen.acc_N",
                  "seen.acc_N", "all.acc_N_given_O", "all.consistency", "prior_incongruent.acc_OP", "prior_incongruent.acc_N"]
