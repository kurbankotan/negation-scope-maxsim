"""NegPairs analysis: per-run reports, seed means, paired seed intervals, scenario-cluster bootstrap, McNemar, the
learnability gate and the prespecified rules of manifest v2.0."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .negpairs import NegRow, report, scenario_flips, cue_lemma
from .stats import paired_t_interval, mcnemar_exact

SUBSETS = ("all", "seen", "unseen")
METRICS = ("acc", "acc_O", "acc_P", "acc_S", "acc_N", "acc_OP", "flip_O_P", "flip_O_S", "flip_O_N", "consistency",
           "negation_gap", "scope_gap", "SS", "acc_P_given_O", "acc_S_given_O", "acc_N_given_O", "pair_ON",
           "N_error_margin_mean", "N_error_frac_margin_gt_0.9")


def load_runs(root: str | Path, split: str = "test") -> Dict[Tuple[str, str, int], List[dict]]:
    out = {}
    for summ in Path(root).glob("*/*/seed*/run_summary.json"):
        d = summ.parent
        pred = d / f"{split}_predictions.jsonl"
        if pred.exists():
            out[(d.parent.parent.name, d.parent.name, int(d.name[4:]))] = [json.loads(l) for l in open(pred) if l.strip()]
    return out


def per_run(runs, rows_by_id: Dict[str, NegRow], extra_subsets: Optional[Dict[str, Sequence[str]]] = None,
            s_same_scenarios: Optional[Sequence[str]] = None) -> List[dict]:
    """One flat record per run. `extra_subsets` maps a name to item ids (e.g. "S_same", "prior_incongruent"); the full
    metric block is added for each as "<name>.<metric>", plus "<name>.n" (records in the subset)."""
    rows = []
    for (enc, agg, seed), recs in sorted(runs.items()):
        rep = report(recs, rows_by_id)
        flat = {"encoder": enc, "aggregator": agg, "seed": seed}
        for sub in SUBSETS:
            if rep.get(sub):
                for m in METRICS:
                    flat[f"{sub}.{m}"] = rep[sub].get(m)
        for fam, b in rep["by_family"].items():
            for m in ("acc_OP", "acc_S", "acc_N", "flip_O_N", "acc_N_given_O"):
                flat[f"family.{fam}.{m}"] = b.get(m)
        fl = scenario_flips(recs, rows_by_id)
        if s_same_scenarios is not None:
            v = [fl[sid]["S"] for sid in set(s_same_scenarios) if sid in fl and "S" in fl[sid]]
            flat["S_same_scen.flip_O_S"] = (100.0 * sum(v) / len(v)) if v else None   # own prefix: the "S_same"
            flat["S_same_scen.n"] = len(v)                                          # item subset is S-only
        # same-cue analysis (seen families): decision changes for the cue lemma 'not' when it negates the decisive
        # words (N) and when it negates an irrelevant detail (S); Proposition 2 predicts equal rates under E0
        by_scen = defaultdict(dict)
        for r in recs:
            row = rows_by_id[r["sample_id"]]
            by_scen[row.passage_id][row.arm] = row
        for arm in ("N", "S"):
            v = [fl[sid][arm] for sid, arms in by_scen.items() if sid in fl and arm in fl[sid] and arms[arm].seen_family
                 and cue_lemma(arms[arm].cue) == "not"]
            flat[f"samecue_not.flip_O_{arm}"] = (100.0 * sum(v) / len(v)) if v else None
            flat[f"samecue_not.n_{arm}"] = len(v)
        for name, ids in (extra_subsets or {}).items():
            keep = set(ids)
            sub = [r for r in recs if r["sample_id"] in keep]
            flat[f"{name}.n"] = len(sub)
            block = report(sub, rows_by_id)["all"] if sub else {}
            for m in METRICS:
                flat[f"{name}.{m}"] = block.get(m)
        rows.append(flat)
    return rows


def seed_means(rows: Sequence[dict]) -> List[dict]:
    by = defaultdict(list)
    for r in rows:
        by[(r["encoder"], r["aggregator"])].append(r)
    out = []
    for (enc, agg), rs in sorted(by.items()):
        rec = {"encoder": enc, "aggregator": agg, "n_seeds": len(rs)}
        keys = [k for k in rs[0] if k not in ("encoder", "aggregator", "seed")]
        for k in keys:
            vals = [r[k] for r in rs if r.get(k) is not None]
            rec[k] = (sum(vals) / len(vals)) if vals else None
            if len(vals) > 1:
                rec[k + "_sd"] = paired_t_interval(vals)["sd"]
        out.append(rec)
    return out


def contrast(rows: Sequence[dict], encoder: str, agg_a: str, agg_b: str, key: str) -> dict:
    a = {r["seed"]: r.get(key) for r in rows if r["encoder"] == encoder and r["aggregator"] == agg_a}
    b = {r["seed"]: r.get(key) for r in rows if r["encoder"] == encoder and r["aggregator"] == agg_b}
    seeds = sorted(s for s in set(a) & set(b) if a[s] is not None and b[s] is not None)
    diffs = [a[s] - b[s] for s in seeds]
    ci = paired_t_interval(diffs) if diffs else {"n": 0}
    return {"encoder": encoder, "contrast": f"{agg_a} - {agg_b}", "metric": key, "seeds": seeds, "diffs": diffs, **ci}


def cluster_bootstrap_acc_diff(runs, rows_by_id: Dict[str, NegRow], encoder: str, agg_a: str, agg_b: str,
                               arms: Sequence[str] = ("N",), subset: str = "all", n_boot: int = 10000, seed: int = 20261003,
                               ids: Optional[Sequence[str]] = None) -> dict:
    """Scenario-cluster percentile bootstrap of the seed-mean accuracy difference agg_a - agg_b on the given arms
    (optionally restricted to the item ids `ids`)."""
    import numpy as np

    seeds = sorted(set(s for (e, g, s) in runs if e == encoder and g == agg_a) & set(s for (e, g, s) in runs if e == encoder and g == agg_b))
    if not seeds:
        return {"point": None, "low": None, "high": None}
    id_set = set(ids) if ids is not None else None
    def keep(r):
        row = rows_by_id[r["sample_id"]]
        return (row.arm in arms and (subset == "all" or (subset == "seen") == row.seen_family)
                and (id_set is None or r["sample_id"] in id_set))
    scen = sorted({rows_by_id[r["sample_id"]].passage_id for s in seeds for r in runs[(encoder, agg_a, s)] if keep(r)})
    if not scen:
        return {"point": None, "low": None, "high": None, "n_clusters": 0}
    idx = {p: k for k, p in enumerate(scen)}
    C = len(scen)
    cor = {g: np.zeros(C) for g in (agg_a, agg_b)}; tot = {g: np.zeros(C) for g in (agg_a, agg_b)}
    for g in (agg_a, agg_b):
        for s in seeds:
            for r in runs[(encoder, g, s)]:
                if keep(r):
                    k = idx[rows_by_id[r["sample_id"]].passage_id]
                    tot[g][k] += 1; cor[g][k] += (r["pred"] == r["label"])
    point = 100.0 * (cor[agg_a].sum() / tot[agg_a].sum() - cor[agg_b].sum() / tot[agg_b].sum())
    rng = np.random.default_rng(seed)
    reps = []
    for start in range(0, n_boot, 1000):
        m = min(1000, n_boot - start)
        ii = rng.integers(0, C, size=(m, C))
        reps.append(100.0 * (cor[agg_a][ii].sum(1) / tot[agg_a][ii].sum(1) - cor[agg_b][ii].sum(1) / tot[agg_b][ii].sum(1)))
    reps = np.concatenate(reps)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"point": float(point), "low": float(lo), "high": float(hi), "n_clusters": C, "n_boot": n_boot, "arms": list(arms),
            "subset": subset, "restricted_to_ids": id_set is not None}


def mcnemar_by_seed(runs, rows_by_id, encoder: str, agg_a: str, agg_b: str, arms=("N",), subset="all",
                    ids: Optional[Sequence[str]] = None, tag: Optional[str] = None) -> List[dict]:
    out = []
    id_set = set(ids) if ids is not None else None
    for s in sorted(set(x for (e, g, x) in runs if e == encoder and g == agg_a) & set(x for (e, g, x) in runs if e == encoder and g == agg_b)):
        B = {r["sample_id"]: r for r in runs[(encoder, agg_b, s)]}
        b = c = 0
        for r in runs[(encoder, agg_a, s)]:
            row = rows_by_id[r["sample_id"]]
            if row.arm not in arms or not (subset == "all" or (subset == "seen") == row.seen_family):
                continue
            if id_set is not None and r["sample_id"] not in id_set:
                continue
            ok_a, ok_b = r["pred"] == r["label"], B[r["sample_id"]]["pred"] == B[r["sample_id"]]["label"]
            b += ok_a and not ok_b; c += ok_b and not ok_a
        out.append({"encoder": encoder, "seed": s, "contrast": f"{agg_a} vs {agg_b}", "arms": list(arms), "subset": tag or subset,
                    **mcnemar_exact(b, c)})
    return out


def _seed_mean(rows, enc, agg, key):
    vals = [r[key] for r in rows if r["encoder"] == enc and r["aggregator"] == agg and r.get(key) is not None]
    return (sum(vals) / len(vals)) if vals else None


def evaluate_rules_accuracy(rows, runs, rows_by_id, encoders: Sequence[str], thr: dict, n_boot: int = 10000,
                   s_same: Optional[Sequence[str]] = None, strata_ids: Optional[Dict[str, Sequence[str]]] = None) -> dict:
    """SECONDARY (descriptive since manifest v2.0 frozen): the accuracy-based gate and rules of draft 1, kept so that the
    accuracy view is reported with the same definitions. The primary rules are in `evaluate_rules_flip`.

    Option-prior strata (from the context-blind control C2, see negpairs.option_prior_strata) enter G and R1:
    G  (per encoder): (i) A1 seed-mean test acc_OP >= gate_acc_OP; (ii) if the prior-incongruent stratum has at least
       min_stratum_scenarios scenarios, A1 seed-mean acc_OP on it >= gate_incongruent_acc_OP (the context must beat the
       option prior); otherwise A1 - C2 seed-mean acc_OP >= gate_margin_fallback.
    R1 (limit): A1 unseen-family negation gap >= R1_min_unseen_negation_gap in every seed, the gap being the mean of the
       gaps in the two option-prior strata (each with at least min_stratum_scenarios unseen scenarios; otherwise the
       pooled unseen gap is used and flagged).
    R2 (scope supplied): A4 - A1 on acc_N >= R2_min_gain_N, positive in every seed, paired-seed CI low > 0,
       and A4 - A1 on acc_OP >= R2_max_loss_OP.
    R3 (scope precision): on S items whose cue shares a sentence with the decisive words (`s_same`), A4 - A6 accuracy
       >= R3_min_gain_S, positive in every seed, paired-seed CI low > 0; evaluated only if len(s_same) >= R3_min_items.
    """
    out = {"thresholds": thr, "G": {}, "R1": {}, "R2": {}, "R3": {}, "R4_descriptive": {}, "D1_oracle_descriptive": {}}
    n_same = len(s_same) if s_same is not None else 0
    sid = strata_ids or {}
    n_scen = {k: len({rows_by_id[i].passage_id for i in v}) for k, v in sid.items()}
    k_min = thr["min_stratum_scenarios"]
    for enc in encoders:
        g_a1 = _seed_mean(rows, enc, "max", "all.acc_OP")
        g_c2 = _seed_mean(rows, enc, "option_only", "all.acc_OP")
        g_inc = _seed_mean(rows, enc, "max", "prior_incongruent.acc_OP")
        use_stratum = n_scen.get("prior_incongruent", 0) >= k_min
        cond2 = ((g_inc is not None and g_inc >= thr["gate_incongruent_acc_OP"]) if use_stratum
                 else (g_a1 is not None and g_c2 is not None and g_a1 - g_c2 >= thr["gate_margin_fallback"]))
        passed = bool(g_a1 is not None and g_a1 >= thr["gate_acc_OP"] and cond2)
        out["G"][enc] = {"A1_seed_mean_acc_OP": g_a1, "A1_seed_mean_acc_OP_prior_incongruent": g_inc,
                         "C2_option_only_seed_mean_acc_OP": g_c2, "margin_over_C2": (g_a1 - g_c2) if (g_a1 is not None and g_c2 is not None) else None,
                         "n_scenarios_by_stratum": n_scen, "condition_ii": "prior_incongruent" if use_stratum else "margin_fallback",
                         "passed": passed}
        a1 = sorted((r for r in rows if r["encoder"] == enc and r["aggregator"] == "max"), key=lambda r: r["seed"])
        balanced = n_scen.get("unseen_prior_congruent", 0) >= k_min and n_scen.get("unseen_prior_incongruent", 0) >= k_min
        gaps = []
        for r in a1:
            if balanced:
                a, b = r.get("unseen_prior_congruent.negation_gap"), r.get("unseen_prior_incongruent.negation_gap")
                gaps.append(None if a is None or b is None else (a + b) / 2)
            else:
                gaps.append(r.get("unseen.negation_gap"))
        out["R1"][enc] = {"evaluated": passed, "gap_definition": "stratum-balanced" if balanced else "pooled (strata too small)",
                          "A1_unseen_negation_gap_by_seed": gaps,
                          "A1_unseen_pooled_gap_by_seed": [r.get("unseen.negation_gap") for r in a1],
                          "A1_unseen_acc_N_given_O_by_seed": [r.get("unseen.acc_N_given_O") for r in a1],
                          "met": bool(passed and gaps and all(g is not None and g >= thr["R1_min_unseen_negation_gap"] for g in gaps))}
        cN = contrast(rows, enc, "signed_max", "max", "all.acc_N")
        cOP = contrast(rows, enc, "signed_max", "max", "all.acc_OP")
        bN = cluster_bootstrap_acc_diff(runs, rows_by_id, enc, "signed_max", "max", ("N",), "all", n_boot)
        met2 = (passed and cN.get("n", 0) > 0 and cN["mean"] >= thr["R2_min_gain_N"] and cN["positive_count"] == cN["n"]
                and cN["low"] > 0 and cOP.get("n", 0) > 0 and cOP["mean"] >= thr["R2_max_loss_OP"])
        out["R2"][enc] = {"evaluated": passed, "A4_minus_A1_acc_N": cN, "cluster_ci_acc_N": bN, "A4_minus_A1_acc_OP": cOP,
                          "secondary_unseen_acc_N": contrast(rows, enc, "signed_max", "max", "unseen.acc_N"),
                          "met": bool(met2)}
        ev3 = bool(passed and n_same >= thr["R3_min_items"])
        cS = contrast(rows, enc, "signed_max", "signed_max_sentence", "S_same.acc")
        bS = (cluster_bootstrap_acc_diff(runs, rows_by_id, enc, "signed_max", "signed_max_sentence", ("S",), "all", n_boot, ids=s_same)
              if s_same else {"point": None})
        met3 = (ev3 and cS.get("n", 0) > 0 and cS["mean"] >= thr["R3_min_gain_S"] and cS["positive_count"] == cS["n"]
                and cS["low"] > 0)
        out["R3"][enc] = {"evaluated": ev3, "n_S_same_items": n_same, "A4_minus_A6_acc_S_same": cS, "cluster_ci_acc_S_same": bS,
                          "secondary_all_S": contrast(rows, enc, "signed_max", "signed_max_sentence", "all.acc_S"), "met": bool(met3)}
        out["R4_descriptive"][enc] = {k: contrast(rows, enc, a, b, key) for k, (a, b, key) in {
            "A5_minus_A1_unseen_acc_N": ("signed_max_gate", "max", "unseen.acc_N"),
            "A5_minus_A4_unseen_acc_N": ("signed_max_gate", "signed_max", "unseen.acc_N"),
            "A5_minus_A1_seen_acc_N": ("signed_max_gate", "max", "seen.acc_N"),
            "A5_minus_A1_acc_S": ("signed_max_gate", "max", "all.acc_S")}.items()}
        out["D1_oracle_descriptive"][enc] = {k: contrast(rows, enc, a, b, key) for k, (a, b, key) in {
            "oracle_minus_A1_acc_N": ("signed_max_oracle", "max", "all.acc_N"),
            "oracle_minus_A4_acc_N": ("signed_max_oracle", "signed_max", "all.acc_N"),
            "oracle_minus_A4_unseen_acc_N": ("signed_max_oracle", "signed_max", "unseen.acc_N"),
            "oracle_minus_A1_acc_OP": ("signed_max_oracle", "max", "all.acc_OP")}.items()}
    return out


def cluster_bootstrap_flip(runs, rows_by_id: Dict[str, NegRow], encoder: str, agg_a: str, agg_b: Optional[str] = None,
                           kind: str = "SS", subset: str = "unseen", scenario_ids: Optional[Sequence[str]] = None,
                           n_boot: int = 10000, seed: int = 20261003) -> dict:
    """Scenario-cluster percentile bootstrap for flip-based estimands, pooled over the seeds both conditions share.

    kind "SS": per scenario (flip_N - flip_S); kind "S": flip_S; kind "N": flip_N. With agg_b, the difference
    agg_a - agg_b of the pooled rates; without, the rate of agg_a alone. subset: "seen", "unseen" or "all"."""
    import numpy as np

    aggs = [agg_a] + ([agg_b] if agg_b else [])
    seed_sets = [set(s for (e, g, s) in runs if e == encoder and g == a) for a in aggs]
    seeds = sorted(set.intersection(*seed_sets)) if seed_sets else []
    if not seeds:
        return {"point": None, "low": None, "high": None}
    keep = set(scenario_ids) if scenario_ids is not None else None
    def ok(sid, row):
        return (subset == "all" or (subset == "seen") == row.seen_family) and (keep is None or sid in keep)
    per = {}
    for a in aggs:
        for sd in seeds:
            recs = runs[(encoder, a, sd)]
            fl = scenario_flips(recs, rows_by_id)
            row_of = {rows_by_id[r["sample_id"]].passage_id: rows_by_id[r["sample_id"]] for r in recs}
            for sid, v in fl.items():
                if not ok(sid, row_of[sid]):
                    continue
                if kind == "SS" and ("N" not in v or "S" not in v):
                    continue
                if kind in ("S", "N") and kind not in v:
                    continue
                val = (v["N"] - v["S"]) if kind == "SS" else v[kind]
                d = per.setdefault(sid, {x: [0.0, 0] for x in aggs})
                d[a][0] += val; d[a][1] += 1
    scen = sorted(per)
    if not scen:
        return {"point": None, "low": None, "high": None, "n_clusters": 0}
    num = {a: np.array([per[sid][a][0] for sid in scen]) for a in aggs}
    den = {a: np.array([per[sid][a][1] for sid in scen]) for a in aggs}
    def stat(idx=None):
        vals = []
        for a in aggs:
            n_, d_ = (num[a], den[a]) if idx is None else (num[a][idx].sum(axis=-1), den[a][idx].sum(axis=-1))
            vals.append(100.0 * (n_.sum() / max(1, d_.sum())) if idx is None else 100.0 * n_ / np.maximum(1, d_))
        return vals[0] - vals[1] if agg_b else vals[0]
    point = float(stat())
    rng = np.random.default_rng(seed)
    C = len(scen); reps = []
    for start_ in range(0, n_boot, 1000):
        m = min(1000, n_boot - start_)
        reps.append(stat(rng.integers(0, C, size=(m, C))))
    reps = np.concatenate(reps)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"point": point, "low": float(lo), "high": float(hi), "n_clusters": C, "n_boot": n_boot, "seeds": seeds,
            "kind": kind, "subset": subset, "contrast": f"{agg_a} - {agg_b}" if agg_b else agg_a}


def evaluate_rules_flip(rows, runs, rows_by_id, encoders: Sequence[str], thr: dict, n_boot: int = 10000,
                        s_same_scenarios: Optional[Sequence[str]] = None) -> dict:
    """PRIMARY rules of manifest v2.0 (frozen 4 October 2026). SS = flip_O_N - flip_O_S (percentage points).

    G  (per encoder): A1 seed-mean seen-family SS >= gate_SS_seen.
    R1 (E0 encoders only, Proposition 2): A1 seed-mean unseen-family SS <= R1_max_unseen_SS.
    R2 (supplied scope): A4 - A1 unseen-family SS >= R2_min_gain_unseen_SS, positive in every seed, paired-seed
       interval above zero.
    R3 (scope precision): on S-same scenarios, A6 - A4 flip_O_S >= R3_min_gain_flipS, positive in every seed,
       paired-seed interval above zero; evaluated only with at least R3_min_items S-same scenarios.
    R2 and R3 are evaluated for encoders that pass G; R1 for E0 if it passes G. Point estimates decide; scenario-cluster
    bootstrap intervals are attached. Descriptive: the bridge (A1 SS seen and unseen for every encoder), R4 (A5),
    D1 (dataset-supplied polarity), and the same-cue analysis."""
    out = {"thresholds": thr, "G": {}, "R1": {}, "R2": {}, "R3": {}, "bridge_descriptive": {}, "R4_descriptive": {},
           "D1_oracle_descriptive": {}, "samecue_not_descriptive": {}}
    n_same = len(s_same_scenarios) if s_same_scenarios is not None else 0
    for enc in encoders:
        g = _seed_mean(rows, enc, "max", "seen.SS")
        passed = bool(g is not None and g >= thr["gate_SS_seen"])
        out["G"][enc] = {"A1_seed_mean_seen_SS": g, "A1_seen_SS_by_seed": [r.get("seen.SS") for r in sorted(
            (r for r in rows if r["encoder"] == enc and r["aggregator"] == "max"), key=lambda r: r["seed"])], "passed": passed}
        a1u = _seed_mean(rows, enc, "max", "unseen.SS")
        out["bridge_descriptive"][enc] = {"A1_seen_SS": g, "A1_unseen_SS": a1u,
                                          "A1_unseen_flip_O_N": _seed_mean(rows, enc, "max", "unseen.flip_O_N"),
                                          "A1_unseen_flip_O_S": _seed_mean(rows, enc, "max", "unseen.flip_O_S"),
                                          "A1_unseen_SS_cluster_ci": cluster_bootstrap_flip(runs, rows_by_id, enc, "max", None, "SS", "unseen", None, n_boot)}
        if enc.startswith("E0"):
            out["R1"][enc] = {"evaluated": passed, "A1_seed_mean_unseen_SS": a1u,
                              "A1_unseen_SS_by_seed": [r.get("unseen.SS") for r in sorted(
                                  (r for r in rows if r["encoder"] == enc and r["aggregator"] == "max"), key=lambda r: r["seed"])],
                              "cluster_ci": out["bridge_descriptive"][enc]["A1_unseen_SS_cluster_ci"],
                              "met": bool(passed and a1u is not None and a1u <= thr["R1_max_unseen_SS"])}
        c2 = contrast(rows, enc, "signed_max", "max", "unseen.SS")
        b2 = cluster_bootstrap_flip(runs, rows_by_id, enc, "signed_max", "max", "SS", "unseen", None, n_boot)
        met2 = (passed and c2.get("n", 0) > 0 and c2["mean"] >= thr["R2_min_gain_unseen_SS"]
                and c2["positive_count"] == c2["n"] and c2["low"] > 0)
        out["R2"][enc] = {"evaluated": passed, "A4_minus_A1_unseen_SS": c2, "cluster_ci": b2,
                          "secondary_seen_SS": contrast(rows, enc, "signed_max", "max", "seen.SS"),
                          "A4_unseen_flip_O_S_minus_A1": contrast(rows, enc, "signed_max", "max", "unseen.flip_O_S"),
                          "met": bool(met2)}
        ev3 = bool(passed and n_same >= thr["R3_min_items"])
        c3 = contrast(rows, enc, "signed_max_sentence", "signed_max", "S_same_scen.flip_O_S")
        b3 = cluster_bootstrap_flip(runs, rows_by_id, enc, "signed_max_sentence", "signed_max", "S", "all", s_same_scenarios, n_boot) \
            if s_same_scenarios else {"point": None}
        met3 = (ev3 and c3.get("n", 0) > 0 and c3["mean"] >= thr["R3_min_gain_flipS"] and c3["positive_count"] == c3["n"]
                and c3["low"] > 0)
        out["R3"][enc] = {"evaluated": ev3, "n_S_same_scenarios": n_same, "A6_minus_A4_S_same_flip_O_S": c3, "cluster_ci": b3,
                          "met": bool(met3)}
        out["R4_descriptive"][enc] = {k: contrast(rows, enc, a, b, key) for k, (a, b, key) in {
            "A5_minus_A1_unseen_SS": ("signed_max_gate", "max", "unseen.SS"),
            "A5_minus_A4_unseen_SS": ("signed_max_gate", "signed_max", "unseen.SS"),
            "A5_minus_A1_seen_SS": ("signed_max_gate", "max", "seen.SS")}.items()}
        out["D1_oracle_descriptive"][enc] = {k: contrast(rows, enc, a, b, key) for k, (a, b, key) in {
            "oracle_minus_A1_unseen_SS": ("signed_max_oracle", "max", "unseen.SS"),
            "oracle_minus_A4_unseen_SS": ("signed_max_oracle", "signed_max", "unseen.SS"),
            "oracle_minus_A1_seen_SS": ("signed_max_oracle", "max", "seen.SS")}.items()}
        out["samecue_not_descriptive"][enc] = {agg: {"flip_O_N": _seed_mean(rows, enc, agg, "samecue_not.flip_O_N"),
                                                     "flip_O_S": _seed_mean(rows, enc, agg, "samecue_not.flip_O_S"),
                                                     "n_N": _seed_mean(rows, enc, agg, "samecue_not.n_N"),
                                                     "n_S": _seed_mean(rows, enc, agg, "samecue_not.n_S")}
                                               for agg in ("max", "lse", "mean", "signed_max", "signed_max_gate")}
    return out


def markdown(rows: Sequence[dict], cols: Sequence[str], digits: int = 1) -> str:
    f = lambda v: "—" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))
    return ("| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
            + "".join("| " + " | ".join(f(r.get(c)) for c in cols) + " |\n" for r in rows))
