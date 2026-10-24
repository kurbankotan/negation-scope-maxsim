"""Independent recomputation of the NegPairs v3.0 run (does not import cacsa2).

Inputs: the compact raw predictions (sample_id, pred, probs) of every run, the data files (v1 test, Fresh, Perm, dev),
the S-same scenario lists of the run and the Perm clause-order field. Recomputes, per run and set: flip rates,
scope sensitivity (all / seen / unseen / family), S-same flip_O_S, same-cue "not" flips, arm accuracies, the Perm
order-split estimands and joint (N, S) accuracy, the dev gate quantity; compares every value with the notebook's
per_run_*.json and gate.json; recomputes the seed means and the rule quantities (seed means, t intervals).
"""
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

RES = Path(sys.argv[1])          # results folder (np_v3_main)
PRED = Path(sys.argv[2])         # predictions folder (np_v3_main with runs/)
DATA_V1 = Path(sys.argv[3])      # negpairs_v1 folder (test.jsonl, dev.jsonl)
DATA_V3 = Path(sys.argv[4])      # negpairs_v3 folder (fresh.jsonl, perm.jsonl)
OUT = Path(sys.argv[5])
ARMS = ("O", "P", "S", "N")


def load_rows(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[r["item_id"]] = r
    return rows


ROWS = {"v1test": load_rows(DATA_V1 / "test.jsonl"), "dev": load_rows(DATA_V1 / "dev.jsonl"),
        "fresh": load_rows(DATA_V3 / "fresh.jsonl"), "perm": load_rows(DATA_V3 / "perm.jsonl")}
S_SAME_SCEN = {s: sorted({ROWS[s][i]["scenario_id"] for i in json.load(open(RES / f"S_same_ids_{s}.json"))}) for s in ("v1test", "fresh")}


def cue_lemma(cue):
    c = (cue or "").lower().strip()
    if ("n't" in c or re.search(r"\bnot\b", c)) and "cannot" not in c:
        return "not"
    c = re.sub(r"^(is|are|was|were|does|do|did|has|have|had|can|could|will|would)\s+", "", c)
    return c.split()[0] if c else ""


def rate(vals):
    return (100.0 * sum(vals) / len(vals)) if vals else None


def scenarios(recs, rows):
    by = defaultdict(dict)
    for r in recs:
        row = rows[r["sample_id"]]
        by[row["scenario_id"]][row["arm"]] = (r["pred"], "AB"[int(row["label"])], row)
    return by


def flips(by):
    out = {}
    for sid, arms in by.items():
        if "O" in arms:
            out[sid] = {a: int(arms[a][0] != arms["O"][0]) for a in ("N", "S", "P") if a in arms}
    return out


def standard_block(recs, rows):
    by = scenarios(recs, rows)
    fl = flips(by)
    out = {}
    for a in ("N", "S", "P"):
        out[f"flip_O_{a}"] = rate([v[a] for v in fl.values() if a in v])
    out["SS"] = out["flip_O_N"] - out["flip_O_S"] if out["flip_O_N"] is not None and out["flip_O_S"] is not None else None
    for a in ARMS:
        out[f"acc_{a}"] = rate([int(p == l) for arms in by.values() if a in arms for (p, l, _) in [arms[a]]])
    aff = [x for x in (out["acc_O"], out["acc_P"]) if x is not None]
    out["acc_OP"] = sum(aff) / len(aff) if aff else None
    out["acc"] = rate([int(r["pred"] == "AB"[int(rows[r["sample_id"]]["label"])]) for r in recs])
    return out


def standard_metrics(recs, rows, s_same):
    flat = {}
    for name, pred in (("all", lambda r: True), ("seen", lambda r: r["seen_family"]), ("unseen", lambda r: not r["seen_family"])):
        sub = [r for r in recs if pred(rows[r["sample_id"]])]
        if sub:
            for k, v in standard_block(sub, rows).items():
                flat[f"{name}.{k}"] = v
    fl = flips(scenarios(recs, rows))
    ss = set(s_same)
    v = [fl[s]["S"] for s in fl if s in ss and "S" in fl[s]]
    flat["S_same_scen.flip_O_S"] = rate(v)
    by = scenarios(recs, rows)
    for a in ("N", "S"):
        v = [fl[sid][a] for sid, arms in by.items() if sid in fl and a in fl[sid] and arms[a][2]["seen_family"] and cue_lemma(arms[a][2]["cue"]) == "not"]
        flat[f"samecue_not.flip_O_{a}"] = rate(v)
        flat[f"samecue_not.n_{a}"] = len(v)
    return flat


def perm_block(recs, rows):
    by = scenarios(recs, rows)
    full = [arms for arms in by.values() if all(a in arms for a in ARMS)]
    out = {"n_scenarios": len(full)}
    for a in ("N", "S", "P"):
        out[f"flip_O_{a}"] = rate([int(arms[a][0] != arms["O"][0]) for arms in full])
    out["SS"] = out["flip_O_N"] - out["flip_O_S"] if full else None
    out["joint_NS_acc"] = rate([int(arms["N"][0] == arms["N"][1] and arms["S"][0] == arms["S"][1]) for arms in full])
    for a in ARMS:
        out[f"acc_{a}"] = rate([int(arms[a][0] == arms[a][1]) for arms in full])
    return out


def perm_metrics(recs, rows):
    flat = {}
    groups = {"all": lambda r: True, "seen": lambda r: r["seen_family"], "unseen": lambda r: not r["seen_family"]}
    for fam in sorted({rows[r["sample_id"]]["family"] for r in recs}):
        groups[f"family.{fam}"] = (lambda f: (lambda r: r["family"] == f))(fam)
    for name, pred in groups.items():
        sub = [r for r in recs if pred(rows[r["sample_id"]])]
        blocks = {"pooled": perm_block(sub, rows)}
        for order, flag in (("first", True), ("second", False)):
            blocks[order] = perm_block([r for r in sub if bool(rows[r["sample_id"]]["perm"]["decisive_first"]) == flag], rows)
        for k, b in blocks.items():
            for m, v in b.items():
                flat[f"{name}.{k}.{m}"] = v
        a, b = blocks["first"]["SS"], blocks["second"]["SS"]
        flat[f"{name}.SS_bal"] = (a + b) / 2 if a is not None and b is not None else None
    return flat


def read_preds(path):
    return [json.loads(l) for l in open(path) if l.strip()]


runs = sorted(p.parent for p in (PRED / "runs").glob("*/*/seed*/fresh_predictions.jsonl"))
print(f"{len(runs)} runs with predictions")
report = {"runs": len(runs), "compared": 0, "mismatches": [], "max_abs_diff": 0.0, "missing_keys": 0}
per_run_nb = {s: {(r["encoder"], r["aggregator"], r["seed"]): r for r in json.load(open(RES / f"per_run_{s}.json"))} for s in ("fresh", "v1test", "perm")}
recomputed = {s: [] for s in ("fresh", "v1test", "perm")}
for d in runs:
    enc, cond, seed = d.parent.parent.name, d.parent.name, int(d.name[4:])
    for s in ("fresh", "v1test", "perm"):
        recs = read_preds(d / f"{s}_predictions.jsonl")
        if s == "perm":
            flat = perm_metrics(recs, ROWS["perm"])
        else:
            flat = standard_metrics(recs, ROWS[s], S_SAME_SCEN[s])
        flat.update({"encoder": enc, "aggregator": cond, "seed": seed})
        recomputed[s].append(flat)
        nb = per_run_nb[s].get((enc, cond, seed))
        if nb is None:
            report["mismatches"].append({"run": f"{enc}/{cond}/{seed}", "set": s, "problem": "run missing in notebook per_run"})
            continue
        for k, v in flat.items():
            if k in ("encoder", "aggregator", "seed"):
                continue
            if k not in nb:
                report["missing_keys"] += 1
                continue
            w = nb[k]
            report["compared"] += 1
            if (v is None) != (w is None):
                report["mismatches"].append({"run": f"{enc}/{cond}/{seed}", "set": s, "key": k, "recomputed": v, "notebook": w})
            elif v is not None:
                diff = abs(v - w)
                report["max_abs_diff"] = max(report["max_abs_diff"], diff)
                if diff > 1e-9:
                    report["mismatches"].append({"run": f"{enc}/{cond}/{seed}", "set": s, "key": k, "recomputed": v, "notebook": w})

# dev gate: A1 seed-mean SS on dev
gate_nb = json.load(open(RES / "gate.json"))
gate = {}
for d in runs:
    enc, cond, seed = d.parent.parent.name, d.parent.name, int(d.name[4:])
    if cond != "max" or not (d / "dev_predictions.jsonl").exists():
        continue
    b = standard_block(read_preds(d / "dev_predictions.jsonl"), ROWS["dev"])
    gate.setdefault(enc, {})[seed] = b["SS"]
gate_cmp = {}
for enc, vals in gate.items():
    m = sum(vals.values()) / len(vals)
    nbv = gate_nb["encoders"][enc]["A1_seed_mean_dev_SS"]
    gate_cmp[enc] = {"recomputed": m, "notebook": nbv, "passed_recomputed": m >= gate_nb["threshold"], "passed_notebook": gate_nb["encoders"][enc]["passed"]}
    report["compared"] += 1
    if abs(m - nbv) > 1e-9:
        report["mismatches"].append({"gate": enc, "recomputed": m, "notebook": nbv})


def seed_means(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["encoder"], r["aggregator"])].append(r)
    out = {}
    for key, rs in by.items():
        rec = {}
        for k in rs[0]:
            if k in ("encoder", "aggregator", "seed"):
                continue
            vals = [r[k] for r in rs if r.get(k) is not None]
            rec[k] = sum(vals) / len(vals) if vals else None
        out[key] = rec
    return out


def t_interval(vals, conf):
    from scipy import stats
    n = len(vals)
    m = sum(vals) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))
    h = stats.t.ppf(0.5 + conf / 2, df=n - 1) * sd / math.sqrt(n)
    return m, m - h, m + h, sum(v > 0 for v in vals)


rules_nb = json.load(open(RES / "rules.json"))
sm_fresh, sm_perm = seed_means(recomputed["fresh"]), seed_means(recomputed["perm"])
rule_cmp = {}
k23, k8 = rules_nb["gate"]["k_R2_R3"], rules_nb["gate"]["k_R8"]
for enc in gate_nb["encoders"]:
    fr = {r["seed"]: r for r in recomputed["fresh"] if r["encoder"] == enc}
    rows_a4 = {r["seed"]: r for r in recomputed["fresh"] if r["encoder"] == enc and r["aggregator"] == "signed_max"}
    rows_a1 = {r["seed"]: r for r in recomputed["fresh"] if r["encoder"] == enc and r["aggregator"] == "max"}
    rows_a6 = {r["seed"]: r for r in recomputed["fresh"] if r["encoder"] == enc and r["aggregator"] == "signed_max_sentence"}
    seeds = sorted(set(rows_a4) & set(rows_a1))
    d2 = [rows_a4[s]["unseen.SS"] - rows_a1[s]["unseen.SS"] for s in seeds]
    d3 = [rows_a6[s]["S_same_scen.flip_O_S"] - rows_a4[s]["S_same_scen.flip_O_S"] for s in seeds]
    ent = {}
    if d2:
        m, lo, hi, pos = t_interval(d2, 1 - 0.05 / max(1, k23))
        ent["R2"] = {"mean": m, "low": lo, "high": hi, "positive": pos, "notebook_mean": rules_nb["R2"][enc].get("t_interval", {}).get("mean")}
        m, lo, hi, pos = t_interval(d3, 1 - 0.05 / max(1, k23))
        ent["R3"] = {"mean": m, "low": lo, "high": hi, "positive": pos, "notebook_mean": rules_nb["R3"][enc].get("t_interval", {}).get("mean")}
    pr = {r["seed"]: r for r in recomputed["perm"] if r["encoder"] == enc and r["aggregator"] == "max"}
    if pr and enc in rules_nb["R8"]:
        v8 = [pr[s]["seen.SS_bal"] for s in sorted(pr)]
        m, lo, hi, pos = t_interval(v8, 1 - 0.05 / max(1, k8))
        ent["R8"] = {"mean": m, "low": lo, "high": hi, "positive": pos, "notebook_mean": rules_nb["R8"][enc].get("t_interval", {}).get("mean")}
    rule_cmp[enc] = ent
    for rule, e in ent.items():
        if e["notebook_mean"] is not None:
            report["compared"] += 1
            if abs(e["mean"] - e["notebook_mean"]) > 1e-9:
                report["mismatches"].append({"rule": rule, "encoder": enc, "recomputed": e["mean"], "notebook": e["notebook_mean"]})
report["gate"] = gate_cmp
report["rules_t_intervals"] = rule_cmp
report["n_mismatches"] = len(report["mismatches"])
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "recompute_report.json").write_text(json.dumps(report, indent=1, default=str))
json.dump({s: v for s, v in recomputed.items()}, open(OUT / "per_run_recomputed.json", "w"))
print(json.dumps({k: report[k] for k in ("runs", "compared", "n_mismatches", "max_abs_diff", "missing_keys")}, indent=1))
print("gate:", {e: (round(v["recomputed"], 4), round(v["notebook"], 4), v["passed_recomputed"] == v["passed_notebook"]) for e, v in gate_cmp.items()})
for e, ent in rule_cmp.items():
    print(e, {r: (round(x["mean"], 3), round(x["low"], 3), round(x["high"], 3), x["positive"]) for r, x in ent.items()})
print("mismatches (first 10):", report["mismatches"][:10])
