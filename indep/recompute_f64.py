"""Independent check of the float64 follow-up of the NegPairs v3.0 run (does not import cacsa2).

Inputs: the follow-up folder (per_run_perm_f64_static.json, seed_means_perm_f64_static.json, perm_f64_vs_gpu.json,
FILE_MANIFEST_SHA256.csv and runs/*/*/seed*/perm_predictions_f64.jsonl), the GPU predictions folder of the main run,
the Perm data file and the main run's FILE_MANIFEST_SHA256.csv.

Checks: (1) SHA-256 of every follow-up file against the follow-up manifest, and that every entry of the main run's
manifest is unchanged in the follow-up manifest; (2) per-run Perm metrics recomputed from the float64 predictions and
compared with per_run_perm_f64_static.json; (3) seed means recomputed and compared; (4) item-level decision
disagreements between the GPU and the float64 predictions recounted and compared with perm_f64_vs_gpu.json;
(5) Proposition A read off the float64 scores: for every unmasked condition and option, |S(N) - S(S)| within rounding
of the 9-decimal archived scores; (6) decision agreement N against S for the unmasked conditions.
"""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

FU = Path(sys.argv[1])            # follow-up folder (np_v3_main with runs/)
PRED = Path(sys.argv[2])          # GPU predictions folder (np_v3_main with runs/)
RES = Path(sys.argv[3])           # main results folder (FILE_MANIFEST_SHA256.csv of the main run)
PERM = Path(sys.argv[4])          # perm.jsonl
OUT = Path(sys.argv[5])
ARMS = ("O", "P", "S", "N")
UNMASKED = {"max", "mean", "lse", "signed_max_gate", "m1", "head_only", "m0", "option_only"}

ROWS = {}
for line in open(PERM, encoding="utf-8"):
    if line.strip():
        r = json.loads(line)
        ROWS[r["item_id"]] = r


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rate(vals):
    return (100.0 * sum(vals) / len(vals)) if vals else None


def scenarios(recs):
    by = defaultdict(dict)
    for r in recs:
        row = ROWS[r["sample_id"]]
        by[row["scenario_id"]][row["arm"]] = (r["pred"], "AB"[int(row["label"])], row, r)
    return by


def perm_block(recs):
    by = scenarios(recs)
    full = [arms for arms in by.values() if all(a in arms for a in ARMS)]
    out = {"n_scenarios": len(full)}
    for a in ("N", "S", "P"):
        out[f"flip_O_{a}"] = rate([int(arms[a][0] != arms["O"][0]) for arms in full])
    out["SS"] = out["flip_O_N"] - out["flip_O_S"] if full else None
    out["joint_NS_acc"] = rate([int(arms["N"][0] == arms["N"][1] and arms["S"][0] == arms["S"][1]) for arms in full])
    for a in ARMS:
        out[f"acc_{a}"] = rate([int(arms[a][0] == arms[a][1]) for arms in full])
    return out


def perm_metrics(recs):
    flat = {}
    groups = {"all": lambda r: True, "seen": lambda r: r["seen_family"], "unseen": lambda r: not r["seen_family"]}
    for fam in sorted({ROWS[r["sample_id"]]["family"] for r in recs}):
        groups[f"family.{fam}"] = (lambda f: (lambda r: r["family"] == f))(fam)
    for name, pred in groups.items():
        sub = [r for r in recs if pred(ROWS[r["sample_id"]])]
        blocks = {"pooled": perm_block(sub)}
        for order, flag in (("first", True), ("second", False)):
            blocks[order] = perm_block([r for r in sub if bool(ROWS[r["sample_id"]]["perm"]["decisive_first"]) == flag])
        for k, b in blocks.items():
            for m, v in b.items():
                flat[f"{name}.{k}.{m}"] = v
        a, b = blocks["first"]["SS"], blocks["second"]["SS"]
        flat[f"{name}.SS_bal"] = (a + b) / 2 if a is not None and b is not None else None
    return flat


def read(path):
    return [json.loads(l) for l in open(path) if l.strip()]


report = {"files": {}, "compared": 0, "mismatches": [], "max_abs_diff": 0.0, "seed_means_compared": 0,
          "seed_means_max_abs_diff": 0.0, "gpu_vs_f64": {}, "propA": {}, "runs": 0}

# ---- (1) manifests
man_fu = {r["path"]: r for r in csv.DictReader(open(FU / "FILE_MANIFEST_SHA256.csv", encoding="utf-8"))}
man_main = {r["path"]: r for r in csv.DictReader(open(RES / "FILE_MANIFEST_SHA256.csv", encoding="utf-8"))}
changed = [p for p, r in man_main.items() if p in man_fu and man_fu[p]["sha256"] != r["sha256"]]
missing = [p for p in man_main if p not in man_fu]
report["files"]["main_manifest_entries"] = len(man_main)
report["files"]["followup_manifest_entries"] = len(man_fu)
report["files"]["main_entries_changed_in_followup"] = changed
report["files"]["main_entries_missing_in_followup"] = missing
present = [p for p in FU.rglob("*") if p.is_file() and p.name != "FILE_MANIFEST_SHA256.csv" and not p.name.endswith(".zip")]
ok = bad = unlisted = 0
for p in present:
    rel = str(p.relative_to(FU))
    if rel not in man_fu:
        unlisted += 1
        continue
    if sha(p) == man_fu[rel]["sha256"] and int(man_fu[rel]["bytes"]) == p.stat().st_size:
        ok += 1
    else:
        bad += 1
        report["mismatches"].append({"file": rel, "problem": "sha256 or size differs from the follow-up manifest"})
report["files"].update({"present": len(present), "hash_ok": ok, "hash_bad": bad, "unlisted": unlisted,
                        "f64_prediction_files_in_manifest": sum(1 for p in man_fu if p.endswith("perm_predictions_f64.jsonl"))})

# ---- (2) per-run metrics
nb = {(r["encoder"], r["aggregator"], r["seed"]): r for r in json.load(open(FU / "per_run_perm_f64_static.json"))}
recomputed = []
dis_total = 0
dis_by_run = {}
propA_max = defaultdict(float)
propA_dec_dis = defaultdict(int)
propA_checks = defaultdict(int)
runs = sorted(FU.glob("runs/*/*/seed*/perm_predictions_f64.jsonl"))
report["runs"] = len(runs)
for f in runs:
    d = f.parent
    enc, cond, seed = d.parent.parent.name, d.parent.name, int(d.name[4:])
    recs = read(f)
    assert all(r.get("dtype") == "float64" for r in recs), f"dtype flag missing in {f}"
    flat = perm_metrics(recs)
    flat.update({"encoder": enc, "aggregator": cond, "seed": seed})
    recomputed.append(flat)
    ref = nb.get((enc, cond, seed))
    if ref is None:
        report["mismatches"].append({"run": f"{enc}/{cond}/{seed}", "problem": "missing in per_run_perm_f64_static.json"})
    else:
        for k, v in flat.items():
            if k in ("encoder", "aggregator", "seed"):
                continue
            if k not in ref:
                report["mismatches"].append({"run": f"{enc}/{cond}/{seed}", "key": k, "problem": "key missing in notebook file"})
                continue
            w = ref[k]
            report["compared"] += 1
            if (v is None) != (w is None):
                report["mismatches"].append({"run": f"{enc}/{cond}/{seed}", "key": k, "recomputed": v, "notebook": w})
            elif v is not None:
                diff = abs(v - w)
                report["max_abs_diff"] = max(report["max_abs_diff"], diff)
                if diff > 1e-9:
                    report["mismatches"].append({"run": f"{enc}/{cond}/{seed}", "key": k, "recomputed": v, "notebook": w})
    # (4) GPU against float64 decisions
    g = PRED / "runs" / enc / cond / d.name / "perm_predictions.jsonl"
    gp = {r["sample_id"]: r["pred"] for r in read(g)}
    n_dis = sum(1 for r in recs if gp[r["sample_id"]] != r["pred"])
    dis_total += n_dis
    if n_dis:
        dis_by_run[f"{enc}/{cond}/{seed}"] = n_dis
    # (5), (6) Proposition A from the archived float64 scores (9 decimals)
    if cond in UNMASKED:
        by = scenarios(recs)
        for sid, arms in by.items():
            if "N" in arms and "S" in arms:
                sn, ss = arms["N"][3]["scores"], arms["S"][3]["scores"]
                for a, b in zip(sn, ss):
                    propA_max[enc] = max(propA_max[enc], abs(a - b))
                propA_checks[enc] += 1
                if arms["N"][0] != arms["S"][0]:
                    propA_dec_dis[enc] += 1
report["gpu_vs_f64"] = {"total_recounted": dis_total, "runs_with_disagreements": len(dis_by_run)}
rep_nb = json.load(open(FU / "perm_f64_vs_gpu.json"))
report["gpu_vs_f64"]["notebook_total"] = rep_nb["item_decision_disagreements_gpu_vs_f64"]["total"]
report["gpu_vs_f64"]["notebook_runs_with_disagreements"] = len(rep_nb["item_decision_disagreements_gpu_vs_f64"]["by_run_nonzero"])
report["gpu_vs_f64"]["by_run_equal"] = dis_by_run == {k: v for k, v in rep_nb["item_decision_disagreements_gpu_vs_f64"]["by_run_nonzero"].items()}
report["propA"] = {enc: {"scenario_checks": propA_checks[enc], "max_abs_score_diff_N_vs_S": propA_max[enc],
                         "decision_disagreements_N_vs_S": propA_dec_dis[enc]} for enc in propA_checks}

# ---- (3) seed means
sm_nb = {(r["encoder"], r["aggregator"]): r for r in json.load(open(FU / "seed_means_perm_f64_static.json"))}
by = defaultdict(list)
for r in recomputed:
    by[(r["encoder"], r["aggregator"])].append(r)
for key, rs in by.items():
    ref = sm_nb[key]
    for k in rs[0]:
        if k in ("encoder", "aggregator", "seed"):
            continue
        vals = [r[k] for r in rs if r.get(k) is not None]
        if not vals or k not in ref or ref[k] is None:
            continue
        m = sum(vals) / len(vals)
        report["seed_means_compared"] += 1
        diff = abs(m - ref[k])
        report["seed_means_max_abs_diff"] = max(report["seed_means_max_abs_diff"], diff)
        if diff > 1e-9:
            report["mismatches"].append({"seed_mean": f"{key[0]}/{key[1]}", "key": k, "recomputed": m, "notebook": ref[k]})
# number of seed-mean differences between float64 and GPU (notebook list) recounted from the GPU seed means
sm_gpu = {(r["encoder"], r["aggregator"]): r for r in json.load(open(RES / "seed_means_perm.json"))}
n_diff = 0
max_diff = 0.0
for key, rs in by.items():
    for k in ("seen.SS_bal", "seen.pooled.SS", "seen.first.SS", "seen.second.SS", "seen.pooled.joint_NS_acc", "unseen.SS_bal", "unseen.pooled.joint_NS_acc", "all.pooled.flip_O_P"):
        vals = [r[k] for r in rs if r.get(k) is not None]
        a = sum(vals) / len(vals)
        b = sm_gpu[key][k]
        if abs(a - b) > 1e-9:
            n_diff += 1
            max_diff = max(max_diff, abs(a - b))
report["gpu_vs_f64"]["seed_mean_differences_recounted"] = n_diff
report["gpu_vs_f64"]["seed_mean_differences_notebook"] = rep_nb["n_differences"]
report["gpu_vs_f64"]["largest_seed_mean_difference_pp"] = max_diff
report["n_mismatches"] = len(report["mismatches"])
OUT.mkdir(parents=True, exist_ok=True)
json.dump(report, open(OUT / "recompute_f64_report.json", "w"), indent=1)
json.dump(recomputed, open(OUT / "per_run_perm_f64_recomputed.json", "w"), indent=0)
print(json.dumps({k: v for k, v in report.items() if k != "mismatches"}, indent=1))
print("mismatches:", report["n_mismatches"])
