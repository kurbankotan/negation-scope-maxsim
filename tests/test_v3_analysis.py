"""Tests of the v3.0 analysis code: permutation estimands, rule conditions, order-stratified bootstrap, exact checks,
NevIR paired accuracy, probing word selection, ColBERT native scoring, packs."""
import json
import random
from collections import Counter
from pathlib import Path

import pytest
import torch

from cacsa2 import colbert, exact, model as cmodel, negpairs as npx, nevir, np_analysis_v3 as v3, packs, probing
from cacsa2.encoders import EncodedStore, HashEncoder
from cacsa2.stats import paired_t_interval

HERE = Path(__file__).parent


def perm_rows(n_scen=12, seed=0):
    """Synthetic Perm rows: two clauses, 'not' inserted in the decisive (N) or other (S) clause."""
    rng = random.Random(seed)
    rows = []
    for k in range(n_scen):
        first = k % 2 == 0
        fam = "syntactic" if k % 4 < 2 else "affixal"
        dec, oth = f"the shop{k} is open today", f"the park{k} is busy"
        O = (f"{dec}, and {oth}." if first else f"{oth}, and {dec}.")
        O = O[0].upper() + O[1:]
        def edit(clause_dec):
            d = dec.replace("is open", "is not open") if clause_dec else dec
            o = oth if clause_dec else oth.replace("is busy", "is not busy")
            s = (f"{d}, and {o}." if first else f"{o}, and {d}.")
            return s[0].upper() + s[1:]
        N, S = edit(True), edit(False)
        def spans(ctx):
            c = ctx.index(", and")
            c1, c2 = (0, c), (c + 6, len(ctx))
            cue = ctx.find("not")
            return {"clause_decisive": list(c1 if first else c2), "clause_other": list(c2 if first else c1), "connector": [c, c + 5],
                    "cue": [cue, cue + 3] if cue >= 0 else None, "unedited_W": None, "sites": None}
        for arm, ctx, lab in (("O", O, 0), ("P", O.replace("today", "now"), 0), ("N", N, 1), ("S", S, 0)):
            rows.append(npx.NegRow(sample_id=f"p{k}-{arm}", passage_id=f"p{k}", question_id="q", edit_id=npx.ARM_CODE[arm], label="BA"[lab],
                                   cue="not" if arm in "NS" else "", arm=arm, family=fam, seen_family=fam == "syntactic", domain="d",
                                   split="perm", context=ctx, question="What should Ana do?", options=("go to the shop", "stay"),
                                   decisive=f"shop{k} is open", perm={"pattern": "not-insertion", "decisive_first": first, "spans": spans(ctx)}))
    return rows


def records_from(rows, pred_of):
    return [{"sample_id": r.sample_id, "label": r.label, "pred": pred_of(r), "probs": [0.5, 0.5]} for r in rows]


def test_perm_report_bag_invariant_scorer_and_position_heuristic():
    rows = perm_rows(12)
    by = {r.sample_id: r for r in rows}
    # a scorer that ignores position: same prediction for N and S -> SS 0, joint accuracy 0
    rep = v3.perm_report(records_from(rows, lambda r: "A"), by)
    assert rep["all"]["pooled"]["SS"] == 0 and rep["all"]["pooled"]["joint_NS_acc"] == 0 and rep["all"]["SS_bal"] == 0
    # a heuristic that flips whenever the cue sits in the first clause
    def first_clause(r):
        if r.arm in "NS" and r.perm["spans"]["cue"] and r.perm["spans"]["cue"][0] < r.perm["spans"]["connector"][0]:
            return "B"
        return "A"
    rep = v3.perm_report(records_from(rows, first_clause), by)
    assert rep["all"]["first"]["SS"] == 100 and rep["all"]["second"]["SS"] == -100 and rep["all"]["SS_bal"] == 0
    # a scope-aware scorer (labels: "B" for O, P and S; "A" for N)
    rep = v3.perm_report(records_from(rows, lambda r: "A" if r.arm == "N" else "B"), by)
    assert rep["all"]["SS_bal"] == 100 and rep["all"]["pooled"]["joint_NS_acc"] == 100 and rep["seen"]["pooled"]["n_scenarios"] == 6
    flat = v3.flatten_perm(rep)
    assert flat["seen.SS_bal"] == 100 and flat["family.affixal.pooled.flip_O_P"] == 0


def test_conditions_and_rules_v3():
    thr = {"family_alpha": 0.05, "min_positive_seeds": 9, "n_seeds_required": 10}
    vals = {s: 20.0 + (s % 3) for s in range(10)}
    boot = {"low": 15.0, "high": 25.0}
    res = v3._conditions(vals, 15.0, 3, thr, boot)
    assert res["met"] and abs(res["conf"] - (1 - 0.05 / 3)) < 1e-12
    assert res["t_interval"]["low"] > 0
    res = v3._conditions({**vals, 0: -1.0, 1: -2.0}, 15.0, 3, thr, boot)
    assert not res["met"] and not res["conditions"]["b_positive_seeds"]
    res = v3._conditions(vals, 15.0, 3, thr, {"low": -1.0})
    assert not res["met"] and not res["conditions"]["d_cluster_interval_above_zero"]
    res = v3._conditions({s: 20.0 for s in range(9)}, 15.0, 3, thr, boot)       # a seed missing: condition (b) fails
    assert not res["conditions"]["b_positive_seeds"]


def test_dev_gate_and_k():
    rows = perm_rows(8)
    dev = [r for r in rows if r.arm in "OPNS"]
    by = {r.sample_id: r for r in dev}
    runs = {}
    for seed in (1, 2):
        runs[("E1", "max", seed)] = records_from(dev, lambda r: "B" if r.arm == "N" else "A")       # SS 100
        runs[("E0", "max", seed)] = records_from(dev, lambda r: "A")                                # SS 0
    g = v3.dev_gate(runs, by, ["E1", "E0"], ["E1"], {"gate_SS_seen_dev": 20.0})
    assert g["passed"] == ["E1"] and g["k_R2_R3"] == 1 and g["k_R8"] == 1
    assert g["encoders"]["E0"]["A1_seed_mean_dev_SS"] == 0


def test_order_stratified_bootstrap():
    rows = perm_rows(40)
    by = {r.sample_id: r for r in rows}
    runs = {("E", "max", s): records_from(rows, lambda r: "B" if r.arm == "N" else "A") for s in (1, 2)}
    b = v3.cluster_bootstrap_flip(runs, by, "E", "max", None, "SS", "all", None, 200, conf=0.9, by_order=True)
    assert b["point"] == 100 and b["low"] == 100 and b["n_clusters"] == [20, 20]
    runs = {("E", "max", s): records_from(rows, lambda r: "A") for s in (1, 2)}
    b = v3.cluster_bootstrap_flip(runs, by, "E", "max", None, "SS", "seen", None, 200, by_order=True)
    assert b["point"] == 0 and b["high"] == 0


def test_exact_checks_with_random_modules():
    rows = perm_rows(10)
    enc = HashEncoder(dim=32)
    store = EncodedStore(enc)
    store.get_or_encode([r.text for r in rows] + [o for r in rows for o in r.options])
    items = {r.sample_id: npx.build_items([r], store, None)[0] for r in rows}
    item_of = lambda r: items[r.sample_id]
    pp = exact.permutation_pairs(rows); ip = exact.insertion_pairs(rows)
    assert len(pp) == 10 and len(ip) == 20
    for kind in ("max", "mean", "lse", "signed_max_gate", "m1", "head_only"):
        torch.manual_seed(0)
        m = cmodel.make_module(kind, 32)
        r6 = exact.check_r6(m, item_of, pp)
        assert r6["score_violations"] == 0 and r6["decision_disagreements"] == 0, kind
    for kind in ("max", "lse", "m1"):
        torch.manual_seed(1)
        assert exact.check_r5(cmodel.make_module(kind, 32), item_of, ip)["violations"] == 0
    torch.manual_seed(1)
    m = cmodel.make_module("mean", 32)
    with torch.no_grad():
        for prm in m.parameters():
            prm.add_(0.5 * torch.randn_like(prm))             # away from the near-constant random initialization
    assert exact.check_r5(m, item_of, ip)["violations"] > 0
    chk = exact.pilot_exact_checks(32, list(items.values()), rows, item_of, seeds=[1, 2])
    assert chk["passed"] and chk["insertion"]["mean"]["violations"] > 0 and chk["insertion"]["max"]["largest_decrease"] is not None


def test_nested_instances_and_paired_accuracy():
    inst = [{"id": "a", "q1": "x", "q2": "y", "doc1": "The cat sat.", "doc2": "The cat sat on the mat."},
            {"id": "b", "q1": "x", "q2": "y", "doc1": "Sat the cat.", "doc2": "The cat sat."},
            {"id": "c", "q1": "x", "q2": "y", "doc1": "Dogs run.", "doc2": "Cats sleep."}]
    nested = exact.nested_instances(inst)
    assert [(i["id"], i["contained"]) for i in nested] == [("a", "doc1"), ("b", "equal")]
    scores = {"a": {"q1d1": 1.0, "q1d2": 0.5, "q2d1": 0.2, "q2d2": 0.9}, "b": {"q1d1": 1.0, "q1d2": 1.0, "q2d1": 0.0, "q2d2": 1.0},
              "c": {"q1d1": 0.1, "q1d2": 0.2, "q2d1": 0.2, "q2d2": 0.3}}
    acc = nevir.paired_accuracy(scores, inst)
    assert acc["correct"] == [1, 0, 0] and abs(acc["paired_accuracy"] - 100 / 3) < 1e-9
    acc = nevir.paired_accuracy(scores, inst, margin=0.6)
    assert acc["correct"] == [0, 0, 0]
    ev = nevir.evaluate_scorer(scores, inst, [i["id"] for i in nested], 0.0, 100)
    assert ev["nested"]["n"] == 2 and ev["complement"]["n"] == 1 and ev["all"]["bootstrap"]["n"] == 3


def test_probe_words_and_twins():
    rows = perm_rows(4)
    n = next(r for r in rows if r.arm == "N" and r.perm["decisive_first"])
    words = probing.probe_words(n)
    labs = [l for _, _, l in words]
    assert 1 in labs and 0 in labs and all(n.context[s:e] != "not" for s, e, _ in words)
    ctx, span = probing.neutral_twin(n)
    assert "also" in ctx and ctx[span[0]:span[1]] == "also" and "not" not in ctx
    twin = packs.row_from_dict({**packs.row_to_dict(n), "context": ctx})
    tw = probing.probe_words(twin, twin_cue_span=span)
    assert len(tw) == len(words)
    enc = HashEncoder(dim=16)
    e = enc.encode(n.text)
    feats = probing.word_features(e.states, e.offsets, words)
    assert len(feats) == len(words) and feats[0][0].shape == (16,)
    assert probing.gate_auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0


def test_colbert_score_shape():
    Q = torch.nn.functional.normalize(torch.randn(2, 5, 8), dim=-1)
    D = torch.nn.functional.normalize(torch.randn(2, 7, 8), dim=-1)
    s = colbert.ColBERTNative.score(Q, D)
    assert s.shape == (2,) and torch.allclose(s, (Q @ D.transpose(1, 2)).max(2).values.sum(1))


def test_pack_roundtrip(tmp_path):
    rows = perm_rows(6)
    enc = HashEncoder(dim=16); enc.name = "E0-hash"
    store = EncodedStore(enc)
    store.get_or_encode([r.text for r in rows] + [o for r in rows for o in r.options])
    packs.save_pack(tmp_path / "E0-hash", store, {"perm": rows}, None)
    pk = packs.Pack(tmp_path / "E0-hash")
    items = pk.items("perm", "oracle")
    n_items = [it for it in items if it.row.arm == "N"]
    assert all(int(it.polarity.sum()) >= 1 for it in n_items)               # cue (and decisive words) marked
    s_items = [it for it in items if it.row.arm == "S"]
    assert all(int(it.polarity.sum()) == 1 for it in s_items)               # the cue only
    assert pk.rows["perm"][0].perm["spans"]["connector"] == rows[0].perm["spans"]["connector"]
    assert all(int(it.polarity.sum()) == 0 for it in pk.zero_mask_items("perm"))
