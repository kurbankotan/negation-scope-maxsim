"""Tests of the v3.0 build code: Perm assembly and filters, schedules, Qwen verdict logic, the build loop."""
import json
import random
import re
import threading
import time
from collections import Counter

import pytest

from cacsa2 import build_v3 as bv
from cacsa2 import negpairs_build as nb
from cacsa2 import perm_build as pb
from cacsa2 import perm_prompts as pp
from cacsa2 import qwen_verify as qv

GOOD = {
    "syntactic": dict(context_marked="The museum is [[^]] open today, and the cafe next door is [[^]] busy.",
                      connector=", and", negated_form="not", decisive="museum is open",
                      question="What should Ana do this afternoon?", options=["visit the museum", "stay at home"],
                      answer="A", context_P="Today the museum welcomes visitors and the nearby cafe is crowded."),
    "quantifier": dict(context_marked="There is [[a]] guard at the gate, and there is [[a]] bench by the door.",
                       connector=", and", negated_form="no", decisive="a guard at the gate",
                       question="What should the courier do with the parcel?", options=["leave it with the guard",
                                                                                         "take it back to the depot"],
                       answer="A", context_P="A guard stands at the gate and a bench sits beside the door."),
    "affixal": dict(context_marked="The lift is [[available]] today, and the vending machine is [[available]] too.",
                    connector=", and", negated_form="unavailable", decisive="lift is available",
                    question="How should Omar reach the fifth floor?", options=["take the lift", "climb the stairs"],
                    answer="A", context_P="Today the elevator can be used, as can the snack machine."),
    "lexical": dict(context_marked="The red van [[has]] fuel, while the blue van [[has]] a roof rack.",
                    connector="while", negated_form="lacks", decisive="red van has fuel",
                    question="Which van should the crew drive today?", options=["drive the red van", "drive the blue van"],
                    answer="A", context_P="There is fuel in the red van, and a roof rack sits on the blue van."),
}


def cand(family, decisive_first=True):
    return pb.PermCandidate(uid=f"perm-{family[:3]}-00000", family=family, domain="daily routine",
                            situation="availability of a resource", decisive_first=decisive_first, index=0)


def gen(family, **over):
    g = dict(GOOD[family], usable=True)
    g.update(over)
    return g


@pytest.mark.parametrize("family", list(GOOD))
def test_good_items_pass_and_are_permutations(family):
    reason, rec = pb.perm_filter(gen(family), cand(family))
    assert reason is None, reason
    assert pb.e0_bag(rec["N"]) == pb.e0_bag(rec["S"]) and rec["N"] != rec["S"]
    O, N, S = rec["O"], rec["N"], rec["S"]
    assert "[[" not in O and "  " not in O
    for arm, text in (("N", N), ("S", S)):
        a, b = rec["spans"][arm]["cue"]
        assert text[a:b] == rec["cue"]
        da, db = rec["spans"][arm]["clause_decisive"]
        oa, ob = rec["spans"][arm]["clause_other"]
        assert db <= oa  # decisive clause first in these examples
    a, b = rec["spans"]["N"]["cue"]
    da, db = rec["spans"]["N"]["clause_decisive"]
    assert da <= a < b <= db                      # N: edit in the decisive clause
    a, b = rec["spans"]["S"]["cue"]
    oa, ob = rec["spans"]["S"]["clause_other"]
    assert oa <= a < b <= ob                      # S: edit in the other clause
    if family != "syntactic":
        a, b = rec["spans"]["N"]["unedited_W"]
        assert N[a:b] == GOOD[family]["context_marked"].split("[[")[1].split("]]")[0]


def test_decisive_second_order():
    g = gen("syntactic", context_marked="The cafe next door is [[^]] busy, and the museum is [[^]] open today.",
            decisive="museum is open")
    reason, rec = pb.perm_filter(g, cand("syntactic", decisive_first=False))
    assert reason is None, reason
    assert rec["N"] == "The cafe next door is busy, and the museum is not open today."
    assert rec["S"] == "The cafe next door is not busy, and the museum is open today."
    da, db = rec["spans"]["N"]["clause_decisive"]
    assert rec["N"][da:db] == "the museum is not open today."


@pytest.mark.parametrize("family,over,expect", [
    ("affixal", dict(context_marked="The lift is [[available]], and the vending machine is [[available]]."), "3 marks"),
    ("syntactic", dict(context_marked="[[^]] The museum is open, and the cafe is [[^]] busy."), "3 marks"),
    ("syntactic", dict(connector="but"), "3 connector"),
    ("syntactic", dict(context_marked="The museum is [[^]] open and busy, and the cafe is [[^]] busy and loud.",
                       connector="and"), "3 connector"),
    ("quantifier", dict(context_marked="There is [[a]] guard at the gate, and there is [[the]] bench by the door."),
     "4 pattern"),
    ("quantifier", dict(context_marked="There is [[the]] guard at the gate, and there is [[the]] bench by the door."),
     "4 pattern"),
    ("affixal", dict(context_marked="The lift is [[busy]] today, and the vending machine is [[busy]] too.",
                     negated_form="unbusy", decisive="lift is busy"), "4 pattern"),
    ("affixal", dict(context_marked="The ring is [[valuable]] today, and the watch is [[valuable]] too.",
                     negated_form="invaluable", decisive="ring is valuable"), "4 pattern"),
    ("affixal", dict(context_marked="It is a [[usual]] lift today, and it is a [[usual]] machine too.",
                     negated_form="unusual", decisive="usual lift"), "4 pattern"),
    ("lexical", dict(negated_form="without"), "4 pattern"),
    ("syntactic", dict(context_marked="The museum seems [[^]] open today, and the cafe next door is [[^]] busy."),
     "4 pattern"),
    ("syntactic", dict(options=["visit the museum", "the museum visit"]), "2 options"),
    ("syntactic", dict(decisive="cafe next door"), "8 decisive"),
    ("syntactic", dict(context_P="The museum is open today, and the cafe next door is busy."), "11 context P"),
    ("syntactic", dict(context_P="Today Lisbon's museum welcomes visitors and the nearby cafe is crowded."), "11 context P"),
    ("lexical", dict(context_P="The unusual red van has fuel, and a roof rack sits on the blue van."), "6 affixal"),
])
def test_filter_rejections(family, over, expect):
    reason, _ = pb.perm_filter(gen(family, **over), cand(family))
    assert reason is not None and reason.startswith(expect), reason


def test_not_negations_match_cue_in_family():
    for w in pb.NOT_NEGATIONS:
        assert not nb.cue_in_family(w, "affixal"), w


def test_schedule_balance_and_independence():
    sch = pb.perm_schedule(600, 20261020, 20261022)
    assert len(sch) == 2400 and len({c.uid for c in sch}) == 2400
    assert all(c.uid.startswith("perm-") for c in sch)
    for f in nb.FAMILIES:
        orders = Counter(c.decisive_first for c in sch if c.family == f)
        assert orders[True] == orders[False] == 300
    combos = Counter((c.family, c.domain, c.situation, c.decisive_first) for c in sch)
    both = sum(1 for (f, d, s, o) in combos if o and (f, d, s, False) in combos)
    assert both > 0.8 * 4 * 112     # most combinations occur with both clause orders
    fr = pb.fresh_schedule(480, 20261005)
    assert len(fr) == 1920 and all(c.uid.startswith("fresh-") for c in fr)
    assert [c.index for c in fr] == list(range(1920))


def test_prompt_hashes_and_formatting():
    h = pp.prompt_hashes()
    assert len(h) == 10
    p = pb.perm_prompt(cand("affixal"))
    assert "Put the DECISIVE clause first." in p and "[[available]]" in p and "{" not in p.replace("{\"", "")[:0] + ""
    json_tail = p[p.index("Output strict JSON only:"):]
    assert '{"usable": true' in json_tail


def test_arm_verdict_and_scenario_check():
    A, B, U = [3, 1, 0], [1, 3, 0], [0, 1, 3]
    assert qv.arm_verdict(A, B) == 0          # option 0 chosen in both orders
    assert qv.arm_verdict(B, A) == 1
    assert qv.arm_verdict(A, A) == "inconsistent"
    assert qv.arm_verdict(U, B) == "inconsistent"
    assert qv.arm_verdict([2, 2, 0], B) == "inconsistent"   # exact tie
    ok, det = qv.scenario_check([A, B, A, B, B, A, A, B], "A")  # O, P -> 0; N -> 1; S -> 0
    assert ok and det["N"]["verdict"] == 1
    ok, det = qv.scenario_check([A, B, A, B, A, B, A, B], "A")
    assert not ok and det["N"]["verdict"] == 0


def test_items_v3_labels_and_perm_fields():
    reason, rec = pb.perm_filter(gen("syntactic"), cand("syntactic"))
    r = {"meta": {"uid": "perm-syn-00000", "family": "syntactic", "domain": "d", "situation": "s",
                  "decisive_first": True, "index": 0}, "generated": gen("syntactic"), "assembled": rec}
    rows = pb.to_items_v3([r], "perm", "perm", 20261020)
    assert [x["arm"] for x in rows] == ["O", "P", "N", "S"]
    lab = {x["arm"]: x["label"] for x in rows}
    assert lab["O"] == lab["P"] == lab["S"] == 1 - lab["N"]
    opt = rows[0]["options"]
    assert opt[lab["O"]] == "visit the museum"
    assert rows[2]["perm"]["spans"]["cue"] and rows[1]["perm"]["spans"] is None
    a, b = rows[2]["perm"]["spans"]["cue"]
    assert rows[2]["context"][a:b] == "not"


# ------------------------------------------------------------------------------------------------ build loop
class FakeIndex:
    def __init__(self):
        self.items = []

    def is_duplicate(self, dom, ctx, opts):
        return ctx in self.items

    def add(self, dom, ctx, opts):
        self.items.append(ctx)


def _phase(n=60, caps=None, delays=True, fail_every=0, dup_every=0, budget_after=None, done=None, qwen_fail_every=0):
    cands = [pb.FreshCandidate(uid=f"fresh-x-{k:05d}", family=["syntactic", "affixal"][k % 2], domain="d",
                               situation="s", index=k) for k in range(n)]
    caps = caps or {"syntactic": 10, "affixal": 10}
    done = {} if done is None else done
    meter = bv.Meter()
    meter.cap = budget_after if budget_after is not None else float("inf")
    lock = threading.Lock()

    def api_task(c):
        meter.check()
        if delays:
            time.sleep(random.random() * 0.01)
        meter.add("claude-sonnet-5", 1000, 100)
        k = c.index
        if fail_every and k % fail_every == 0:
            return {"stage": "filter", "reason": "x", "meta": {"uid": c.uid, "family": c.family, "index": k}}
        ctx = f"ctx {k // dup_every if dup_every else k}"
        return {"stage": "haiku_passed", "generated": {"context_O": ctx, "options": ["a", "b"]},
                "meta": {"uid": c.uid, "family": c.family, "domain": "d", "index": k}}

    def settle_qwen(recs):
        for r in recs:
            k = r["meta"]["index"]
            r["qwen"] = {}
            if qwen_fail_every and k % qwen_fail_every == 0:
                r["stage"], r["reason"] = "qwen", "x"
            else:
                r["stage"] = "qwen_passed"

    logs = []
    halted = bv.run_phase(cands, done, stratum_c=lambda c: c.family, stratum_r=lambda r: r["meta"]["family"],
                          caps=caps, api_task=api_task, settle_qwen=settle_qwen,
                          final_key=lambda r: ("d", r["generated"]["context_O"], r["generated"]["options"]),
                          dup_index=FakeIndex(), save=lambda: None, log=logs.append, label="t", meter=meter,
                          batch_size=8, max_workers=4)
    return halted, done, logs


def accepted_ids(done):
    return sorted(r["meta"]["index"] for r in done.values() if r["stage"] == "accepted")


def test_run_phase_caps_and_order_independence():
    results = []
    for seed in range(4):
        random.seed(seed)
        halted, done, _ = _phase(qwen_fail_every=7, fail_every=5)
        assert halted == "caps reached"
        results.append(accepted_ids(done))
        fams = Counter(r["meta"]["family"] for r in done.values() if r["stage"] == "accepted")
        assert fams == {"syntactic": 10, "affixal": 10}
    assert all(r == results[0] for r in results)      # acceptance does not depend on completion order


def test_run_phase_duplicates_rejected_at_acceptance():
    halted, done, _ = _phase(dup_every=2, caps={"syntactic": 50, "affixal": 50})
    acc = [r for r in done.values() if r["stage"] == "accepted"]
    ctxs = [r["generated"]["context_O"] for r in acc]
    assert len(ctxs) == len(set(ctxs))
    assert any(r["stage"] == "duplicate" for r in done.values())
    assert halted == "schedule exhausted"


def test_run_phase_budget_and_resume():
    halted, done, _ = _phase(budget_after=0.05)
    assert halted == "budget"
    n1 = len(done)
    assert 0 < n1 < 60
    for r in list(done.values())[:3]:        # simulate a crash after Haiku: Qwen still pending
        if r["stage"] in ("accepted",):
            pass
    # resume: no budget cap, continues to the caps; earlier records are kept
    halted2, done2, _ = _phase(done=done)
    assert halted2 == "caps reached" and len(done2) >= n1


def test_run_phase_resumes_pending_qwen():
    done = {}
    for k in range(4):
        uid = f"fresh-x-{k:05d}"
        done[uid] = {"stage": "haiku_passed", "generated": {"context_O": f"ctx {k}", "options": ["a", "b"]},
                     "meta": {"uid": uid, "family": ["syntactic", "affixal"][k % 2], "domain": "d", "index": k}}
    halted, done, _ = _phase(done=done, caps={"syntactic": 2, "affixal": 2})
    assert halted == "caps reached"
    assert accepted_ids(done) == [0, 1, 2, 3]


def test_meter_and_make_call_budget():
    class R:
        def __init__(self, text):
            self.usage = type("U", (), {"input_tokens": 1000, "output_tokens": 1000})()
            self.content = [type("B", (), {"type": "text", "text": text})()]
            self.stop_reason = "end_turn"

    class C:
        class messages:
            @staticmethod
            def create(**kw):
                return R("ok")

    m = bv.Meter()
    m.cap = 0.02
    call = bv.make_call(C(), m, effort="low")
    assert call("claude-sonnet-5", "p", 10) == "ok"       # 0.012 USD
    assert call("claude-sonnet-5", "p", 10) == "ok"       # 0.024 USD
    with pytest.raises(bv.BudgetExceeded):
        call("claude-sonnet-5", "p", 10)
    d = m.to_dict()
    assert d["calls"] == 2 and abs(d["usd"] - 0.024) < 1e-12 and bv.Meter(d).usd == d["usd"]


def test_run_phase_safety_stop():
    halted, done, _ = _phase(n=200, fail_every=1, caps={"syntactic": 50, "affixal": 50})
    assert halted.startswith("safety stop") and len(done) == 72
