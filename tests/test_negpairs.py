"""Tests for NegPairs helpers: decisive/cue spans, the dataset-supplied (oracle) mask, sentence diagnostics, paired
metrics, and the option-only control. Run: python -m unittest tests.test_negpairs"""
import unittest

import torch

from cacsa2.negpairs import (NegRow, decisive_token_spans, oracle_mask, report, arm_diagnostics, s_same_ids,
                             summarize_diagnostics, subset_accuracy)
from cacsa2.encoders import word_tokenize


def row(arm, context, cue="", decisive="", family="syntactic", sid="s1", label="A", question="What should Ana do?"):
    return NegRow(sample_id=f"{sid}-{arm}", passage_id=sid, question_id="q", edit_id={"O": 0, "P": 1, "S": 2, "N": 3}[arm],
                  label=label, cue=cue, arm=arm, family=family, seen_family=family in ("syntactic", "quantifier"),
                  domain="d", split="test", context=context, question=question, options=("buy bread there", "bake at home"),
                  decisive=decisive)


def marked_words(r):
    toks, offs = word_tokenize(r.text)
    m = oracle_mask(r, offs)
    return [t for t, k in zip(toks, m) if k]


class TestSpans(unittest.TestCase):
    def test_syntactic(self):
        r = row("N", "The bakery on Elm Street is not open today.", cue="not", decisive="is open")
        spans, cue = decisive_token_spans(r)
        self.assertEqual(r.text[cue[0]:cue[1]], "not")
        self.assertEqual([r.text[a:b] for a, b in spans], ["open"])

    def test_contraction_cue(self):
        r = row("N", "The bakery isn't open.", cue="n't", decisive="is open")
        _, cue = decisive_token_spans(r)
        self.assertIsNotNone(cue)
        self.assertEqual(r.text[cue[0]:cue[1]].lower(), "n't")

    def test_affixal_cue_contains_decisive(self):
        r = row("N", "The oven in unit 4 is powerless.", cue="powerless", decisive="has power", family="affixal")
        spans, cue = decisive_token_spans(r)
        self.assertEqual([r.text[a:b] for a, b in spans], ["powerless"])

    def test_question_never_marked(self):
        r = row("N", "The bakery is not open.", cue="not", decisive="is open", question="Is the bakery open now?")
        spans, _ = decisive_token_spans(r)
        self.assertTrue(all(b <= len(r.context) for _, b in spans))


class TestOracleMask(unittest.TestCase):
    def test_arms(self):
        self.assertEqual(marked_words(row("O", "The bakery is open.", decisive="is open")), [])
        self.assertEqual(marked_words(row("P", "The bakery has opened its doors.", decisive="is open")), [])
        self.assertEqual(marked_words(row("N", "The bakery is not open.", cue="not", decisive="is open")), ["not", "open"])
        self.assertEqual(marked_words(row("S", "The bakery, not the cafe, is open.", cue="not", decisive="is open")), ["not"])
        self.assertEqual(marked_words(row("N", "The oven has no power.", cue="no", decisive="has power", family="quantifier")),
                         ["no", "power"])
        self.assertEqual(marked_words(row("N", "The oven lacks power.", cue="lacks", decisive="has power", family="lexical")),
                         ["lacks", "power"])


class TestDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cacsa2.scope import ScopeResolver
        cls.res = ScopeResolver()

    def test_same_sentence_and_scope(self):
        rows = [row("N", "The bakery is not open. Ana has flour at home.", cue="not", decisive="is open", sid="a"),
                row("S", "The bakery is open. Ana's neighbour is not at home.", cue="not", decisive="is open", sid="b"),
                row("S", "The bakery that is not new is open.", cue="not", decisive="is open", sid="c"),
                row("O", "The bakery is open.", decisive="is open", sid="d")]
        d = arm_diagnostics(rows, self.res)
        self.assertTrue(d["a-N"]["cue_detected"])
        self.assertTrue(d["a-N"]["same_sentence"])
        self.assertTrue(d["a-N"]["decisive_in_scope"])
        self.assertFalse(d["b-S"]["same_sentence"])
        self.assertFalse(d["b-S"]["decisive_in_scope"])
        self.assertTrue(d["c-S"]["same_sentence"])
        self.assertFalse(d["d-O"]["any_cue_detected"])
        self.assertEqual(s_same_ids(d), ["c-S"])
        summ = summarize_diagnostics(d, by=("arm",))
        self.assertEqual(summ["S"]["n"], 2)
        self.assertEqual(summ["S"]["same_sentence"], 50.0)


class TestPairedMetrics(unittest.TestCase):
    def test_option_only_reader_scores_zero_on_paired_N(self):
        # an option-only reader predicts the same option in every arm of a scenario
        rows, recs = {}, []
        for k in range(10):
            y = "A" if k % 2 == 0 else "B"
            flip = "B" if y == "A" else "A"
            for arm, lab in (("O", y), ("P", y), ("S", y), ("N", flip)):
                r = row(arm, "ctx", sid=f"s{k}", label=lab)
                rows[r.sample_id] = r
                pred = y if k < 7 else flip          # 7 of 10 scenarios: the option bias matches the affirmative answer
                recs.append({"sample_id": r.sample_id, "label": lab, "pred": pred, "probs": [0.6, 0.4]})
        rep = report(recs, rows)["all"]
        self.assertAlmostEqual(rep["acc_OP"], 70.0)
        self.assertAlmostEqual(rep["acc_N"], 30.0)
        self.assertAlmostEqual(rep["negation_gap"], 40.0)      # bias alone produces a large raw gap ...
        self.assertAlmostEqual(rep["acc_N_given_O"], 0.0)     # ... the paired view does not
        self.assertAlmostEqual(rep["pair_ON"], 0.0)
        self.assertAlmostEqual(rep["acc_S_given_O"], 100.0)
        acc, n = subset_accuracy(recs, [r["sample_id"] for r in recs if r["sample_id"].endswith("-S")])
        self.assertEqual(n, 10)
        self.assertAlmostEqual(acc, 70.0)


class TestStrata(unittest.TestCase):
    def test_majority_vote_and_ids(self):
        from cacsa2.negpairs import option_prior_strata, strata_item_ids
        rows = {}
        for sid in ("s1", "s2"):
            for arm, lab in (("O", "A"), ("P", "A"), ("S", "A"), ("N", "B")):
                r = row(arm, "ctx", sid=sid, label=lab)
                rows[r.sample_id] = r
        runs = [[{"sample_id": "s1-O", "label": "A", "pred": "A"}, {"sample_id": "s2-O", "label": "A", "pred": "B"}],
                [{"sample_id": "s1-O", "label": "A", "pred": "A"}, {"sample_id": "s2-O", "label": "A", "pred": "B"}],
                [{"sample_id": "s1-O", "label": "A", "pred": "B"}, {"sample_id": "s2-O", "label": "A", "pred": "A"},
                 {"sample_id": "s2-N", "label": "B", "pred": "A"}]]          # non-O records are ignored
        st = option_prior_strata(runs, rows)
        self.assertEqual(st, {"s1": "congruent", "s2": "incongruent"})
        ids = strata_item_ids(st, rows.values())
        self.assertEqual(ids["prior_incongruent"], ["s2-N", "s2-O", "s2-P", "s2-S"])
        self.assertEqual(ids["seen_prior_incongruent"], ["s2-N", "s2-O", "s2-P", "s2-S"])
        self.assertEqual(ids["unseen_prior_congruent"], [])


class TestOptionOnly(unittest.TestCase):
    def test_context_blind(self):
        from cacsa2.controls import OptionOnlyScorer
        torch.manual_seed(0)
        m = OptionOnlyScorer(16).eval()
        c, cm = torch.randn(2, 2, 3, 16), torch.ones(2, 2, 3, dtype=torch.bool)
        a = m(torch.randn(2, 5, 16), torch.ones(2, 5, dtype=torch.bool), c, cm)
        b = m(torch.randn(2, 9, 16), torch.ones(2, 9, dtype=torch.bool), c, cm)
        self.assertTrue(torch.equal(a, b))



class TestFlipRules(unittest.TestCase):
    """Synthetic runs with known flip rates; checks per-run flip fields, the scenario bootstrap and the primary rules."""

    @staticmethod
    def make(n_seen=40, n_unseen=40):
        rows = {}
        fams = [("syntactic", True)] * n_seen + [("affixal", False)] * n_unseen
        for k, (fam, seen) in enumerate(fams):
            for arm, lab in (("O", "A"), ("P", "A"), ("S", "A"), ("N", "B")):
                cue = {"N": "not" if seen else "unavailable", "S": "not" if seen else "unhelpful"}.get(arm, "")
                r = NegRow(sample_id=f"s{k}-{arm}", passage_id=f"s{k}", question_id="q", edit_id=0, label=lab, cue=cue,
                           arm=arm, family=fam, seen_family=seen, domain="d", split="test", context="c", question="q?",
                           options=("x", "y"))
                rows[r.sample_id] = r
        return rows

    @staticmethod
    def make_run(rows, pN_seen, pS_seen, pN_unseen, pS_unseen, seed):
        recs = []
        for sid in sorted({r.passage_id for r in rows.values()}):
            k = int(sid[1:]); seen = rows[f"{sid}-O"].seen_family
            pN, pS = (pN_seen, pS_seen) if seen else (pN_unseen, pS_unseen)
            u = ((k * 37 + seed * 11) % 100) / 100.0           # deterministic pseudo-uniform per scenario and seed
            for arm in ("O", "P", "N", "S"):
                flip = (arm == "N" and u < pN) or (arm == "S" and u < pS)
                recs.append({"sample_id": f"{sid}-{arm}", "label": rows[f"{sid}-{arm}"].label, "pred": "B" if flip else "A",
                             "probs": [0.6, 0.4]})
        return recs

    def test_rules(self):
        from cacsa2 import np_analysis as npa
        rows = self.make()
        seeds = [1, 2, 3, 4, 5]
        spec = {("E0-hash", "max"): (0.40, 0.05, 0.03, 0.02), ("E0-hash", "signed_max"): (0.60, 0.05, 0.40, 0.05),
                ("E0-hash", "signed_max_sentence"): (0.60, 0.50, 0.40, 0.45)}
        runs = {(e, g, sd): self.make_run(rows, *p, sd) for (e, g), p in spec.items() for sd in seeds}
        s_same = sorted({r.passage_id for r in rows.values()})
        s_same_items = [f"{sid}-S" for sid in s_same]
        flat = npa.per_run(runs, rows, extra_subsets={"S_same": s_same_items}, s_same_scenarios=s_same)  # both, as in the notebook
        r0 = next(r for r in flat if r["aggregator"] == "max" and r["seed"] == 1)
        self.assertIsNotNone(r0["S_same_scen.flip_O_S"])        # regression: the S-only item subset must not overwrite it
        self.assertAlmostEqual(r0["seen.SS"], r0["seen.flip_O_N"] - r0["seen.flip_O_S"])
        self.assertIsNotNone(r0["samecue_not.flip_O_N"])
        self.assertEqual(r0["samecue_not.n_N"], 40)
        thr = {"gate_SS_seen": 20.0, "R1_max_unseen_SS": 5.0, "R2_min_gain_unseen_SS": 15.0, "R3_min_gain_flipS": 10.0,
               "R3_min_items": 50}
        res = npa.evaluate_rules_flip(flat, runs, rows, ["E0-hash"], thr, n_boot=500, s_same_scenarios=s_same)
        self.assertTrue(res["G"]["E0-hash"]["passed"])
        self.assertTrue(res["R1"]["E0-hash"]["met"])
        self.assertTrue(res["R2"]["E0-hash"]["met"])
        self.assertTrue(res["R3"]["E0-hash"]["met"])
        b = npa.cluster_bootstrap_flip(runs, rows, "E0-hash", "signed_max", "max", "SS", "unseen", None, 500)
        manual = (sum(x["unseen.SS"] for x in flat if x["aggregator"] == "signed_max")
                  - sum(x["unseen.SS"] for x in flat if x["aggregator"] == "max")) / 5
        self.assertAlmostEqual(b["point"], manual, places=6)
        self.assertLess(b["low"], b["point"]); self.assertGreater(b["high"], b["point"])
        # a model that never flips is not responsive: G fails and nothing is evaluated
        runs2 = {("E2-deberta", g, sd): self.make_run(rows, 0, 0, 0, 0, sd) for g in ("max", "signed_max", "signed_max_sentence") for sd in seeds}
        flat2 = npa.per_run(runs2, rows, s_same_scenarios=s_same)
        res2 = npa.evaluate_rules_flip(flat2, runs2, rows, ["E2-deberta"], thr, n_boot=200, s_same_scenarios=s_same)
        self.assertFalse(res2["G"]["E2-deberta"]["passed"])
        self.assertFalse(res2["R2"]["E2-deberta"]["met"])
        self.assertNotIn("E2-deberta", res2["R1"])

if __name__ == "__main__":
    unittest.main()
