"""Tests for the NegPairs construction filters (v1 defects and their v2 fixes), on real generator outputs from the
v1 build (tests/negpairs_build_fixtures.json). Run: python -m unittest tests.test_negpairs_build"""
import json
import unittest
from pathlib import Path

from cacsa2 import negpairs_build as nb

FX = json.loads((Path(__file__).parent / "negpairs_build_fixtures.json").read_text(encoding="utf-8"))


def v1(uid):
    return nb.structural_filter(FX[uid]["generated"], FX[uid]["family"])


def v2(uid):
    return nb.structural_filter_v2(FX[uid]["generated"], FX[uid]["family"])


class TestPromptsUnchanged(unittest.TestCase):
    def test_hashes(self):
        h = nb.prompt_hashes()
        self.assertEqual(h["GEN_PROMPT"], "8270c1684edaadff9941da5244a8812e7fcdc62b95ada62ea21bfa38e668af35")
        self.assertEqual(h["VERIFY_PROMPT"], "0bc5d895d7974a2db135412cb4661e0806916cd84469030c1c87b33a637a4742")
        self.assertIn("VERIFY_PROMPT_V2", h)


class TestWordMatching(unittest.TestCase):
    def test_has_word(self):
        self.assertFalse(nb.has_word("The notebook's spiral binding is broken.", "no"))
        self.assertFalse(nb.has_word("The afternoon slot matches her schedule.", "no"))
        self.assertTrue(nb.has_word("There is no milk left.", "no"))
        self.assertTrue(nb.has_word("None of the seats are free.", "none of"))
        self.assertFalse(nb.has_word("The store is open.", ""))

    def test_new_entities(self):
        self.assertEqual(nb.new_entities("All of Lena's workout clothes are clean.", "Lena's gym clothes are clean."), set())
        self.assertEqual(nb.new_entities("It is ready for Mara's talk.", "so Mara can use it"), set())
        self.assertEqual(nb.new_entities("above 8,000 feet", "above 8,000 feet"), set())
        self.assertEqual(nb.new_entities("She asked Tom for help.", "She asked for help."), {"tom"})
        self.assertEqual(nb.new_entities("It costs 45 dollars.", "It costs 40 dollars."), {"45"})


class TestV1DefectsFixedInV2(unittest.TestCase):
    def test_substring_cue_defect(self):
        for uid in ("qua-00002", "qua-00009"):          # "no" inside "notebook" / "afternoon"
            self.assertEqual(v1(uid), "cue_N already present in context_O")
            self.assertIsNone(v2(uid))

    def test_name_position_defect(self):
        for uid in ("qua-00000", "aff-00027", "qua-00039", "lex-00017"):   # sentence-initial or possessive names
            self.assertEqual(v1(uid), "context_P introduces a new name or number")
            self.assertIsNone(v2(uid))

    def test_wider_n_edit_limit(self):
        self.assertTrue(v1("lex-00001").startswith("context_N edit distance 5"))
        self.assertIsNone(v2("lex-00001"))                                   # "has ... in stock" -> "is out of"

    def test_kept_v1_item_still_passes(self):
        self.assertIsNone(v1("syn-00157"))
        self.assertIsNone(v2("syn-00157"))                                   # numbers with thousands separators


class TestLegitimateRejectionsKept(unittest.TestCase):
    def test_decisive_changed_in_S(self):
        for uid in ("lex-00013", "qua-00034"):
            self.assertEqual(v2(uid), "decisive words changed in context_S")

    def test_non_minimal_edits(self):
        self.assertEqual(v2("syn-00000"), "context_S edit too large")
        self.assertTrue(v2("qua-00014").startswith("context_N edit distance 8"))


class TestVerifierV2(unittest.TestCase):
    def test_expected_answers_and_early_stop(self):
        g = FX["qua-00000"]["generated"]
        y = g["answer_O"]; flip = "B" if y == "A" else "A"
        calls = []

        def stub(model, prompt, max_tokens):
            calls.append(prompt)
            ctx = prompt.split("CONTEXT: ")[1].split("\n")[0]
            ans = flip if ctx == g["context_N"] else y
            return json.dumps({"answer": ans, "natural": True})
        reason, verdicts = nb.verify_arms_v2(g, stub, "stub")
        self.assertIsNone(reason)
        self.assertEqual(len(calls), 4)

        def unclear_on_P(model, prompt, max_tokens):
            ctx = prompt.split("CONTEXT: ")[1].split("\n")[0]
            if ctx == g["context_P"]:
                return '{"answer": "unclear", "natural": true}'
            return json.dumps({"answer": flip if ctx == g["context_N"] else y, "natural": True})
        reason, verdicts = nb.verify_arms_v2(g, unclear_on_P, "stub")
        self.assertEqual(reason, "arm P read as 'unclear'")
        self.assertEqual(list(verdicts), ["O", "P"])


if __name__ == "__main__":
    unittest.main()
