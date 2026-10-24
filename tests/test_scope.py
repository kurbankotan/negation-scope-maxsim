"""Scope-resolver tests. The first test guards against the v0.1.0 infinite loop: walking up from a token in a
sentence whose root is not a verb must terminate (spaCy tokens are new objects on every access, so the root
must be found by index, not identity)."""
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FRAGMENTS = ["No exceptions.", "Nothing.", "No.", "Not at all!", "None of the above", "Unknown.",
             "Without water", "Never again, never.", "With the exception of Paris.", "", "   ", "No no no no."]


def _resolve_all(q):
    from cacsa2.scope import ScopeResolver, CueLexicon
    r = ScopeResolver(CueLexicon())
    for t in FRAGMENTS:
        r.resolve(t)
    r.prefetch(FRAGMENTS + ["He did not go home."])
    q.put("done")


def test_fragments_terminate_within_timeout():
    q = mp.Queue()
    p = mp.Process(target=_resolve_all, args=(q,))
    p.start()
    p.join(120)
    alive = p.is_alive()
    if alive:
        p.terminate()
    assert not alive, "scope resolver did not terminate on noun-rooted fragments"
    assert q.get(timeout=5) == "done"


def test_expected_scopes():
    from cacsa2.scope import ScopeResolver, CueLexicon
    r = ScopeResolver(CueLexicon())
    def neg(t):
        res = r.resolve(t)
        return [t[a:b] for a, b in res.negated_spans], [c[2] for c in res.cues]
    words, cues = neg("The park is not close, it is far away.")
    assert cues == ["not"] and "close" in words and "far" not in words and "park" not in words
    words, cues = neg("The dish is completely free of any nuts.")
    assert cues == ["free of"] and words == ["any", "nuts"]
    words, cues = neg("She is not only talented but also kind.")
    assert cues == [] and words == []
    words, cues = neg("He left without saying a word.")
    assert cues == ["without"] and words == ["saying", "a", "word"]


def test_polarity_mask_alignment():
    from cacsa2.scope import ScopeResolver, CueLexicon
    r = ScopeResolver(CueLexicon())
    text = "He did not go home."
    offsets = [(0, 2), (3, 6), (7, 10), (11, 13), (14, 18), (18, 19), (0, 0)]
    mask = r.polarity_mask(text, offsets)
    assert mask[0] == 0 and mask[2] == 0 and mask[3] == 1 and mask[4] == 1 and mask[5] == 0 and mask[6] == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
