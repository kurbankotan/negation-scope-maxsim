"""Tests of cacsa2.finetune_li on the tiny ColBERT fixture: the trainer's scores equal the evaluation scorer's scores
before any update, a few steps reduce the loss and change the selected state, selection and patience work on both
training sets, nDCG@10 is right on a hand-made ranking, and the brute-force SciFact retrieval runs."""
import math
import random
from pathlib import Path

import pytest
import torch

from cacsa2 import finetune_li as ft, negpairs as npx, public_systems as ps
from tests.test_public_systems import VOCAB, tiny, make_tok  # noqa: F401  (fixture)


@pytest.fixture
def tiny_colbert(tiny, monkeypatch):
    torch.manual_seed(3)
    W = torch.randn(16, 32)
    monkeypatch.setattr(ps, "load_colbert_projection", lambda repo, revision=None, filename="model.safetensors": (W.clone(), "linear.weight"))
    monkeypatch.setattr(ps, "load_colbert_metadata", lambda repo, revision=None: {"query_maxlen": 12, "doc_maxlen": 24, "dim": 16})
    return ps.ColBERTCheckpoint(tiny["bert"], name="tiny-colbert", device="cpu")


def rows(n=12):
    out = []
    for k in range(n):
        dec = f"the shop{k % 3} is open today"
        for arm, ctx, lab in (("O", f"{dec}, and the park is busy.", 0), ("P", f"{dec}, and the park is busy now.", 0),
                              ("N", f"the shop{k % 3} is not open today, and the park is busy.", 1), ("S", f"{dec}, and the park is not busy.", 0)):
            out.append(npx.NegRow(sample_id=f"p{k}-{arm}", passage_id=f"p{k}", question_id="q", edit_id=npx.ARM_CODE[arm], label="AB"[lab], cue="not" if arm in "NS" else "",
                                  arm=arm, family="syntactic", seen_family=True, domain="d", split="train", context=ctx, question="what should ana do?",
                                  options=("go to the shop", "stay home"), decisive=dec, perm=None))
    return out


def instances(n=6):
    return [{"id": f"n{k}", "q1": f"shop{k % 3} open", "q2": f"park busy", "doc1": f"the shop{k % 3} is open today.", "doc2": "the park is busy and the shop is not open."} for k in range(n)]


def test_trainer_matches_scorer_and_learns(tiny_colbert):
    sc = tiny_colbert
    qs, ds = ["go to the shop", "stay home"], ["the shop0 is open today, and the park is busy. what should ana do?"] * 2
    ref = sc.score_pairs(qs, ds)
    tr = ft.ColBERTTrainer(sc, lr=1e-3, total_steps=10)
    tr.model.eval()
    with torch.no_grad():
        got = tr.scores(qs, ds).tolist()
    assert all(abs(a - b) < 1e-4 for a, b in zip(got, ref))                 # same tokenization, markers, masking, projection
    data = rows(8)
    losses = []
    for _ in range(6):
        tr.model.train()
        loss = ft.negpairs_loss(tr, data[:16])
        losses.append(float(loss))
        tr.step(loss)
    assert losses[-1] < losses[0]
    tr.sync_eval()
    assert any(abs(a - b) > 1e-6 for a, b in zip(sc.score_pairs(qs, ds), ref))   # the evaluation scorer sees the trained weights
    sd = tr.state()
    assert "__projection__" in sd and sd["__projection__"].shape == (16, 32)
    loss_n = ft.nevir_loss(tr, instances(3))
    assert math.isfinite(float(loss_n)) and loss_n.requires_grad


def test_finetune_selection_and_patience(tiny_colbert):
    sc = tiny_colbert
    train, dev = rows(10), rows(4)
    s = ft.finetune(sc, "negpairs", train, dev, seed=1, lr=1e-3, epochs=4, batch=8, patience=1, verbose=False)
    assert s["selected_epoch"] >= 0 and len(s["history"]) >= 2 and s["n_train"] == 40 and s["n_dev"] == 16
    assert s["selected_dev"] >= s["dev_before"] and len(s["state_sha256"]) == 64 and s["parameters"] > 0
    assert len(s["history"]) <= 1 + 4
    acc = ft.negpairs_accuracy(sc, dev)
    assert abs(acc - s["selected_dev"]) < 1e-9                                 # the selected state is loaded
    s2 = ft.finetune(sc, "nevir", instances(8), instances(3), seed=2, lr=1e-3, epochs=2, batch=4, patience=2, verbose=False, max_steps=3)
    assert s2["train_set"] == "nevir" and s2["steps"] == 3 and 0 <= s2["selected_dev"] <= 100


def test_ndcg_and_scifact_bruteforce(tiny_colbert):
    assert ft.ndcg_at_k(["a", "b", "c"], {"a": 1}, 10) == 1.0
    assert abs(ft.ndcg_at_k(["b", "a", "c"], {"a": 1}, 10) - 1 / math.log2(3)) < 1e-12
    assert ft.ndcg_at_k(["b", "c"], {"a": 1}, 10) == 0.0 and ft.ndcg_at_k(["a"], {}, 10) == 0.0
    two = ft.ndcg_at_k(["a", "x", "b"], {"a": 1, "b": 1}, 10)
    assert abs(two - (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))) < 1e-12
    docs = [{"id": f"d{k}", "text": f"the shop{k % 4} is open today and the park is busy w{k}"} for k in range(20)]
    queries = [{"id": "q1", "text": "shop1 open"}, {"id": "q2", "text": "park busy w7"}]
    rel = {"q1": {"d1": 1, "d5": 1}, "q2": {"d7": 1}}
    out = ft.scifact_ndcg(tiny_colbert, queries, docs, rel, k=10, batch_docs=7, batch_queries=1)
    assert out["n_queries"] == 2 and out["n_docs"] == 20 and 0 <= out["ndcg@10"] <= 100 and set(out["per_query"]) == {"q1", "q2"}
    # brute force agrees with pairwise scoring for the top document
    nat = tiny_colbert.native
    s = tiny_colbert.score_pairs([queries[0]["text"]] * 20, [d["text"] for d in docs])
    best = docs[max(range(20), key=lambda i: s[i])]["id"]
    out2 = ft.scifact_ndcg(tiny_colbert, queries[:1], docs, {"q1": {best: 1}}, k=1, batch_docs=20, batch_queries=1)
    assert out2["ndcg@10"] == 100.0
