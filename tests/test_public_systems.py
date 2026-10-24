"""Tests of cacsa2.public_systems with tiny local models (no network): every scorer class runs end to end on a few
pairs, the bge-m3 heads agree with an independent computation, the pointwise LLM score is the Yes/No log-odds, and
the metric glue reproduces the article's estimands on synthetic Perm rows."""
import json
import math
import os
import random
from pathlib import Path

import pytest
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from cacsa2 import negpairs as npx, nevir, public_systems as ps

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[unused0]", "[unused1]", "the", "park", "is", "not", "close", ",", "it", "far",
         "away", ".", "he", "did", "go", "home", "was", "unable", "to", "recover", "data", "no", "exceptions", "question", "answer", "yes",
         "true", "false", "shop", "open", "today", "and", "busy", "now", "what", "should", "ana", "do", "?", "stay", "context", ":", "proposed",
         "according", "correct", "query", "document", "relevant", "describe", "asks", "for", "a", "b", "which", "more", "<s>", "</s>", "user",
         "assistant", "system", "|", "careful", "reader", "exactly", "one", "word", "or", "this", "store", "sells", "nothing", "everything", "!",
         "team", "use", "main", "hall", "room", "storage", "but", "sufficient", "insufficient", "whole", "extra", "chairs", "space", "gathering"]
VOCAB += [f"w{i}" for i in range(40)]


def make_tok(vocab_file, **kw):
    """A WordPiece tokenizer over VOCAB (transformers 5 takes `vocab`, 4.x `vocab_file`); checked, not assumed."""
    from transformers import BertTokenizerFast
    try:
        tok = BertTokenizerFast(vocab=str(vocab_file), do_lower_case=True, **kw)
    except TypeError:
        tok = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True, **kw)
    if tok.encode("true the", add_special_tokens=False) != [VOCAB.index("true"), VOCAB.index("the")]:
        from transformers import PreTrainedTokenizerFast
        from tokenizers import BertWordPieceTokenizer
        tok = PreTrainedTokenizerFast(tokenizer_object=BertWordPieceTokenizer(vocab=str(vocab_file), lowercase=True), unk_token="[UNK]",
                                      pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", **kw)
    assert tok.encode("true the", add_special_tokens=False) == [VOCAB.index("true"), VOCAB.index("the")]
    return tok


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    """Tiny models sharing one WordPiece vocabulary: BERT (encoder), MLM head, 1-label classifier, T5, causal LM."""
    from transformers import (BertConfig, BertForMaskedLM, BertForSequenceClassification, BertModel, BertTokenizerFast,
                              GPT2Config, GPT2LMHeadModel, T5Config, T5ForConditionalGeneration)
    root = tmp_path_factory.mktemp("tiny")
    vocab_file = root / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    tok = make_tok(vocab_file)
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=len(VOCAB), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                     max_position_embeddings=128, pad_token_id=0)
    out = {}
    for key, cls in (("bert", BertModel), ("mlm", BertForMaskedLM), ("cls", BertForSequenceClassification)):
        d = root / key
        m = cls(cfg) if key != "cls" else cls(BertConfig(**{**cfg.to_dict(), "num_labels": 1}))
        m.eval().save_pretrained(d)
        tok.save_pretrained(d)
        out[key] = str(d)
    t5 = root / "t5"
    T5ForConditionalGeneration(T5Config(vocab_size=len(VOCAB), d_model=32, d_ff=64, num_layers=2, num_decoder_layers=2, num_heads=2, d_kv=16,
                                        decoder_start_token_id=0, pad_token_id=0, eos_token_id=3)).eval().save_pretrained(t5)
    tok.save_pretrained(t5)
    out["t5"] = str(t5)
    lm = root / "lm"
    GPT2LMHeadModel(GPT2Config(vocab_size=len(VOCAB), n_embd=32, n_layer=2, n_head=2, n_positions=256, bos_token_id=2, eos_token_id=3,
                               pad_token_id=0)).eval().save_pretrained(lm)
    tok2 = make_tok(vocab_file)
    tok2.chat_template = ("{% for m in messages %}<s>{{ m['role'] }} : {{ m['content'] }} </s>{% endfor %}"
                          "{% if add_generation_prompt %}<s>assistant :{% endif %}")
    tok2.save_pretrained(lm)
    out["lm"] = str(lm)
    m3 = root / "m3"
    BertModel(cfg).eval().save_pretrained(m3)
    make_tok(vocab_file, bos_token="[CLS]", eos_token="[SEP]").save_pretrained(m3)   # XLM-R-like special-token roles
    torch.manual_seed(1)
    torch.save(torch.nn.Linear(32, 32).state_dict(), m3 / "colbert_linear.pt")
    torch.save(torch.nn.Linear(32, 1).state_dict(), m3 / "sparse_linear.pt")
    out["m3"] = str(m3)
    return out


PAIRS = [("the shop is open", "the shop is open today and the park is busy"), ("the shop is not open", "the park is busy, the shop is open"),
         ("go home", "he did go home. he was unable to recover data"), ("stay", "the store sells nothing")]


def _check(scores, n):
    assert len(scores) == n and all(isinstance(s, float) and math.isfinite(s) for s in scores)


def test_colbert_checkpoint_with_metadata(tiny, monkeypatch):
    torch.manual_seed(3)
    W = torch.randn(16, 32)
    monkeypatch.setattr(ps, "load_colbert_projection", lambda repo, revision=None, filename="model.safetensors": (W, "linear.weight"))
    monkeypatch.setattr(ps, "load_colbert_metadata", lambda repo, revision=None: {"query_maxlen": 12, "doc_maxlen": 24, "dim": 16, "similarity": "cosine",
                                                                                  "attend_to_mask_tokens": False, "mask_punctuation": True})
    sc = ps.ColBERTCheckpoint(tiny["bert"], name="tiny-colbert", device="cpu")
    assert sc.settings["query_maxlen"] == 12 and sc.settings["doc_maxlen"] == 24 and sc.settings["dim"] == 16
    s = sc.score_pairs([q for q, _ in PAIRS], [d for _, d in PAIRS])
    _check(s, 4)
    # the same pairs scored through the NegPairs path use "<context> <question>" as the document
    s2 = sc.score_negpairs([("the shop is open", "the shop is open today and the park is busy", "")])
    assert abs(s2[0] - s[0]) < 1e-5
    # attend_to_mask_tokens changes the query states (the mask positions attend), mask_punctuation False keeps punctuation
    monkeypatch.setattr(ps, "load_colbert_metadata", lambda repo, revision=None: {"query_maxlen": 12, "doc_maxlen": 24, "dim": 16,
                                                                                  "attend_to_mask_tokens": True, "mask_punctuation": False})
    sc2 = ps.ColBERTCheckpoint(tiny["bert"], name="tiny-colbert-2", device="cpu")
    assert sc2.settings["attend_to_mask_tokens"] and not sc2.settings["mask_punctuation"] and sc2.native.skiplist == set()
    s3 = sc2.score_pairs([q for q, _ in PAIRS], [d for _, d in PAIRS])
    _check(s3, 4)
    assert any(abs(a - b) > 1e-6 for a, b in zip(s, s3))
    # a dim mismatch between the projection and the metadata is an error
    monkeypatch.setattr(ps, "load_colbert_metadata", lambda repo, revision=None: {"dim": 8})
    with pytest.raises(RuntimeError):
        ps.ColBERTCheckpoint(tiny["bert"], device="cpu")


def test_st_colbert_replicates_pylate_encoding(tiny):
    """A PyLate-format directory built from the tiny BERT: prefix tokens added, [MASK] expansion of queries to
    query_length, skiplist and padding removed from documents, Dense projection. (The replication was also checked
    against the PyLate library itself on identical weights: identical scores in three configurations.)"""
    import shutil
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    root = Path(tiny["bert"]).parent / "st_colbert"
    if not root.exists():
        shutil.copytree(tiny["bert"], root)
        (root / "modules.json").write_text(json.dumps([{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
                                                       {"idx": 1, "name": "1", "path": "1_Dense", "type": "pylate.models.Dense.Dense"}]))
        (root / "config_sentence_transformers.json").write_text(json.dumps({"model_type": "ColBERT", "query_prefix": "[Q] ", "document_prefix": "[D] ",
                                                                            "query_length": 12, "document_length": 24, "attend_to_expansion_tokens": False,
                                                                            "do_query_expansion": True}))
        (root / "1_Dense").mkdir()
        (root / "1_Dense" / "config.json").write_text(json.dumps({"in_features": 32, "out_features": 16, "bias": False,
                                                                 "activation_function": "torch.nn.modules.linear.Identity"}))
        torch.manual_seed(5)
        save_file({"linear.weight": torch.randn(16, 32)}, str(root / "1_Dense" / "model.safetensors"))
    sc = ps.STColBERT(str(root), device="cpu", name="tiny-st-colbert")
    assert sc.settings["query_length"] == 12 and sc.settings["document_length"] == 24 and sc.settings["dim"] == 16
    assert sc.settings["pad_token_for_expansion"] == "[MASK]" and sc.q_prefix_id != sc.d_prefix_id
    assert sc.q_prefix_id not in (None, sc.tok.unk_token_id) and sc.d_prefix_id not in (None, sc.tok.unk_token_id)
    qs = ["the shop is open", "go home!"]
    ds = ["the shop is open today and the park is busy.", "stay home? the shop is open today and the park is busy and busy and busy and busy and busy and busy and busy and busy"]
    Q = sc.encode(qs, True)
    D = sc.encode(ds, False)
    assert all(q.shape == (12, 16) for q in Q)                                   # every query vector takes part (expansion)
    tok = AutoTokenizer.from_pretrained(tiny["bert"])
    n_doc0 = len(tok(ds[0])["input_ids"]) + 1 - 1                                # tokens + prefix - the final '.' (skiplist)
    assert D[0].shape == (n_doc0, 16) and D[1].shape[0] <= 24 - 1 and sc.truncated_docs == 1
    assert all(abs(float(v.norm()) - 1) < 1e-4 for v in Q[0]) and all(abs(float(v.norm()) - 1) < 1e-4 for v in D[0])
    s = sc.score_pairs(qs * 2, ds + ds[::-1], batch_size=3)
    _check(s, 4)
    assert abs(s[0] - float((Q[0] @ D[0].T).max(dim=1).values.sum())) < 1e-4
    # the prefix sits right after the first token, and expansion tokens are [MASK] ids with attention 0
    ids, attn = sc._tokenize(qs[:1], True)
    assert ids[0, 1].item() == sc.q_prefix_id and ids.shape[1] == 12 and ids[0, -1].item() == sc.tok.mask_token_id and attn[0, -1].item() == 0
    assert ps.build_scorer({"name": "x", "type": "st_colbert", "repo": str(root)}, device="cpu").settings["dim"] == 16


def test_bgem3_heads_agree_with_independent_computation(tiny, monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, fn, revision=None: str(Path(repo) / fn))
    sc = ps.BGEM3(tiny["m3"], device="cpu", name="tiny-m3", max_length=64)
    qs, ds = [q for q, _ in PAIRS], [d for _, d in PAIRS]
    got = {h: sc.score_pairs(qs, ds, head=h) for h in sc.heads}
    for h in sc.heads:
        _check(got[h], 4)
    # independent computation, one text at a time
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tiny["m3"])
    model = AutoModel.from_pretrained(tiny["m3"]).eval()
    cl = torch.nn.Linear(32, 32); cl.load_state_dict(torch.load(Path(tiny["m3"]) / "colbert_linear.pt"))
    sl = torch.nn.Linear(32, 1); sl.load_state_dict(torch.load(Path(tiny["m3"]) / "sparse_linear.pt"))
    unused = {tok.cls_token_id, tok.sep_token_id, tok.pad_token_id, tok.unk_token_id, tok.eos_token_id}

    def rep(text):
        enc = tok(text, return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state[0]
            dense = torch.nn.functional.normalize(h[0], dim=-1)
            w = torch.relu(sl(h)).squeeze(-1)
            sp = {}
            for tid, ww in zip(enc["input_ids"][0].tolist(), w.tolist()):
                if ww > 0 and tid not in unused:
                    sp[tid] = max(sp.get(tid, 0.0), ww)
            cv = torch.nn.functional.normalize(cl(h[1:]), dim=-1)
        return dense, sp, cv

    for k, (q, d) in enumerate(PAIRS):
        qd, qs_, qc = rep(q)
        dd, ds_, dc = rep(d)
        assert abs(got["dense"][k] - float(qd @ dd)) < 1e-4
        assert abs(got["sparse"][k] - sum(w * ds_[t] for t, w in qs_.items() if t in ds_)) < 1e-4
        assert abs(got["colbert"][k] - float((qc @ dc.T).max(dim=1).values.sum() / qc.shape[0])) < 1e-3
    # the [SEP] vector takes part in the colbert head (reference behaviour), the [CLS] vector does not
    assert sc._cache[qs[0]][2].shape[0] == len(tok(qs[0])["input_ids"]) - 1
    sc.close()
    assert not sc._cache


def test_dense_sparse_cross_encoder_monot5(tiny):
    qs, ds = [q for q, _ in PAIRS], [d for _, d in PAIRS]
    dense = ps.SentenceDense(tiny["bert"], device="cpu", name="tiny-dense", query_prefix="query: ", doc_prefix="passage: ")
    s = dense.score_pairs(qs, ds)
    _check(s, 4)
    assert all(-1.0001 <= x <= 1.0001 for x in s)
    assert dense.settings["query_prefix"] == "query: "
    splade = ps.SpladeScorer(tiny["mlm"], device="cpu", name="tiny-splade", max_length=64)
    s = splade.score_pairs(qs, ds)
    _check(s, 4)
    assert all(x >= 0 for x in s) and all(len(v) > 0 for v in splade._cache.values())
    # dot product of the two log1p(relu) max-pooled vectors, computed independently for the first pair
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tiny["mlm"]); m = AutoModelForMaskedLM.from_pretrained(tiny["mlm"]).eval()
    def vec(t):
        enc = tok(t, return_tensors="pt")
        with torch.no_grad():
            return torch.log1p(torch.relu(m(**enc).logits[0])).max(dim=0).values
    assert abs(s[0] - float(vec(qs[0]) @ vec(ds[0]))) < 1e-4
    ce = ps.CrossEncoderScorer(tiny["cls"], device="cpu", name="tiny-ce", max_length=64)
    s = ce.score_pairs(qs, ds)
    _check(s, 4)
    assert ce.settings["num_labels"] == 1
    t5 = ps.MonoT5Scorer(tiny["t5"], device="cpu", name="tiny-monot5", max_length=64)
    s = t5.score_pairs(qs, ds, batch_size=3)
    _check(s, 4)
    assert all(x <= 0 for x in s)                                  # log-probabilities
    assert t5.settings["true_id"] == VOCAB.index("true") and t5.settings["false_id"] == VOCAB.index("false")
    # independent computation for the first pair: log softmax over {false, true} at the first decoder step
    from transformers import T5ForConditionalGeneration
    model = T5ForConditionalGeneration.from_pretrained(tiny["t5"]).eval()
    enc = tok(f"Query: {qs[0]} Document: {ds[0]} Relevant:", return_tensors="pt")
    with torch.no_grad():
        lg = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], decoder_input_ids=torch.tensor([[0]])).logits[0, 0]
    two = torch.log_softmax(lg[[VOCAB.index("false"), VOCAB.index("true")]], dim=-1)
    assert abs(s[0] - float(two[1])) < 1e-4


def test_llm_pointwise_log_odds_and_truncation(tiny):
    sc = ps.LLMPointwise(tiny["lm"], device="cpu", name="tiny-llm", max_doc_tokens=6)
    assert sc.yes_ids == [VOCAB.index("yes")] and sc.no_ids == [VOCAB.index("no")]
    items = [("go home", "he did go home. he was unable to recover data", "what should ana do?"), ("stay", "the store sells nothing", "what should ana do?")]
    s = sc.score_negpairs(items, batch_size=2)
    _check(s, 2)
    # independent computation with the same prompt, one at a time (left padding must not change the result)
    for k, (o, c, q) in enumerate(items):
        p = sc.prompt(sc.NEGPAIRS_USER.format(context=sc.truncate(c, 6), question=q, option=o))
        enc = sc.tok(p, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            lp = torch.log_softmax(sc.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits[0, -1].float(), dim=-1)
        assert abs(s[k] - float(lp[VOCAB.index("yes")] - lp[VOCAB.index("no")])) < 1e-4
    assert len(sc.tok.encode(sc.truncate("he did go home. he was unable to recover data", 6), add_special_tokens=False)) == 6
    assert sc.settings["system"] is not None and "<s>system" in sc.prompt("x")
    s2 = sc.score_nevir([("the shop is open", "the shop is open today and the park is busy")])
    _check(s2, 1)


def test_api_pairwise_scores_and_cap():
    def call(model, system, user):
        # prefers the document that contains the query word; order-independent except for query q2 of instance i2
        a = user.split("Document A: ")[1].split("\n\nDocument B: ")[0]
        q = user.split("Query: ")[1].split("\n\nDocument A")[0]
        if q == "second query of i2":
            return "A", 100, 1                                        # always the first shown: order-dependent -> tie
        return ("A" if q in a else "B"), 100, 1
    inst = [{"id": "i1", "q1": "open", "q2": "busy", "doc1": "the shop is open", "doc2": "the shop is busy"},
            {"id": "i2", "q1": "open", "q2": "second query of i2", "doc1": "the park is open", "doc2": "the shop is busy"}]
    sc = ps.ClaudePairwiseNevIR("m", call, (1.0, 5.0), cap_usd=1.0)
    out = sc.score_instances(inst)
    assert out["i1"]["q1d1"] == 2 and out["i1"]["q1d2"] == 0 and out["i1"]["q2d2"] == 2 and out["i2"]["q2d1"] == 1 and out["i2"]["q2d2"] == 1
    assert out["i2"]["q2_answers"] == ["A", "A"]
    assert sc.calls == 8 and abs(sc.usd - 8 * (100 * 1.0 + 1 * 5.0) / 1e6) < 1e-12
    m = ps.nevir_metrics(out, inst, ["i1"], [], n_boot=50)
    assert m["all"]["paired_accuracy"] == 50.0 and m["instances_with_tie"] == 1 and m["nested"]["paired_accuracy"] == 100.0
    # resume: finished instances are not judged again; the cap stops the run
    sc.calls = 0
    out2 = sc.score_instances(inst, done=out)
    assert out2 == out and sc.calls == 0
    sc2 = ps.ClaudePairwiseNevIR("m", call, (1.0, 5.0), cap_usd=0.0)
    with pytest.raises(RuntimeError):
        sc2.score_instances(inst)


def perm_rows(n_scen=12):
    rows = []
    for k in range(n_scen):
        first = k % 2 == 0
        fam = "syntactic" if k % 4 < 2 else "affixal"
        dec, oth = f"the shop{k} is open today", f"the park{k} is busy"
        def ctx(cd, co):
            s = f"{cd}, and {co}." if first else f"{co}, and {cd}."
            return s[0].upper() + s[1:]
        O, N, S = ctx(dec, oth), ctx(dec.replace("is open", "is not open"), oth), ctx(dec, oth.replace("is busy", "is not busy"))
        for arm, c, lab in (("O", O, 0), ("P", O.replace("today", "now"), 0), ("N", N, 1), ("S", S, 0)):
            rows.append(npx.NegRow(sample_id=f"p{k}-{arm}", passage_id=f"p{k}", question_id="q", edit_id=npx.ARM_CODE[arm], label="AB"[lab],
                                   cue="not" if arm in "NS" else "", arm=arm, family=fam, seen_family=fam == "syntactic", domain="d", split="perm",
                                   context=c, question="What should Ana do?", options=("go to the shop", "stay"), decisive=f"shop{k} is open",
                                   perm={"pattern": "not-insertion", "decisive_first": first, "spans": {}}))
    return rows


def test_metric_glue_on_synthetic_rows():
    rows = perm_rows(16)
    by = {r.sample_id: r for r in rows}
    # scope-aware scores: option B wins on N only; a tie on one O item (first option on a tie)
    scores = {r.sample_id: ([0.2, 0.9] if r.arm == "N" else [0.9, 0.2]) for r in rows}
    scores["p0-O"] = [0.5, 0.5]
    recs = ps.negpairs_records(rows, scores)
    assert sum(r["tie"] for r in recs) == 1 and recs[0]["pred"] == "A" and all(r["pred"] == r["label"] for r in recs)
    m = ps.perm_metrics(recs, by, n_boot=100)
    assert m["flat"]["all.SS_bal"] == 100 and m["flat"]["all.pooled.joint_NS_acc"] == 100 and m["ties"] == 1
    assert m["SS_bal_all_ci"]["point"] == 100 and m["SS_bal_all_ci"]["n_clusters"] == [8, 8]
    # a bag-invariant scorer: identical N and S decisions -> SS 0
    scores = {r.sample_id: [0.9, 0.2] for r in rows}
    m0 = ps.perm_metrics(ps.negpairs_records(rows, scores), by, n_boot=100)
    assert m0["flat"]["all.SS_bal"] == 0 and m0["flat"]["all.pooled.flip_O_N"] == 0
    f = ps.fresh_metrics(ps.negpairs_records(rows, scores), by, n_boot=100)
    assert f["report"]["all"]["SS"] == 0 and f["SS_seen_ci"]["point"] == 0
    table = ps.perm_table([{"name": "x", "kind": "dense", "perm": m}, {"name": "y", "kind": "sparse", "perm": m0}])
    assert table.count("\n") == 3 and "| x | dense |" in table and "100.0" in table
    ft = ps.fresh_table([{"name": "x", "kind": "dense", "fresh": f}])
    assert "| x | dense |" in ft
    # NevIR glue: strict comparisons, ties are errors, equal-bag subset reported
    inst = [{"id": f"n{k}", "q1": "a", "q2": "b", "doc1": "d1", "doc2": "d2"} for k in range(6)]
    sc = {f"n{k}": {"q1d1": 1.0, "q1d2": 0.0, "q2d1": 0.0, "q2d2": 1.0} for k in range(6)}
    sc["n5"] = {"q1d1": 1.0, "q1d2": 1.0, "q2d1": 0.0, "q2d2": 1.0}
    nm = ps.nevir_metrics(sc, inst, ["n0", "n1"], ["n1"], n_boot=50)
    assert abs(nm["all"]["paired_accuracy"] - 500 / 6) < 1e-9 and nm["nested"]["paired_accuracy"] == 100 and nm["equal_bags"]["paired_accuracy"] == 100
    assert nm["instances_with_tie"] == 1
    nt = ps.nevir_table([{"name": "x", "kind": "dense", "nevir": nm}])
    assert "| x | dense | 83.3 |" in nt
    anchor = ps.anchor_against_archived({"i": [1.0, 0.5]}, {"i": [1.0, 0.4]}, sc, sc)
    assert anchor["negpairs_decision_agreement_pct"] == 100 and abs(anchor["negpairs_max_abs_diff"] - 0.1) < 1e-12 and anchor["nevir_comparison_agreement_pct"] == 100


def test_build_scorer_unknown_type():
    with pytest.raises(ValueError):
        ps.build_scorer({"name": "x", "type": "nope"})
