"""NegPairs: loading, CACSA items, metrics, the float64 Proposition-1 test on minimal pairs, and a fine-tuned
multiple-choice reference.

Formulation (as in the archived CACSA v2 notebook): the context and the question are encoded jointly
("<context> <question>"), each answer option is encoded separately, and

    score(option) = mean over the option's tokens j of  A_i c_ij,   c_ij = e_ij + lambda * tanh(aff_ij)

with A the aggregator over context tokens i. Labels are "A" (first option) and "B" (second option).
Arm codes in the record field `edit_id`: O = 0, P = 1, S = 2, N = 3.
"""
from __future__ import annotations

import json
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .encoders import EncodedStore, word_tokenize
from .scope import ScopeResolver
from .train import Item

LABELS2 = ("A", "B")
ARM_CODE = {"O": 0, "P": 1, "S": 2, "N": 3}
ARMS = ("O", "P", "S", "N")


@dataclass(frozen=True)
class NegRow:
    sample_id: str          # item_id
    passage_id: str         # scenario_id (the resampling cluster)
    question_id: str
    edit_id: int            # arm code
    label: str              # "A" | "B"
    cue: str
    arm: str
    family: str
    seen_family: bool
    domain: str
    split: str
    context: str
    question: str
    options: Tuple[str, str]
    decisive: str = ""      # the words of context_O that decide the answer (generator field; diagnostics and oracle only)
    perm: Optional[dict] = None   # NegPairs-Perm only: pattern, clause order and character spans of this arm (manifest v3.0, 4.3)

    @property
    def text(self) -> str:
        return f"{self.context.strip()} {self.question.strip()}"


def load_split(data_dir: str | Path, split: str) -> List[NegRow]:
    rows = []
    with open(Path(data_dir) / f"{split}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(NegRow(sample_id=r["item_id"], passage_id=r["scenario_id"], question_id="q", edit_id=ARM_CODE[r["arm"]],
                               label=LABELS2[int(r["label"])], cue=r.get("cue", ""), arm=r["arm"], family=r["family"],
                               seen_family=bool(r["seen_family"]), domain=r["domain"], split=r["split"], context=r["context"],
                               question=r["question"], options=tuple(r["options"]), decisive=r.get("decisive", ""),
                               perm=r.get("perm")))
    return rows


# ------------------------------------------------------------------ cue, decisive words and sentences
_WORDS = re.compile(r"[A-Za-z0-9']+")
_STOP = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "am", "to", "of", "in", "on", "at", "for",
         "and", "or", "it", "this", "that", "these", "those", "with", "has", "have", "had", "will", "would", "can", "could",
         "should", "his", "her", "their", "its", "my", "our", "your", "by", "as", "from", "there", "here", "all", "any",
         "some", "now", "today", "still", "also", "very", "just", "so", "than", "then", "does", "do", "did"}


def find_span(text: str, phrase: str, limit: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """First case-insensitive occurrence of `phrase` in text[:limit]; word-bounded if possible (else plain, for "n't")."""
    phrase = (phrase or "").strip().strip('.,;:!?"')
    if not phrase:
        return None
    hay = text if limit is None else text[:limit]
    m = re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", hay, re.I)
    if m:
        return m.start(), m.end()
    k = hay.lower().find(phrase.lower())
    return (k, k + len(phrase)) if k >= 0 else None


def decisive_token_spans(row: NegRow) -> Tuple[List[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """Character spans (in row.text, context part only) of the decisive content words, and the span of the declared cue.

    A word counts as decisive if it is a content word of `row.decisive`, or if it is the cue word itself and contains a
    decisive content word (affixal and fused cues: power -> powerless, available -> unavailable)."""
    ctx_len = len(row.context.strip())
    dec_words = {w.lower() for w in _WORDS.findall(row.decisive)} - _STOP
    cue = find_span(row.text, row.cue, ctx_len) if row.cue else None
    spans = []
    for m in _WORDS.finditer(row.text[:ctx_len]):
        w = m.group(0).lower()
        if w in dec_words:
            spans.append((m.start(), m.end()))
        elif cue and m.start() < cue[1] and cue[0] < m.end() and any(len(d) >= 3 and d in w for d in dec_words):
            spans.append((m.start(), m.end()))
    return spans, cue


def perm_decisive_spans(row: NegRow) -> Tuple[List[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """NegPairs-Perm: decisive content words restricted to the decisive-clause span, and the cue span, both read from
    the stored `perm.spans` of this arm (manifest v3.0, section 4.3). The shared word W occurs in both clauses, so the
    whole-context search of `decisive_token_spans` would mark the unedited W of the other clause."""
    sp = row.perm["spans"]
    a, b = sp["clause_decisive"]
    dec_words = {w.lower() for w in _WORDS.findall(row.decisive)} - _STOP
    spans = [(m.start(), m.end()) for m in _WORDS.finditer(row.context[:b]) if m.start() >= a and m.group(0).lower() in dec_words]
    cue = tuple(sp["cue"]) if sp.get("cue") else None
    if cue and row.arm == "N" and not any(s <= cue[0] < e for s, e in spans):
        spans.append(cue)                      # the negated form of W counts as a decisive word
    return spans, cue


def oracle_mask(row: NegRow, offsets: Sequence[Tuple[int, int]]) -> List[int]:
    """Dataset-supplied polarity for the descriptive upper bound (aggregator directory `signed_max_oracle`):
    N arm: the decisive words and the cue; S arm: the cue only; O and P arms: nothing. It reads the arm identity and the
    generator's `decisive` field, so it is a diagnostic of what exact scope would allow, not a method. NegPairs-Perm
    rows use the stored clause spans (perm_decisive_spans)."""
    if row.arm in ("O", "P"):
        return [0] * len(offsets)
    dec, cue = perm_decisive_spans(row) if row.perm else decisive_token_spans(row)
    spans = ([cue] if cue else []) + (dec if row.arm == "N" else [])
    return [1 if (e > s and any(a < e and s < b for a, b in spans)) else 0 for (s, e) in offsets]


def arm_diagnostics(rows: Sequence[NegRow], resolver: ScopeResolver) -> Dict[str, dict]:
    """Per item: does the resolver detect the declared cue, does its scope cover the decisive words, and does the cue
    share a sentence with the decisive words (spaCy sentences)? Descriptive; never used for training or selection.
    For O and P items only false-positive detection is recorded."""
    resolver.prefetch([r.text for r in rows])
    out = {}
    for r in rows:
        res = resolver.resolve(r.text)
        d = {"arm": r.arm, "family": r.family, "seen_family": r.seen_family,
             "any_cue_detected": bool(res.cues), "any_scope": bool(res.negated_spans)}
        if r.arm in ("N", "S"):
            dec, cue = perm_decisive_spans(r) if r.perm else decisive_token_spans(r)
            d["cue_found"] = cue is not None
            d["cue_detected"] = bool(cue) and any(s < cue[1] and cue[0] < e for (s, e, _, _) in res.cues)
            d["decisive_found"] = bool(dec)
            d["decisive_in_scope"] = any(res.word_in_scope(a, b) for a, b in dec) if dec else None
            if cue and dec:
                sent = next(((a, b) for a, b in res.sentences if a <= cue[0] < b), None)
                d["same_sentence"] = bool(sent) and any(sent[0] <= a < sent[1] for a, _ in dec)
            else:
                d["same_sentence"] = None
            if r.perm:                        # Perm diagnostics from the stored clause spans (manifest v3.0, 5.5)
                sp = r.perm["spans"]
                da, db = sp["clause_decisive"]
                oa, ob = sp["clause_other"]
                words_in = lambda a, b: [(m.start(), m.end()) for m in _WORDS.finditer(r.context[:b]) if m.start() >= a]
                d["scope_in_decisive_clause"] = any(res.word_in_scope(s, e) for s, e in words_in(da, db))
                d["scope_in_other_clause"] = any(res.word_in_scope(s, e) for s, e in words_in(oa, ob))
                d["decisive_first"] = bool(r.perm.get("decisive_first"))
        out[r.sample_id] = d
    return out


def summarize_diagnostics(diag: Dict[str, dict], by: Sequence[str] = ("arm",)) -> Dict[str, dict]:
    groups = defaultdict(list)
    for d in diag.values():
        groups["/".join(str(d[k]) for k in by)].append(d)
    out = {}
    for key, ds in sorted(groups.items()):
        rec = {"n": len(ds)}
        for f in ("any_cue_detected", "any_scope", "cue_found", "cue_detected", "decisive_found", "decisive_in_scope",
                  "same_sentence", "scope_in_decisive_clause", "scope_in_other_clause"):
            vals = [d[f] for d in ds if d.get(f) is not None]
            if vals:
                rec[f] = round(100.0 * sum(vals) / len(vals), 1)
                rec[f + "_n"] = len(vals)
        out[key] = rec
    return out


def s_same_ids(diag: Dict[str, dict]) -> List[str]:
    """S items whose cue shares a sentence with the decisive words: the items on which a scope-blind sentence-wide
    mask and a dependency scope can disagree (subset of rule R3)."""
    return sorted(k for k, d in diag.items() if d["arm"] == "S" and d.get("same_sentence") is True)


def build_items(rows: Sequence[NegRow], store: EncodedStore, resolver: Optional[ScopeResolver], progress: bool = False,
                mask_mode: str = "scope") -> List[Item]:
    """mask_mode: "scope" (dependency scope, A4/A5 runs), "sentence" (sentence-wide, A6), "oracle" (dataset-supplied,
    descriptive upper bound; needs no resolver)."""
    if mask_mode not in ("scope", "sentence", "oracle"):
        raise ValueError(mask_mode)
    texts = [r.text for r in rows]
    opts = sorted({o for r in rows for o in r.options})
    store.get_or_encode(texts, progress=progress)
    store.get_or_encode(opts, progress=progress)
    if resolver is not None and mask_mode != "oracle":
        resolver.prefetch(texts, progress=progress)
    items = []
    for r in rows:
        ctx = store.get(r.text)
        if mask_mode == "oracle":
            pol = torch.tensor(oracle_mask(r, ctx.offsets), dtype=torch.uint8)
        elif resolver is not None:
            pol = torch.tensor(resolver.polarity_mask(r.text, ctx.offsets, mode=mask_mode), dtype=torch.uint8)
        else:
            pol = torch.zeros(len(ctx.offsets), dtype=torch.uint8)
        items.append(Item(row=r, ctx=ctx.states, cands=[store.get(o).states for o in r.options], polarity=pol,
                          label=LABELS2.index(r.label), unseen_cue=not r.seen_family))
    return items


# ----------------------------------------------------------------------------------------------- metrics
def _acc(recs: Sequence[dict]) -> Optional[float]:
    return (100.0 * sum(r["pred"] == r["label"] for r in recs) / len(recs)) if recs else None


def report(records: Sequence[dict], rows_by_id: Dict[str, NegRow]) -> dict:
    """Arm-level accuracy, flip rates relative to O, scenario consistency, gaps; overall and by seen/unseen family."""
    def block(recs):
        by_arm = defaultdict(list)
        scen = defaultdict(dict)
        for r in recs:
            row = rows_by_id[r["sample_id"]]
            by_arm[row.arm].append(r)
            scen[row.passage_id][row.arm] = r
        out = {"n_items": len(recs), "acc": _acc(recs)}
        for arm in ARMS:
            out[f"acc_{arm}"] = _acc(by_arm[arm])
        for arm in ("P", "S", "N"):
            pairs = [(v["O"], v[arm]) for v in scen.values() if "O" in v and arm in v]
            out[f"flip_O_{arm}"] = (100.0 * sum(a["pred"] != b["pred"] for a, b in pairs) / len(pairs)) if pairs else None
            # paired view: among scenarios whose O item is answered correctly, the share whose `arm` item is also correct.
            # An option-only or negation-blind reader scores 0 on N here, whatever its option bias.
            ok = [b for a, b in pairs if a["pred"] == a["label"]]
            out[f"acc_{arm}_given_O"] = (100.0 * sum(b["pred"] == b["label"] for b in ok) / len(ok)) if ok else None
        pairs_n = [(v["O"], v["N"]) for v in scen.values() if "O" in v and "N" in v]
        out["pair_ON"] = (100.0 * sum(a["pred"] == a["label"] and b["pred"] == b["label"] for a, b in pairs_n) / len(pairs_n)
                          if pairs_n else None)
        full = [v for v in scen.values() if all(a in v for a in ARMS)]
        out["consistency"] = (100.0 * sum(all(v[a]["pred"] == v[a]["label"] for a in ARMS) for v in full) / len(full)) if full else None
        out["n_scenarios"] = len(full)
        aff = [x for x in (out["acc_O"], out["acc_P"]) if x is not None]
        out["acc_OP"] = (sum(aff) / len(aff)) if aff else None
        out["negation_gap"] = (out["acc_OP"] - out["acc_N"]) if (out["acc_OP"] is not None and out["acc_N"] is not None) else None
        out["scope_gap"] = (out["acc_OP"] - out["acc_S"]) if (out["acc_OP"] is not None and out["acc_S"] is not None) else None
        # scope sensitivity (primary estimand of manifest v2.0): decision changes caused by the cue on the decisive
        # words minus decision changes caused by a cue of the same family on an irrelevant detail
        out["SS"] = (out["flip_O_N"] - out["flip_O_S"]) if (out["flip_O_N"] is not None and out["flip_O_S"] is not None) else None
        wrong_n = [r for r in by_arm["N"] if r["pred"] != r["label"]]
        out["N_error_margin_mean"] = (sum(abs(r["probs"][0] - r["probs"][1]) for r in wrong_n) / len(wrong_n)) if wrong_n else None
        out["N_error_frac_margin_gt_0.9"] = (sum(abs(r["probs"][0] - r["probs"][1]) > 0.9 for r in wrong_n) / len(wrong_n)) if wrong_n else None
        return out

    seen = [r for r in records if rows_by_id[r["sample_id"]].seen_family]
    unseen = [r for r in records if not rows_by_id[r["sample_id"]].seen_family]
    fams = sorted({rows_by_id[r["sample_id"]].family for r in records})
    return {"all": block(records), "seen": block(seen) if seen else None, "unseen": block(unseen) if unseen else None,
            "by_family": {f: block([r for r in records if rows_by_id[r["sample_id"]].family == f]) for f in fams}}


def subset_accuracy(records: Sequence[dict], ids: Iterable[str]) -> Tuple[Optional[float], int]:
    keep = set(ids)
    sel = [r for r in records if r["sample_id"] in keep]
    return _acc(sel), len(sel)


def option_prior_strata(c2_record_lists: Sequence[Sequence[dict]], rows_by_id: Dict[str, NegRow]) -> Dict[str, str]:
    """Scenario strata from the context-blind control C2. For each scenario, the O-item predictions of the given C2 runs
    are pooled by majority vote (ties count as congruent): "congruent" if the option-only prior picks the affirmative
    answer, "incongruent" otherwise. In incongruent scenarios the affirmative arms can only be answered by reading the
    context, and the option prior favours the N answer."""
    votes = defaultdict(lambda: [0, 0])
    for recs in c2_record_lists:
        for r in recs:
            row = rows_by_id[r["sample_id"]]
            if row.arm == "O":
                votes[row.passage_id][0 if r["pred"] == r["label"] else 1] += 1
    return {sid: ("congruent" if v[0] >= v[1] else "incongruent") for sid, v in votes.items()}


def strata_item_ids(strata: Dict[str, str], rows: Iterable[NegRow]) -> Dict[str, List[str]]:
    """Item ids per option-prior stratum, overall and within seen / unseen negation families."""
    out = defaultdict(list)
    for r in rows:
        s = strata.get(r.passage_id)
        if s is not None:
            out[f"prior_{s}"].append(r.sample_id)
            out[f"{'seen' if r.seen_family else 'unseen'}_prior_{s}"].append(r.sample_id)
    keys = [f"{a}prior_{b}" for a in ("", "seen_", "unseen_") for b in ("congruent", "incongruent")]
    return {k: sorted(out.get(k, [])) for k in keys}


# ------------------------------------------------------------------------- Proposition-1 on minimal pairs
def insertion_pairs(rows: Sequence[NegRow]) -> List[Tuple[NegRow, NegRow]]:
    """(O, X) pairs of the same scenario where X's word tokens contain O's as a sub-multiset (X = O plus inserted
    tokens). Proposition 1 then predicts score_X(o) >= score_O(o) for every option under a context-independent encoder
    and a token-addition-monotone aggregator."""
    by_s = defaultdict(dict)
    for r in rows:
        by_s[r.passage_id][r.arm] = r
    out = []
    for v in by_s.values():
        if "O" not in v:
            continue
        o_tokens = Counter(t.lower() for t in word_tokenize(v["O"].text)[0])
        for arm in ("S", "N"):
            if arm in v:
                x_tokens = Counter(t.lower() for t in word_tokenize(v[arm].text)[0])
                if not (o_tokens - x_tokens) and sum(x_tokens.values()) > sum(o_tokens.values()):
                    out.append((v["O"], v[arm]))
    return out


# ------------------------------------------------------------------- C1: fine-tuned multiple-choice reference
def finetune_mc_run(model_name: str, train_rows: Sequence[NegRow], dev_rows: Sequence[NegRow], test_rows: Sequence[NegRow],
                    seed: int, out_dir: str | Path, lr: float = 2e-5, epochs: int = 5, batch_size: int = 16, max_length: int = 128,
                    warmup_frac: float = 0.06, device: Optional[str] = None, revision: Optional[str] = None,
                    state_dir: str | Path = "/content/fullft_states", verbose: bool = True,
                    extra_tests: Optional[Dict[str, Sequence[NegRow]]] = None) -> dict:
    """Cross-encoder multiple choice: each option is scored with "<context> <question>" [SEP] "<option>". float32 weights,
    NaN guard, selection on dev accuracy, test scored once; weights hashed on local disk and removed."""
    from transformers import AutoModelForMultipleChoice, AutoTokenizer, get_linear_schedule_with_warmup

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    try:
        model = AutoModelForMultipleChoice.from_pretrained(model_name, revision=revision, dtype=torch.float32)
    except TypeError:
        model = AutoModelForMultipleChoice.from_pretrained(model_name, revision=revision, torch_dtype=torch.float32)
    model = model.float().to(device)

    def encode(rows):
        first = [r.text for r in rows for _ in range(2)]
        second = [o for r in rows for o in r.options]
        enc = tok(first, second, truncation="longest_first", max_length=max_length, padding=True, return_tensors="pt")
        return {k: v.view(len(rows), 2, -1).to(device) for k, v in enc.items()}

    @torch.no_grad()
    def predict(rows):
        model.eval()
        out = []
        for i in range(0, len(rows), batch_size * 2):
            chunk = rows[i:i + batch_size * 2]
            logits = model(**encode(chunk)).logits.float()
            out.append(F.softmax(logits, dim=-1).cpu())
        return torch.cat(out) if out else torch.zeros(0, 2)

    def records(rows, probs):
        return [{"sample_id": r.sample_id, "passage_id": r.passage_id, "question_id": r.question_id, "edit_id": r.edit_id,
                 "label": r.label, "pred": LABELS2[int(p.argmax())], "probs": [round(float(x), 6) for x in p], "cue": r.cue,
                 "unseen_cue": not r.seen_family} for r, p in zip(rows, probs)]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(train_rows) + batch_size - 1) // batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(warmup_frac * steps), steps)
    rng = random.Random(seed)
    y_all = torch.tensor([LABELS2.index(r.label) for r in train_rows])
    best = {"dev_acc": -1.0, "epoch": None, "state": None}
    history, t0 = [], time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_rows))); rng.shuffle(order)
        tot = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            logits = model(**encode([train_rows[j] for j in idx])).logits.float()
            loss = F.cross_entropy(logits, y_all[idx].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, batch {i}")
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(idx)
        dp = predict(list(dev_rows))
        dev_acc = 100.0 * sum(LABELS2[int(p.argmax())] == r.label for r, p in zip(dev_rows, dp)) / len(dev_rows)
        history.append({"epoch": epoch, "train_loss": tot / len(order), "dev_acc": dev_acc, "elapsed_s": round(time.time() - t0, 1)})
        if verbose:
            print(f"[mc-ft {model_name.split('/')[-1]} seed {seed}] epoch {epoch} loss {tot/len(order):.4f} dev {dev_acc:.2f}", flush=True)
        if dev_acc > best["dev_acc"]:
            best = {"dev_acc": dev_acc, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dev_rec = records(dev_rows, predict(list(dev_rows)))
    test_rec = records(test_rows, predict(list(test_rows))) if test_rows else []
    summary = {"config": {"model_name": model_name, "revision": revision, "lr": lr, "epochs": epochs, "batch_size": batch_size,
                          "max_length": max_length, "seed": seed, "aggregator": model_name.split("/")[-1] + "-mc-ft"},
               "parameter_counts": {"total": sum(p.numel() for p in model.parameters())},
               "selected_epoch": best["epoch"], "epochs_run": len(history), "history": history,
               "dev": {"accuracy": best["dev_acc"]},
               "test": {"accuracy": (100.0 * sum(r["pred"] == r["label"] for r in test_rec) / len(test_rec)) if test_rec else None},
               "train_time_s": round(time.time() - t0, 1)}
    for name, recs in (("dev_predictions.jsonl", dev_rec), ("test_predictions.jsonl", test_rec)):
        if recs:
            with open(out_dir / name, "w", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name, rows in (extra_tests or {}).items():          # raw predictions only; no metric is computed (manifest v3.0, 6)
        recs = records(list(rows), predict(list(rows)))
        with open(out_dir / f"{name}_predictions.jsonl", "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    state_dir = Path(state_dir); state_dir.mkdir(parents=True, exist_ok=True)
    sd = state_dir / f"{model_name.split('/')[-1]}_mc_seed{seed}.pt"
    torch.save(best["state"], sd)
    from .provenance import sha256_file
    summary["model_state_sha256"] = sha256_file(sd); summary["model_state_bytes"] = sd.stat().st_size
    sd.unlink()
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary



# ------------------------------------------------------------------------ paired decision changes (flips)
def scenario_flips(records: Sequence[dict], rows_by_id: Dict[str, NegRow]) -> Dict[str, Dict[str, int]]:
    """Per scenario: 1 if the prediction for arm N (resp. S) differs from the prediction for arm O, else 0."""
    by = defaultdict(dict)
    for r in records:
        row = rows_by_id[r["sample_id"]]
        by[row.passage_id][row.arm] = r["pred"]
    out = {}
    for sid, v in by.items():
        if "O" in v:
            out[sid] = {a: int(v[a] != v["O"]) for a in ("N", "S", "P") if a in v}
    return out


def cue_lemma(cue: str) -> str:
    """Coarse cue identity for the same-cue analysis: 'not' for not / n't (not cannot), else the first word after a
    leading auxiliary (is, are, does, has, ...)."""
    c = (cue or "").lower().strip()
    if ("n't" in c or re.search(r"\bnot\b", c)) and "cannot" not in c:
        return "not"
    c = re.sub(r"^(is|are|was|were|does|do|did|has|have|had|can|could|will|would)\s+", "", c)
    return c.split()[0] if c else ""


def flip_rate(records: Sequence[dict], rows_by_id: Dict[str, NegRow], arm: str,
              scenario_ids: Optional[Iterable[str]] = None) -> Tuple[Optional[float], int]:
    keep = set(scenario_ids) if scenario_ids is not None else None
    f = [v[arm] for sid, v in scenario_flips(records, rows_by_id).items() if arm in v and (keep is None or sid in keep)]
    return ((100.0 * sum(f) / len(f)) if f else None), len(f)
