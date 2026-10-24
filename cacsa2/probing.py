"""Probing analysis of manifest v3.0, section 7.5 (descriptive).

Question: do the frozen token states encode on which clause a negation edit sits? Data: the N and S arms of
NegPairs-Perm. Words: every word of the two clauses except the cue and, for substitution patterns, the unedited W;
label 1 if the word lies in the clause that carries the edit. Features: token states averaged over the subword
tokens of the word. Probe: L2-regularized logistic regression, C from {0.01, 0.1, 1, 10} by inner 5-fold grouped CV;
(i) outer 5-fold grouped CV on the seen families, (ii) transfer seen -> unseen. Neutral twins (syntactic family,
"also" inserted at the same site) give the selectivity control. The A5 gate AUC uses the same words and labels.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .negpairs import NegRow

_WORDS = re.compile(r"[A-Za-z0-9']+")
C_GRID = (0.01, 0.1, 1.0, 10.0)


def probe_words(row: NegRow, twin_cue_span: Optional[Tuple[int, int]] = None) -> List[Tuple[int, int, int]]:
    """(start, end, label) of the words of the two clauses, excluding the cue and the unedited W; label 1 for the
    clause that carries the edit (decisive clause in N, other clause in S)."""
    sp = row.perm["spans"]
    cue = tuple(sp["cue"]) if sp.get("cue") else twin_cue_span
    skip = [cue] + ([tuple(sp["unedited_W"])] if sp.get("unedited_W") else [])
    edited = sp["clause_decisive"] if row.arm == "N" else sp["clause_other"]
    other = sp["clause_other"] if row.arm == "N" else sp["clause_decisive"]
    out = []
    for m in _WORDS.finditer(row.context):
        s, e = m.start(), m.end()
        if any(x and x[0] <= s < x[1] for x in skip):
            continue
        if edited[0] <= s < edited[1]:
            out.append((s, e, 1))
        elif other[0] <= s < other[1]:
            out.append((s, e, 0))
    return out


def word_features(states: torch.Tensor, offsets: Sequence[Tuple[int, int]], words: Sequence[Tuple[int, int, int]],
                  values: Optional[torch.Tensor] = None) -> List[Tuple[torch.Tensor, int]]:
    """Mean of the encoder token states (or of per-token `values`) over the tokens overlapping each word."""
    out = []
    src = states if values is None else values
    for s, e, lab in words:
        idx = [k for k, (a, b) in enumerate(offsets) if b > a and a < e and s < b]
        if idx:
            out.append((src[idx].float().mean(dim=0), lab))
    return out


def neutral_twin(row: NegRow, word: str = "also") -> Tuple[str, Tuple[int, int]]:
    """Syntactic Perm arm with `word` in place of the inserted "not": the context and the span of the inserted word."""
    a, b = row.perm["spans"]["cue"]
    assert row.context[a:b] == "not", row.context
    ctx = row.context[:a] + word + row.context[b:]
    return ctx, (a, a + len(word))


def _grouped_folds(groups: Sequence[str], n_folds: int, seed: int) -> List[List[int]]:
    import random
    uniq = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    fold_of = {g: k % n_folds for k, g in enumerate(uniq)}
    folds = [[] for _ in range(n_folds)]
    for i, g in enumerate(groups):
        folds[fold_of[g]].append(i)
    return folds


def _fit_predict(Xtr, ytr, Xte, C):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
    clf.fit(Xtr, ytr)
    return clf.predict(Xte)


def _select_C(X, y, groups, seed):
    import numpy as np
    best, best_acc = None, -1.0
    folds = _grouped_folds(groups, 5, seed + 1)
    for C in C_GRID:
        accs = []
        for k in range(5):
            te = folds[k]
            tr = [i for j in range(5) if j != k for i in folds[j]]
            if not te or not tr:
                continue
            pred = _fit_predict(X[tr], y[tr], X[te], C)
            accs.append(float((pred == y[te]).mean()))
        acc = sum(accs) / len(accs) if accs else -1.0
        if acc > best_acc:
            best, best_acc = C, acc
    return best, best_acc


def run_probe(features: Sequence[torch.Tensor], labels: Sequence[int], groups: Sequence[str], seen: Sequence[bool],
              seed: int = 20261031) -> dict:
    """(i) outer 5-fold grouped CV on the seen-family words, (ii) transfer: fit on all seen-family words, test on the
    unseen-family words. Chance is 50% by construction (every word occurs once with each label)."""
    import numpy as np
    X = torch.stack(list(features)).numpy()
    y = np.asarray(labels)
    g = np.asarray(groups)
    seen = np.asarray(seen, dtype=bool)
    Xs, ys, gs = X[seen], y[seen], g[seen]
    folds = _grouped_folds(list(gs), 5, seed)
    cv = []
    for k in range(5):
        te = folds[k]
        tr = [i for j in range(5) if j != k for i in folds[j]]
        if not te or not tr:
            continue
        C, inner = _select_C(Xs[tr], ys[tr], list(gs[tr]), seed + k)
        pred = _fit_predict(Xs[tr], ys[tr], Xs[te], C)
        cv.append({"fold": k, "C": C, "inner_acc": inner, "acc": float(100.0 * (pred == ys[te]).mean()), "n_test": len(te)})
    out = {"n_words_seen": int(seen.sum()), "n_words_unseen": int((~seen).sum()), "cv_folds": cv}
    accs = [f["acc"] for f in cv]
    out["cv_acc_mean"] = (sum(accs) / len(accs)) if accs else None
    out["cv_acc_sd"] = (float(np.std(accs, ddof=1)) if len(accs) > 1 else None)
    if (~seen).sum() > 0 and seen.sum() > 0:
        C, inner = _select_C(Xs, ys, list(gs), seed + 11)
        pred = _fit_predict(Xs, ys, X[~seen], C)
        out["transfer"] = {"C": C, "inner_acc": inner, "acc": float(100.0 * (pred == y[~seen]).mean()), "n_test": int((~seen).sum())}
    return out


def gate_auc(values: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    from sklearn.metrics import roc_auc_score
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(list(labels), list(values)))
