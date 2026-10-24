"""Exact Proposition-1 test in float64 on CPU (no TF32, no autocast).

Proposition 1: with context-independent token representations and an aggregator that is monotone under
token addition (max, log-sum-exp), adding context tokens can never lower an option score. Equivalently,
deleting tokens can never raise one. The v1.1 run evaluated this on GPU in float32 with TF32 matmuls and a
tolerance of 1e-5; the max aggregator showed increases of at most 6.8e-5, which is the size of TF32 rounding
when the context length changes. This module repeats the test in float64 so that rounding is ~1e-15.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .model import CACSAModule


def load_module(state_path: str, dim: int, aggregator: str, dtype: torch.dtype = torch.float64) -> CACSAModule:
    state = torch.load(state_path, map_location="cpu")
    m = CACSAModule(dim, aggregator)
    m.load_state_dict(state)
    return m.to(dtype=dtype).eval()


@torch.no_grad()
def option_scores(model: CACSAModule, ctx_states: torch.Tensor, cand_states: Sequence[torch.Tensor],
                  polarity: Optional[torch.Tensor] = None, dtype: torch.dtype = torch.float64) -> List[float]:
    """Scores of all candidates for one context, computed in `dtype` on CPU without padding of the context."""
    ctx = ctx_states.to(dtype=dtype).unsqueeze(0)                        # (1,n,d)
    n = ctx.shape[1]
    ctx_mask = torch.ones(1, n, dtype=torch.bool)
    m = max(c.shape[0] for c in cand_states)
    d = ctx.shape[2]
    cands = torch.zeros(1, len(cand_states), m, d, dtype=dtype)
    cmask = torch.zeros(1, len(cand_states), m, dtype=torch.bool)
    for k, c in enumerate(cand_states):
        cands[0, k, : c.shape[0]] = c.to(dtype=dtype)
        cmask[0, k, : c.shape[0]] = True
    pol = (polarity.to(dtype=dtype).unsqueeze(0) if polarity is not None else torch.zeros(1, n, dtype=dtype))
    return [float(x) for x in model(ctx, ctx_mask, cands, cmask, polarity=pol)[0]]


def violations(pairs: Iterable[Tuple[Sequence[float], Sequence[float]]], tol: float) -> Dict[str, float]:
    """pairs = (scores_with_more_tokens, scores_with_fewer_tokens); a violation is fewer > more + tol."""
    n = v = 0
    worst = float("-inf")
    for more, fewer in pairs:
        for a, b in zip(more, fewer):
            n += 1
            diff = float(b) - float(a)
            worst = max(worst, diff)
            if diff > tol:
                v += 1
    return {"n_scores": n, "violations": v, "violation_rate": (v / n if n else None),
            "max_increase_after_deletion": (worst if n else None), "tolerance": tol}
