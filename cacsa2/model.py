"""The shared pair scorer (CACSA v2 core) and the five aggregation rules.

score(o) = mean_j A_i(c_ij),   c_ij = e_ij + lambda * tanh(aff_ij)

  e_ij  = F(h_i)^T F(u_j) / sqrt(d')              (cross-sentence attention, Parikh et al. 2016 style)
  aff_ij = MLP([h_i ; u_j ; h_i - u_j ; h_i * u_j]) (affinity network)

The first affinity layer is implemented as the algebraically identical sum of four linear maps so
that only the Hadamard term needs an (n x m x d) tensor. Parameter count is identical to a single
Linear(4d, hidden).

Aggregators over context tokens i (the experimental factor):
  A1 MAX          max_i c_ij                              monotone under token addition
  A2 MEAN         mean_i c_ij                             scope-blind global suppression possible
  A3 LSE          tau * log sum_i exp(c_ij / tau)         monotone under token addition
  A4 SIGNED-MAX   max_{i not in N} c_ij - beta * max_{i in N} c_ij, N from the scope resolver (hard mask)
  A5 SIGNED-MAX-G as A4 with soft membership p_i = sigmoid(w . h_i + b) learned from frozen states
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

AGGREGATORS = ("max", "mean", "lse", "signed_max", "signed_max_gate")
NEG_BIG = 1e4


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class PairScorer(nn.Module):
    def __init__(self, dim: int, proj: int = 128, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = proj
        self.F = nn.Sequential(nn.Linear(dim, proj), nn.ReLU(), nn.Linear(proj, proj))
        self.Wa = nn.Linear(dim, hidden, bias=True)    # acts on h_i (and carries the bias)
        self.Wb = nn.Linear(dim, hidden, bias=False)   # acts on u_j
        self.Wd = nn.Linear(dim, hidden, bias=False)   # acts on (h_i - u_j)
        self.Wh = nn.Linear(dim, hidden, bias=False)   # acts on (h_i * u_j)
        self.rest = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden // 4), nn.ReLU(),
                                  nn.Linear(hidden // 4, 1))
        self.theta = nn.Parameter(torch.zeros(()))     # lambda = sigmoid(theta), init 0.5
        self.use_checkpoint = True                     # recompute the (B,n,m,h) affinity block in backward

    @property
    def lam(self) -> torch.Tensor:
        return torch.sigmoid(self.theta)

    def _affinity(self, ctx: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        ha = self.Wa(ctx) + self.Wd(ctx)                     # (B,n,h)
        hb = self.Wb(cand) - self.Wd(cand)                   # (B,m,h)
        hh = self.Wh(ctx.unsqueeze(2) * cand.unsqueeze(1))   # (B,n,m,h)
        return self.rest(ha.unsqueeze(2) + hb.unsqueeze(1) + hh).squeeze(-1)

    def forward(self, ctx: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """ctx (B,n,d), cand (B,m,d) -> c (B,n,m)."""
        fa, fb = self.F(ctx), self.F(cand)
        e = torch.matmul(fa, fb.transpose(1, 2)) / math.sqrt(self.proj)
        if self.use_checkpoint and torch.is_grad_enabled() and self.training:
            aff = checkpoint(self._affinity, ctx, cand, use_reentrant=False)
        else:
            aff = self._affinity(ctx, cand)
        return e + self.lam * torch.tanh(aff)


def masked_max(c: torch.Tensor, mask: torch.Tensor, empty_value: float = 0.0) -> torch.Tensor:
    """max over dim 1 of c (B,n,m) restricted to mask (B,n); rows with an empty mask get `empty_value`."""
    filled = c.masked_fill(~mask.unsqueeze(2), -NEG_BIG)
    out = filled.max(dim=1).values
    empty = ~mask.any(dim=1)
    if empty.any():
        out = torch.where(empty.unsqueeze(1), torch.full_like(out, empty_value), out)
    return out


def masked_mean(c: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.unsqueeze(2).to(c.dtype)
    return (c * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


class Aggregator(nn.Module):
    def __init__(self, kind: str, dim: int):
        super().__init__()
        if kind not in AGGREGATORS:
            raise ValueError(kind)
        self.kind = kind
        if kind == "lse":
            self.t = nn.Parameter(torch.tensor(_inv_softplus(1.0)))         # tau = softplus(t), init 1
        if kind in ("signed_max", "signed_max_gate"):
            self.b = nn.Parameter(torch.tensor(_inv_softplus(1.0)))         # beta = softplus(b), init 1
        if kind == "signed_max_gate":
            self.gate = nn.Linear(dim, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -2.0)                          # p ~ 0.12 at start

    @property
    def tau(self) -> Optional[torch.Tensor]:
        return F.softplus(self.t) + 1e-3 if self.kind == "lse" else None

    @property
    def beta(self) -> Optional[torch.Tensor]:
        return F.softplus(self.b) if self.kind in ("signed_max", "signed_max_gate") else None

    def forward(self, c: torch.Tensor, ctx_mask: torch.Tensor, ctx_states: Optional[torch.Tensor] = None,
                polarity: Optional[torch.Tensor] = None) -> torch.Tensor:
        """c (B,n,m), ctx_mask (B,n) bool, polarity (B,n) in {0,1} for signed_max -> (B,m)."""
        if self.kind == "max":
            return masked_max(c, ctx_mask)
        if self.kind == "mean":
            return masked_mean(c, ctx_mask)
        if self.kind == "lse":
            tau = self.tau
            filled = (c / tau).masked_fill(~ctx_mask.unsqueeze(2), -NEG_BIG)
            return tau * torch.logsumexp(filled, dim=1)
        if self.kind == "signed_max":
            if polarity is None:
                raise ValueError("signed_max requires a polarity mask")
            neg_mask = ctx_mask & (polarity > 0.5)
            pos_mask = ctx_mask & ~(polarity > 0.5)
            pos = masked_max(c, pos_mask, empty_value=0.0)
            neg = masked_max(c, neg_mask, empty_value=0.0)
            return pos - self.beta * neg
        if self.kind == "signed_max_gate":
            if ctx_states is None:
                raise ValueError("signed_max_gate requires context states")
            p = torch.sigmoid(self.gate(ctx_states)).squeeze(-1)                 # (B,n)
            p = p * ctx_mask.to(p.dtype)
            pos = masked_max(c * (1.0 - p).unsqueeze(2), ctx_mask, empty_value=0.0)
            neg = masked_max(c * p.unsqueeze(2), ctx_mask, empty_value=0.0)
            return pos - self.beta * neg
        raise RuntimeError(self.kind)

    def gate_values(self, ctx_states: torch.Tensor) -> Optional[torch.Tensor]:
        if self.kind != "signed_max_gate":
            return None
        return torch.sigmoid(self.gate(ctx_states)).squeeze(-1)


class CACSAModule(nn.Module):
    """Scores K candidates for one context: scores (B,K)."""

    def __init__(self, dim: int, aggregator: str, proj: int = 128, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.scorer = PairScorer(dim, proj=proj, hidden=hidden, dropout=dropout)
        self.agg = Aggregator(aggregator, dim)
        self.aggregator = aggregator

    def pair_scores(self, ctx: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        return self.scorer(ctx, cand)

    def option_score(self, ctx: torch.Tensor, ctx_mask: torch.Tensor, cand: torch.Tensor, cand_mask: torch.Tensor,
                     polarity: Optional[torch.Tensor] = None) -> torch.Tensor:
        c = self.scorer(ctx, cand)                                            # (B,n,m)
        per_token = self.agg(c, ctx_mask, ctx_states=ctx, polarity=polarity)  # (B,m)
        w = cand_mask.to(per_token.dtype)
        return (per_token * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)       # (B,)

    def forward(self, ctx: torch.Tensor, ctx_mask: torch.Tensor, cands: torch.Tensor, cand_mask: torch.Tensor,
                polarity: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ctx (B,n,d), ctx_mask (B,n), cands (B,K,m,d), cand_mask (B,K,m), polarity (B,n) -> scores (B,K)."""
        K = cands.shape[1]
        scores = [self.option_score(ctx, ctx_mask, cands[:, k], cand_mask[:, k], polarity) for k in range(K)]
        return torch.stack(scores, dim=1)

    def parameter_counts(self) -> Dict[str, int]:
        core = sum(p.numel() for p in self.scorer.parameters())
        agg = sum(p.numel() for p in self.agg.parameters())
        return {"scorer": core, "aggregator": agg, "total": core + agg}


# ------------------------------------------------------------------------------ MaxSim baselines (manifest v3.0, 5.4)
class MaxSimModule(nn.Module):
    """M1, learned MaxSim: c_ij = (W h_i)^T (W u_j) / sqrt(proj) with one linear map W (d -> proj, no bias); maximum
    over context tokens, mean over option tokens. The minimal late-interaction head on a frozen encoder, without the
    affinity network. Same forward signature as CACSAModule (the polarity mask is ignored)."""
    aggregator = "m1"

    def __init__(self, dim: int, proj: int = 128):
        super().__init__()
        self.proj = proj
        self.W = nn.Linear(dim, proj, bias=False)

    def pair_scores(self, ctx: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.W(ctx), self.W(cand).transpose(1, 2)) / math.sqrt(self.proj)

    def option_score(self, ctx, ctx_mask, cand, cand_mask, polarity=None) -> torch.Tensor:
        per_token = masked_max(self.pair_scores(ctx, cand), ctx_mask)          # (B,m)
        w = cand_mask.to(per_token.dtype)
        return (per_token * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

    def forward(self, ctx, ctx_mask, cands, cand_mask, polarity=None) -> torch.Tensor:
        return torch.stack([self.option_score(ctx, ctx_mask, cands[:, k], cand_mask[:, k]) for k in range(cands.shape[1])], dim=1)

    def parameter_counts(self) -> Dict[str, int]:
        n = sum(p.numel() for p in self.parameters())
        return {"scorer": n, "aggregator": 0, "total": n}


class ZeroShotMaxSim(nn.Module):
    """M0, zero-shot MaxSim: c_ij = h_i^T u_j (cosine when the states are unit vectors); maximum over context tokens,
    mean over option tokens. No parameters; scored once, no seeds."""
    aggregator = "m0"

    def __init__(self, dim: int = 0):
        super().__init__()

    def option_score(self, ctx, ctx_mask, cand, cand_mask, polarity=None) -> torch.Tensor:
        per_token = masked_max(torch.matmul(ctx, cand.transpose(1, 2)), ctx_mask)
        w = cand_mask.to(per_token.dtype)
        return (per_token * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

    def forward(self, ctx, ctx_mask, cands, cand_mask, polarity=None) -> torch.Tensor:
        return torch.stack([self.option_score(ctx, ctx_mask, cands[:, k], cand_mask[:, k]) for k in range(cands.shape[1])], dim=1)

    def parameter_counts(self) -> Dict[str, int]:
        return {"scorer": 0, "aggregator": 0, "total": 0}


def m1_factory(dim: int, cfg) -> MaxSimModule:
    return MaxSimModule(dim, proj=getattr(cfg, "proj", 128))


def make_module(kind: str, dim: int, proj: int = 128, hidden: int = 256, dropout: float = 0.1) -> nn.Module:
    """Module by condition name: the CACSA aggregators, "m1", "m0", and the controls "head_only" / "option_only"."""
    if kind in AGGREGATORS:
        return CACSAModule(dim, kind, proj=proj, hidden=hidden, dropout=dropout)
    if kind == "m1":
        return MaxSimModule(dim, proj=proj)
    if kind == "m0":
        return ZeroShotMaxSim(dim)
    if kind in ("head_only", "option_only"):
        from .controls import HeadOnlyScorer, OptionOnlyScorer
        return HeadOnlyScorer(dim) if kind == "head_only" else OptionOnlyScorer(dim, hidden=hidden, dropout=dropout)
    raise ValueError(kind)
