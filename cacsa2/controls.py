"""Controls on frozen states (reference points, not parameter-matched alternatives).

C0 head-only: score_k = w . [ mean(ctx) ; mean(cand_k) ; mean(ctx) * mean(cand_k) ; |mean(ctx) - mean(cand_k)| ] + b.
   It answers whether the frozen encoder's pooled representation already exposes polarity to a linear reader.
C2 option-only: score_k = MLP(mean(cand_k)); the context is never read (see OptionOnlyScorer).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


def masked_mean_states(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.unsqueeze(-1).to(x.dtype)
    return (x * w).sum(dim=-2) / w.sum(dim=-2).clamp_min(1.0)


class HeadOnlyScorer(nn.Module):
    aggregator = "head_only"

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(4 * dim, 1)

    def forward(self, ctx: torch.Tensor, ctx_mask: torch.Tensor, cands: torch.Tensor, cand_mask: torch.Tensor,
                polarity: Optional[torch.Tensor] = None) -> torch.Tensor:
        p = masked_mean_states(ctx, ctx_mask)                       # (B,d)
        c = masked_mean_states(cands, cand_mask)                    # (B,K,d)
        pe = p.unsqueeze(1).expand_as(c)
        feat = torch.cat([pe, c, pe * c, (pe - c).abs()], dim=-1)
        return self.w(feat).squeeze(-1)                             # (B,K)

    def parameter_counts(self) -> Dict[str, int]:
        n = sum(p.numel() for p in self.parameters())
        return {"scorer": n, "aggregator": 0, "total": n}


def head_only_factory(dim: int, cfg) -> HeadOnlyScorer:
    return HeadOnlyScorer(dim)


class OptionOnlyScorer(nn.Module):
    """Control C2 (context-blind): score_k = MLP(mean(cand_k)); the context and the polarity mask are ignored.

    Because the two options of a scenario are identical in all four arms and the label flips only in N, this reader can
    exceed 50% only through properties of the option texts themselves. Its accuracy on the affirmative arms (O, P)
    measures how much of the task is solvable without reading the context; the learnability gate requires A1 to beat
    it by a fixed margin."""
    aggregator = "option_only"

    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, ctx: torch.Tensor, ctx_mask: torch.Tensor, cands: torch.Tensor, cand_mask: torch.Tensor,
                polarity: Optional[torch.Tensor] = None) -> torch.Tensor:
        c = masked_mean_states(cands, cand_mask)                    # (B,K,d)
        return self.mlp(c).squeeze(-1)                              # (B,K)

    def parameter_counts(self) -> Dict[str, int]:
        n = sum(p.numel() for p in self.parameters())
        return {"scorer": n, "aggregator": 0, "total": n}


def option_only_factory(dim: int, cfg) -> OptionOnlyScorer:
    return OptionOnlyScorer(dim, hidden=getattr(cfg, "hidden", 256), dropout=getattr(cfg, "dropout", 0.1))
