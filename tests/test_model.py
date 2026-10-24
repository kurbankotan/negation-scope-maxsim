import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cacsa2.model import AGGREGATORS, CACSAModule, PairScorer, masked_max  # noqa: E402

torch.manual_seed(0)
D = 16


def _rand_module(agg, seed=0):
    torch.manual_seed(seed)
    m = CACSAModule(D, agg, proj=8, hidden=16, dropout=0.0)
    for p in m.parameters():  # make the affinity term matter
        with torch.no_grad():
            p.add_(torch.randn_like(p) * 0.3)
    return m.eval()


def _score(m, ctx, cand, pol=None):
    ctx_mask = torch.ones(1, ctx.shape[1], dtype=torch.bool)
    cand_mask = torch.ones(1, 1, cand.shape[1], dtype=torch.bool)
    return m(ctx, ctx_mask, cand.unsqueeze(1), cand_mask, polarity=pol)[0, 0].item()


def test_affinity_first_layer_matches_concatenated_linear():
    torch.manual_seed(1)
    ps = PairScorer(D, proj=8, hidden=16, dropout=0.0).eval()
    ref = nn.Linear(4 * D, 16)
    with torch.no_grad():
        ref.weight.copy_(torch.cat([ps.Wa.weight, ps.Wb.weight, ps.Wd.weight, ps.Wh.weight], dim=1))
        ref.bias.copy_(ps.Wa.bias)
    a, b = torch.randn(3, D), torch.randn(4, D)
    feat = torch.cat([a.unsqueeze(1).expand(3, 4, D), b.unsqueeze(0).expand(3, 4, D),
                      (a.unsqueeze(1) - b.unsqueeze(0)), (a.unsqueeze(1) * b.unsqueeze(0))], dim=-1)
    want = ref(feat)
    got = ps.Wa(a).unsqueeze(1) + ps.Wd(a).unsqueeze(1) + ps.Wb(b).unsqueeze(0) - ps.Wd(b).unsqueeze(0) \
        + ps.Wh(a.unsqueeze(1) * b.unsqueeze(0))
    assert torch.allclose(want, got, atol=1e-5)


def test_max_and_lse_are_monotone_under_token_addition():
    for agg in ("max", "lse"):
        m = _rand_module(agg)
        for trial in range(50):
            ctx = torch.randn(1, 6, D)
            extra = torch.randn(1, 1, D)
            cand = torch.randn(1, 4, D)
            s0 = _score(m, ctx, cand)
            s1 = _score(m, torch.cat([ctx, extra], dim=1), cand)
            assert s1 >= s0 - 1e-6, (agg, trial, s0, s1)


def test_mean_can_decrease_under_token_addition():
    m = _rand_module("mean")
    found = False
    for trial in range(200):
        ctx = torch.randn(1, 6, D)
        extra = torch.randn(1, 1, D)
        cand = torch.randn(1, 4, D)
        if _score(m, torch.cat([ctx, extra], dim=1), cand) < _score(m, ctx, cand) - 1e-6:
            found = True
            break
    assert found


def test_signed_max_lets_in_scope_token_lower_the_score():
    m = _rand_module("signed_max")
    found = False
    for trial in range(200):
        ctx = torch.randn(1, 6, D)
        extra = torch.randn(1, 1, D) * 3
        cand = torch.randn(1, 4, D)
        pol0 = torch.zeros(1, 6)
        pol1 = torch.cat([torch.zeros(1, 6), torch.ones(1, 1)], dim=1)
        s0 = _score(m, ctx, cand, pol0)
        s1 = _score(m, torch.cat([ctx, extra], dim=1), cand, pol1)
        if s1 < s0 - 1e-6:
            found = True
            break
    assert found


def test_signed_max_with_all_zero_polarity_equals_max():
    mm = _rand_module("max", seed=3)
    ms = _rand_module("signed_max", seed=3)
    ms.scorer.load_state_dict(mm.scorer.state_dict())
    ctx, cand = torch.randn(1, 7, D), torch.randn(1, 3, D)
    assert math.isclose(_score(mm, ctx, cand), _score(ms, ctx, cand, torch.zeros(1, 7)), rel_tol=1e-6, abs_tol=1e-6)


def test_padding_does_not_change_scores():
    for agg in AGGREGATORS:
        m = _rand_module(agg, seed=5)
        ctx, cand = torch.randn(1, 5, D), torch.randn(1, 3, D)
        pol = torch.tensor([[0., 1., 0., 0., 1.]])
        s_single = _score(m, ctx, cand, pol)
        # same example padded inside a batch with a longer example
        ctx2, cand2 = torch.randn(1, 9, D), torch.randn(1, 6, D)
        ctxb = torch.zeros(2, 9, D); ctxb[0, :5] = ctx[0]; ctxb[1] = ctx2[0]
        maskb = torch.zeros(2, 9, dtype=torch.bool); maskb[0, :5] = True; maskb[1] = True
        polb = torch.zeros(2, 9); polb[0, :5] = pol[0]
        candb = torch.zeros(2, 1, 6, D); candb[0, 0, :3] = cand[0]; candb[1, 0] = cand2[0]
        cmaskb = torch.zeros(2, 1, 6, dtype=torch.bool); cmaskb[0, 0, :3] = True; cmaskb[1, 0] = True
        s_batch = m(ctxb, maskb, candb, cmaskb, polarity=polb)[0, 0].item()
        assert math.isclose(s_single, s_batch, rel_tol=1e-5, abs_tol=1e-5), (agg, s_single, s_batch)


def test_masked_max_empty_rows_get_empty_value():
    c = torch.randn(2, 3, 2)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    out = masked_max(c, mask, empty_value=0.0)
    assert torch.allclose(out[1], torch.zeros(2))
    assert torch.allclose(out[0], c[0][[0, 2]].max(dim=0).values)


def test_parameter_counts_identical_core():
    counts = {agg: _rand_module(agg).parameter_counts() for agg in AGGREGATORS}
    cores = {v["scorer"] for v in counts.values()}
    assert len(cores) == 1
    assert counts["max"]["aggregator"] == 0 and counts["mean"]["aggregator"] == 0
    assert counts["lse"]["aggregator"] == 1 and counts["signed_max"]["aggregator"] == 1
    assert counts["signed_max_gate"]["aggregator"] == 1 + D + 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
