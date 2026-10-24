"""Uncertainty: paired seed t-intervals, passage-cluster bootstrap, exact McNemar test."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def sample_sd(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def paired_t_interval(diffs: Sequence[float], conf: float = 0.95) -> Dict[str, float]:
    """Mean of paired differences with a two-sided t interval (df = n-1)."""
    from scipy import stats

    n = len(diffs)
    m = mean(diffs)
    if n < 2:
        return {"n": n, "mean": m, "sd": float("nan"), "low": float("nan"), "high": float("nan"), "positive_count": sum(d > 0 for d in diffs)}
    sd = sample_sd(diffs)
    t = stats.t.ppf(0.5 + conf / 2, df=n - 1)
    h = t * sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "low": m - h, "high": m + h, "positive_count": sum(d > 0 for d in diffs)}


def cluster_bootstrap(items: Sequence, cluster_of: Callable[[object], Hashable], statistic: Callable[[Sequence], float],
                      n_boot: int = 10000, seed: int = 20261002, conf: float = 0.95) -> Dict[str, float]:
    """Percentile bootstrap resampling whole clusters (e.g., passages) with replacement.

    `items` are resampled by cluster; `statistic` maps a list of items to a float. All seeds/edits of a
    cluster travel together when the items carry them.
    """
    clusters: Dict[Hashable, List] = defaultdict(list)
    for it in items:
        clusters[cluster_of(it)].append(it)
    keys = list(clusters)
    rng = random.Random(seed)
    point = statistic(list(items))
    reps = []
    for _ in range(n_boot):
        sample: List = []
        for _ in range(len(keys)):
            sample.extend(clusters[keys[rng.randrange(len(keys))]])
        reps.append(statistic(sample))
    reps.sort()
    lo = reps[int((0.5 - conf / 2) * (n_boot - 1))]
    hi = reps[int((0.5 + conf / 2) * (n_boot - 1))]
    return {"point": point, "low": lo, "high": hi, "n_clusters": len(keys), "n_boot": n_boot, "seed": seed}


def mcnemar_exact(b: int, c: int) -> Dict[str, float]:
    """Exact two-sided McNemar test on discordant counts b (A right, B wrong) and c (A wrong, B right)."""
    from scipy import stats

    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0}
    p = float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))
    return {"b": b, "c": c, "p": p}


def discordant_counts(records_a: Sequence[dict], records_b: Sequence[dict]) -> Tuple[int, int]:
    """Paired by sample_id: b = A right & B wrong, c = A wrong & B right."""
    bmap = {r["sample_id"]: r for r in records_b}
    b = c = 0
    for ra in records_a:
        rb = bmap.get(ra["sample_id"])
        if rb is None:
            continue
        ok_a, ok_b = ra["pred"] == ra["label"], rb["pred"] == rb["label"]
        if ok_a and not ok_b:
            b += 1
        elif ok_b and not ok_a:
            c += 1
    return b, c
