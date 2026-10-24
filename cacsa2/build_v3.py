"""Build loop of the v3.0 sets (manifest v3.0, sections 4.2 to 4.7).

Metered API calls with a cap that the notebook sets per phase; candidates processed in schedule order in batches
(API calls of a batch run concurrently); Qwen verification of the Haiku survivors of each batch; acceptance in schedule
order (near-duplicate check against everything accepted so far, cap per stratum), so the result does not depend on the
order in which calls return. Every record is written to the state before the next step, so a restarted notebook
continues without paying twice. Opus audit of a seeded sample; summaries for the build report.
"""
from __future__ import annotations

import math
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .negpairs_build import VERIFY_PROMPT_V2, extract_json

PRICES = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0), "claude-haiku-4-5-20251001": (1.0, 5.0)}
SUPPORTS_EFFORT = {"claude-opus-5", "claude-sonnet-5"}
PENDING = ("haiku_passed", "qwen_passed")


class BudgetExceeded(Exception):
    pass


class CreditExhausted(Exception):
    pass


class Meter:
    """Thread-safe metered cost (USD per million tokens from PRICES). `cap` is checked before every call."""

    def __init__(self, d: Optional[dict] = None):
        self.lock = threading.Lock()
        self.usd, self.calls, self.tin, self.tout = 0.0, 0, 0, 0
        self.by_model: Dict[str, List[float]] = {}
        self.cap = math.inf
        if d:
            self.usd, self.calls = float(d.get("usd", 0.0)), int(d.get("calls", 0))
            self.tin, self.tout = int(d.get("in", 0)), int(d.get("out", 0))
            self.by_model = {k: list(v) for k, v in d.get("by_model", {}).items()}

    def check(self) -> None:
        with self.lock:
            if self.usd >= self.cap:
                raise BudgetExceeded()

    def add(self, model: str, tin: int, tout: int) -> None:
        pin, pout = PRICES.get(model, (5.0, 25.0))
        usd = (tin * pin + tout * pout) / 1e6
        with self.lock:
            self.usd += usd
            self.calls += 1
            self.tin += tin
            self.tout += tout
            m = self.by_model.setdefault(model, [0.0, 0, 0, 0])
            m[0] += usd
            m[1] += 1
            m[2] += tin
            m[3] += tout

    def to_dict(self) -> dict:
        with self.lock:
            return {"usd": self.usd, "calls": self.calls, "in": self.tin, "out": self.tout,
                    "by_model": {k: list(v) for k, v in self.by_model.items()}}


def make_call(client, meter: Meter, effort: Optional[str] = None, temperature: Optional[float] = None):
    """call(model, prompt, max_tokens) -> text; raises BudgetExceeded, CreditExhausted, or ValueError for a response
    without text (as in the v2.0 build)."""
    def _raw(model, prompt, max_tokens):
        kw = dict(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
        if model in SUPPORTS_EFFORT and effort:
            kw["output_config"] = {"effort": effort}
        if temperature is not None:
            kw["temperature"] = temperature
        for _ in range(3):
            try:
                return client.messages.create(**kw)
            except Exception as e:                       # noqa: BLE001  (API errors are classified by message)
                msg = str(e).lower()
                if "credit balance" in msg or "billing" in msg:
                    raise CreditExhausted(str(e)[:200])
                if ("output_config" in msg or "effort" in msg) and "output_config" in kw:
                    kw.pop("output_config")
                    continue
                if "temperature" in msg and "temperature" in kw:
                    kw.pop("temperature")
                    continue
                raise
        return client.messages.create(**kw)

    def call(model, prompt, max_tokens):
        meter.check()
        r = _raw(model, prompt, max_tokens)
        meter.add(model, r.usage.input_tokens, r.usage.output_tokens)
        text = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
        if not text.strip():
            raise ValueError(f"no text block in response (stop_reason={getattr(r, 'stop_reason', '?')})")
        return text
    return call


def run_phase(cands: Sequence, done: Dict[str, dict], *, stratum_c: Callable, stratum_r: Callable,
              caps: Dict[str, int], api_task: Callable, settle_qwen: Callable, final_key: Callable, dup_index,
              save: Callable[[], None], log: Callable[[str], None], label: str, meter: Optional[Meter] = None,
              batch_size: int = 24, max_workers: int = 8, per_stratum_factor: int = 4,
              safety_after: int = 72) -> str:
    """Process `cands` (schedule order; attributes uid, index) until every stratum reaches its cap, the schedule is
    exhausted, the budget cap is reached, the credit is exhausted or API errors repeat. Returns the reason.
    Operational safeguard (protects the budget, never changes which candidate is accepted): if a phase that has
    accepted nothing yet has processed `safety_after` candidates without any acceptance, it stops for inspection."""
    index_of = lambda r: r["meta"]["index"]
    count = Counter(stratum_r(r) for r in done.values() if r.get("stage") == "accepted")
    t0 = time.time()

    def settle(recs: List[dict]) -> None:
        recs = sorted(recs, key=index_of)
        pend = [r for r in recs if r.get("stage") == "haiku_passed"]
        if pend:
            settle_qwen(pend)
            save()
        for r in recs:
            if r.get("stage") != "qwen_passed":
                continue
            s = stratum_r(r)
            if count[s] >= caps[s]:
                r["stage"], r["reason"] = "cap", "stratum full at acceptance"
                continue
            dom, ctx, opts = final_key(r)
            if dup_index.is_duplicate(dom, ctx, opts):
                r["stage"], r["reason"] = "duplicate", "near-duplicate (at acceptance)"
                continue
            dup_index.add(dom, ctx, opts)
            count[s] += 1
            r["stage"] = "accepted"
        save()

    settle([r for r in done.values() if r.get("stage") in PENDING])          # resume after an interruption
    pending = [c for c in cands if c.uid not in done]
    processed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while True:
            if all(count[s] >= n for s, n in caps.items()):
                return "caps reached"
            batch, per, rest = [], Counter(), []
            for c in pending:
                s = stratum_c(c)
                if count[s] >= caps[s]:
                    continue
                need = caps[s] - count[s]
                if len(batch) < batch_size and per[s] < max(1, per_stratum_factor * need):
                    batch.append(c)
                    per[s] += 1
                else:
                    rest.append(c)
            pending = rest
            if not batch:
                return "schedule exhausted"
            results, halted, errors = {}, None, []
            for c, f in [(c, pool.submit(api_task, c)) for c in batch]:
                try:
                    results[c.uid] = f.result()
                except BudgetExceeded:
                    halted = halted or "budget"
                except CreditExhausted as e:
                    halted = f"credit exhausted: {e}"
                except Exception as e:                     # noqa: BLE001  (not recorded; retried on the next run)
                    errors.append(f"{c.uid}: {type(e).__name__}: {str(e)[:120]}")
            for c in batch:
                if c.uid in results:
                    done[c.uid] = results[c.uid]
            save()
            settle([results[c.uid] for c in batch if c.uid in results])
            processed += len(results)
            stages = Counter(done[c.uid]["stage"] for c in batch if c.uid in done)
            top = Counter(done[c.uid].get("reason") for c in batch if c.uid in done
                          and done[c.uid]["stage"] != "accepted").most_common(1)
            cost = f" | metered ${meter.usd:.2f}" if meter is not None else ""
            log(f"[{label}] accepted {sum(count.values())}/{sum(caps.values())} {dict(sorted(count.items()))} | batch "
                f"{dict(stages)} | processed {processed}{cost} | {(time.time() - t0) / 60:.1f} min"
                + (f" | top reason: {top[0][0]}" if top and top[0][0] else "")
                + (f" | errors {len(errors)}: {errors[0]}" if errors else ""))
            if halted:
                return halted
            if len(errors) >= max(3, len(batch) // 2):
                return f"repeated API errors: {errors[0]}"
            if safety_after and processed >= safety_after and sum(count.values()) == 0:
                return f"safety stop: no candidate accepted among the first {processed} (see the discard reasons)"


def verify_all_arms(gv: dict, call: Callable[[str, str, int], str], model: str, max_tokens: int) -> Tuple[bool, dict]:
    """Audit: the v2 verifier prompt on all four arms (no early stop)."""
    y = gv["answer_O"]
    flip = "B" if y == "A" else "A"
    expect = {"O": y, "P": y, "N": flip, "S": y}
    verdicts, agrees = {}, True
    for arm, ans in expect.items():
        v = extract_json(call(model, VERIFY_PROMPT_V2.format(context=gv["context_" + arm], question=gv["question"],
                                                             option_a=gv["options"][0], option_b=gv["options"][1]),
                              max_tokens)) or {}
        verdicts[arm] = v
        if v.get("answer") != ans or v.get("natural") is False:
            agrees = False
    return agrees, verdicts


def audit_sample(ids: Sequence[str], seed: int, fraction: float) -> List[str]:
    ids = sorted(ids)
    if not ids:
        return []
    k = max(1, int(round(fraction * len(ids))))
    return sorted(random.Random(seed + 7).sample(ids, k))


def run_audit(accepted: List[dict], gv_of: Callable[[dict], dict], call, model: str, sample: List[str], max_tokens: int,
              store: Dict[str, dict], save: Callable[[], None], log: Callable[[str], None]) -> dict:
    by = {r["meta"]["uid"]: r for r in accepted}
    stopped = None
    for uid in sample:
        if uid in store:
            continue
        try:
            agrees, verdicts = verify_all_arms(gv_of(by[uid]), call, model, max_tokens)
        except BudgetExceeded:
            stopped = "budget"
            break
        except CreditExhausted as e:
            stopped = f"credit exhausted: {e}"
            break
        except Exception as e:                             # noqa: BLE001
            log(f"audit call failed for {uid}: {type(e).__name__} {str(e)[:100]}")
            continue
        store[uid] = {"agrees": agrees, "verdicts": verdicts}
        save()
    cur = {u: store[u] for u in sample if u in store}
    return {"sample": len(sample), "checked": len(cur), "agree_all_arms": sum(v["agrees"] for v in cur.values()),
            "stopped": stopped}


def summarize(done: Dict[str, dict], stratum_r: Callable, caps: Dict[str, int]) -> dict:
    stages = Counter(r.get("stage") for r in done.values())
    by_stratum = defaultdict(Counter)
    for r in done.values():
        by_stratum[stratum_r(r)][r.get("stage")] += 1
    reasons = Counter(r.get("reason") for r in done.values() if r.get("stage") != "accepted")
    reached_qwen = [r for r in done.values() if r.get("qwen")]
    qwen_pass = sum(1 for r in reached_qwen if all(r["qwen"][a]["verdict"] == r["qwen"][a]["expected"] for a in "OPNS"))
    return {"candidates": len(done), "stages": dict(stages),
            "accepted_by_stratum": {s: by_stratum[s]["accepted"] for s in sorted(caps)},
            "caps": dict(caps), "stages_by_stratum": {s: dict(c) for s, c in sorted(by_stratum.items())},
            "qwen": {"checked": len(reached_qwen), "passed": qwen_pass},
            "discard_reasons": dict(reasons.most_common(40))}
