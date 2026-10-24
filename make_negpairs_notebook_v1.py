"""Generate the v1 NegPairs builder notebook (archived: it examined the first candidates of the schedule on 3 October 2026, see build_state_v1_backup.json; superseded by make_negpairs_notebook.py, construction v2)."""
import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# NegPairs builder: controlled negation minimal pairs in the CACSA format

**Runtime:** CPU is enough. Add `ANTHROPIC_API_KEY` in the Colab **Secrets** panel (key icon) and switch on *Notebook
access*. Put `cacsa2_project.zip` (0.2.0 or later) in `MyDrive/CACSA_v2/`. Then `Runtime > Run all`.

**What it builds.** Scenarios with a context, a question and two content-bearing options, each in four arms:
**O** affirmative (answer y), **P** paraphrase (answer y), **N** one negation cue on the decisive words (answer flips),
**S** one negation cue on an irrelevant detail (answer stays y). Negation families *syntactic* and *quantifier* are
"seen" (split 60/15/25 into train/dev/test); *affixal* and *lexical* are "unseen" (test only).

**How.** One generator call per scenario (`claude-sonnet-5`), free structural filters, a near-duplicate check, then
each arm is shown **alone** to a different model (`claude-haiku-4-5`) that answers A/B and judges naturalness and the
presence of negation; a scenario is kept only if all four arms come back as intended. A seeded 10% sample is
re-verified by `claude-opus-5` to estimate label noise. Everything is cached in Drive: re-running continues where it
stopped and never pays twice. A hard budget cap stops the run.

**Status of the data.** Model-generated and model-verified, **not human-verified**. The paper must describe it so.

| Step | Expected |
|---|---|
| setup + preflight (one inspected call) | 1–2 min |
| generation and verification | about 1–2 h; a line every few seconds with kept counts, cost and ETA |
| splits, Opus audit, report, zip | 5–10 min |
""")

code("""# ============================== CONFIGURATION ==============================
PROJECT_ROOT  = "/content/drive/MyDrive/CACSA_v2"
PACKAGE_ZIP   = f"{PROJECT_ROOT}/cacsa2_project.zip"
OUT_NAME      = "negpairs_v1"
TARGETS       = {"syntactic": 500, "quantifier": 500, "affixal": 250, "lexical": 250}   # kept scenarios per family
OVERSAMPLE    = 2.2          # candidates scheduled per kept target (yield is expected around 50-70%)
BUDGET_USD    = 35.00        # hard stop, metered from real token usage
GEN_MODEL     = "claude-sonnet-5"
VERIFY_MODEL  = "claude-haiku-4-5-20251001"
AUDIT_MODEL   = "claude-opus-5"
GEN_EFFORT    = "medium"
AUDIT_FRACTION = 0.10
MAX_WORKERS   = 4
SEED          = 20261003
# ===========================================================================
""")

code("""%pip install -q anthropic wordfreq
from google.colab import drive
drive.mount('/content/drive')
import os, sys, json, zipfile, time, random, threading, hashlib
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
ROOT = Path(PROJECT_ROOT)
import shutil, importlib, hashlib
_zip_sha = hashlib.sha256(open(PACKAGE_ZIP, 'rb').read()).hexdigest()
PKG_DIR = Path('/content') / f'cacsa2_pkg_{_zip_sha[:12]}'          # fresh folder per zip version (no stale files)
shutil.rmtree(PKG_DIR, ignore_errors=True)
with zipfile.ZipFile(PACKAGE_ZIP) as z: z.extractall(PKG_DIR)
for _m in [m for m in list(sys.modules) if m == 'cacsa2' or m.startswith('cacsa2.')]: del sys.modules[_m]
sys.path = [p for p in sys.path if 'cacsa2_unzipped' not in p and 'cacsa2_pkg_' not in p]
sys.path.insert(0, str(next(PKG_DIR.rglob('cacsa2/__init__.py')).parent.parent))
importlib.invalidate_caches()
import cacsa2
_v = tuple(int(x) for x in cacsa2.__version__.split('.'))
print('cacsa2', cacsa2.__version__, '| zip sha256', _zip_sha[:12], '|', cacsa2.__file__)
assert _v >= (0, 2, 0), f'cacsa2_project.zip on Drive is version {cacsa2.__version__}; upload the newest cacsa2_project.zip (>= 0.2.0)'
from cacsa2 import negpairs_build as nb, provenance
OUT = ROOT / 'data' / OUT_NAME; OUT.mkdir(parents=True, exist_ok=True)
STATE = OUT / 'build_state.json'                      # resume file: every candidate's outcome
print('cacsa2', cacsa2.__version__, '| output', OUT)

API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
if not API_KEY:
    try:
        from google.colab import userdata
        API_KEY = userdata.get('ANTHROPIC_API_KEY')
    except Exception:
        pass
assert API_KEY, 'No API key: Secrets panel -> add ANTHROPIC_API_KEY -> switch on Notebook access.'
""")

code("""# ---- API harness: metered calls, budget cap, adaptive-thinking-safe text extraction ----------------------
from anthropic import Anthropic
PRICES = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0), "claude-haiku-4-5-20251001": (1.0, 5.0)}
SUPPORTS_EFFORT = {"claude-opus-5", "claude-sonnet-5", "claude-fable-5"}
_client = Anthropic(api_key=API_KEY, max_retries=6, timeout=90.0)
_lock = threading.Lock()
METER = {'usd': 0.0, 'calls': 0, 'in': 0, 'out': 0}
if STATE.exists():
    METER.update(json.loads(STATE.read_text()).get('meter', {}))

class BudgetExceeded(Exception):
    pass

def _raw(model, prompt, max_tokens, effort):
    kw = dict(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
    if model in SUPPORTS_EFFORT and effort:
        kw["output_config"] = {"effort": effort}
    try:
        return _client.messages.create(**kw)
    except Exception as e:
        if "output_config" in str(e) or "effort" in str(e) or isinstance(e, TypeError):
            kw.pop("output_config", None)
            return _client.messages.create(**kw)
        raise

def make_call(effort):
    def call(model, prompt, max_tokens):
        with _lock:
            if METER['usd'] >= BUDGET_USD:
                raise BudgetExceeded()
        r = _raw(model, prompt, max_tokens, effort)
        pin, pout = PRICES.get(model, (5.0, 25.0))
        with _lock:
            METER['usd'] += (r.usage.input_tokens * pin + r.usage.output_tokens * pout) / 1e6
            METER['calls'] += 1; METER['in'] += r.usage.input_tokens; METER['out'] += r.usage.output_tokens
        text = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
        if not text.strip():
            raise ValueError("no text block in response")
        return text
    return call

call_gen = make_call(GEN_EFFORT)       # generator (effort applies to Claude 5 models)
call_low = make_call("low")            # verifier and audit

# preflight: one generator call and one verifier call, inspected before the batch
work = nb.schedule(TARGETS, OVERSAMPLE, SEED)
c0 = work[0]
fam = nb.FAMILIES[c0.family]
g0 = nb.extract_json(call_gen(GEN_MODEL, nb.GEN_PROMPT.format(domain=c0.domain, situation=c0.situation, banned=nb.BANNED_TOPICS,
                                                              family=c0.family, family_desc=fam['desc'], examples=fam['examples']), 1200))
print('PREFLIGHT generator output:'); print(json.dumps(g0, indent=1, ensure_ascii=False))
assert g0 is not None, 'generator output did not parse as JSON'
if g0.get('usable'):
    print('structural filter:', nb.structural_filter(g0, c0.family) or 'pass')
    v0 = nb.extract_json(call_low(VERIFY_MODEL, nb.VERIFY_PROMPT.format(context=g0['context_O'], question=g0['question'],
                                                                         option_a=g0['options'][0], option_b=g0['options'][1]), 200))
    print('verifier on arm O:', v0)
print(f"preflight cost so far ${METER['usd']:.4f}; {len(work)} candidates scheduled")
""")

code("""# ---- generation + blind verification (resume-safe, budget-capped) ---------------------------------------
state = json.loads(STATE.read_text()) if STATE.exists() else {'done': {}, 'meter': METER}
done = state['done']
dup_index = nb.NearDuplicateIndex(threshold=0.6)
kept_count = Counter()
for uid, res in done.items():
    if res.get('stage') == 'kept':
        m = res['meta']; dup_index.add(m['domain'], res['generated']['context_O'], res['generated']['options'])
        kept_count[m['family']] += 1
dup_lock = threading.Lock()
def is_dup(domain, ctx, opts):
    with dup_lock:
        return dup_index.is_duplicate(domain, ctx, opts)

def run_one(c):
    return nb.process(c, lambda m, p, t: (call_gen if m == GEN_MODEL else call_low)(m, p, t), GEN_MODEL, VERIFY_MODEL, is_dup)

queue = [c for c in work if c.uid not in done]
t0, halted = time.time(), None
kept_at_start = sum(kept_count.values())
print(f'resuming: {sum(kept_count.values())} kept, {len(done)} examined; queue {len(queue)}')
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    i = 0
    while i < len(queue) and not halted:
        if all(kept_count[f] >= n for f, n in TARGETS.items()):
            break
        chunk = []
        while i < len(queue) and len(chunk) < MAX_WORKERS * 3:
            c = queue[i]; i += 1
            if kept_count[c.family] < TARGETS[c.family]:
                chunk.append(c)
        futs = {pool.submit(run_one, c): c for c in chunk}
        for f, c in futs.items():
            try:
                res = f.result()
            except BudgetExceeded:
                halted = 'budget'; continue
            except Exception as e:
                res = {'stage': 'error', 'reason': f'{type(e).__name__}: {str(e)[:120]}'}
            if res.get('stage') == 'kept':
                with dup_lock:
                    if dup_index.is_duplicate(c.domain, res['generated']['context_O'], res['generated']['options']):
                        res = {'stage': 'duplicate', 'reason': 'near-duplicate (post-verification)', 'generated': res['generated']}
                    else:
                        dup_index.add(c.domain, res['generated']['context_O'], res['generated']['options'])
                        kept_count[c.family] += 1
            done[c.uid] = res
        state['meter'] = METER
        STATE.write_text(json.dumps(state, ensure_ascii=False))
        el = (time.time() - t0) / 60
        total_kept, total_target = sum(kept_count.values()), sum(TARGETS.values())
        new_kept = total_kept - kept_at_start
        eta = (el / new_kept * (total_target - total_kept)) if new_kept else float('nan')
        print(f"kept {dict(kept_count)} ({total_kept}/{total_target}) | examined {len(done)} | ${METER['usd']:.2f} | "
              f"{METER['calls']} calls | {el:.1f} min | ETA ~{eta:.0f} min", flush=True)
stages = Counter(r.get('stage') for r in done.values())
print('\\nSTOPPED AT BUDGET CAP' if halted else '\\nfinished', '| stages:', dict(stages))
print('top discard reasons:', Counter(r.get('reason') for r in done.values() if r.get('stage') != 'kept').most_common(12))
""")

code("""# ---- splits, item files, Opus audit on a seeded 10% sample -----------------------------------------------
kept_all = sorted([r for r in done.values() if r.get('stage') == 'kept'], key=lambda r: r['meta']['uid'])
# parallel workers can overshoot a family's target by a few scenarios; keep exactly the target (lowest uids first)
kept, per_fam = [], Counter()
for r in kept_all:
    f = r['meta']['family']
    if per_fam[f] < TARGETS[f]:
        kept.append(r); per_fam[f] += 1
print('kept per family (trimmed to targets):', dict(per_fam), '| overshoot discarded:', len(kept_all) - len(kept))
splits = nb.assign_splits(kept, SEED)
items = nb.to_items(kept, splits, SEED)
for split in ('train', 'dev', 'test'):
    with open(OUT / f'{split}.jsonl', 'w', encoding='utf-8') as fh:
        for it in items:
            if it['split'] == split:
                fh.write(json.dumps(it, ensure_ascii=False) + '\\n')
with open(OUT / 'scenarios.jsonl', 'w', encoding='utf-8') as fh:
    for r in sorted(kept, key=lambda x: x['meta']['uid']):
        fh.write(json.dumps({**r['meta'], 'split': splits[r['meta']['uid']], **r['generated'], 'verdicts': r['verdicts']}, ensure_ascii=False) + '\\n')

rng = random.Random(SEED + 7)
audit_ids = sorted(rng.sample(sorted(r['meta']['uid'] for r in kept), max(1, int(round(AUDIT_FRACTION * len(kept))))))
AUDIT = OUT / 'opus_audit.json'
audit = json.loads(AUDIT.read_text()) if AUDIT.exists() else {}
by_uid = {r['meta']['uid']: r for r in kept}
for uid in audit_ids:
    if uid in audit:
        continue
    try:
        reason, verdicts = nb.verify_arms(by_uid[uid]['generated'], call_low, AUDIT_MODEL)
    except BudgetExceeded:
        print('budget cap reached during audit'); break
    audit[uid] = {'agrees': reason is None, 'reason': reason, 'verdicts': verdicts}
    AUDIT.write_text(json.dumps(audit, indent=1, ensure_ascii=False))
    state['meter'] = METER; STATE.write_text(json.dumps(state, ensure_ascii=False))
agree = sum(v['agrees'] for v in audit.values())
print(f'Opus audit: {agree}/{len(audit)} scenarios agree on all four arms ({100*agree/max(1,len(audit)):.1f}%)')
""")

code("""# ---- report, dataset card, examples, manifest, zip --------------------------------------------------------
counts = Counter((it['split'], it['family']) for it in items if it['arm'] == 'O')
report = {
    'built_utc': datetime.now(timezone.utc).isoformat(), 'cacsa2_version': cacsa2.__version__,
    'models': {'generator': GEN_MODEL, 'generator_effort': GEN_EFFORT, 'verifier': VERIFY_MODEL, 'audit': AUDIT_MODEL},
    'prompt_sha256': nb.prompt_hashes(), 'targets': TARGETS, 'seed': SEED,
    'funnel': dict(Counter(r.get('stage') for r in done.values())),
    'discard_reasons': dict(Counter(r.get('reason') for r in done.values() if r.get('stage') != 'kept').most_common(40)),
    'kept_scenarios': len(kept), 'items': len(items),
    'scenarios_by_split_and_family': {f'{s}/{f}': n for (s, f), n in sorted(counts.items())},
    'label_balance_by_split': {s: dict(Counter(it['label'] for it in items if it['split'] == s)) for s in ('train', 'dev', 'test')},
    'opus_audit': {'n': len(audit), 'agree_all_arms': agree},
    'cost_usd': round(METER['usd'], 2), 'api_calls': METER['calls'],
}
(OUT / 'build_report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
(OUT / 'README.md').write_text(f'''# {OUT_NAME}

Controlled negation minimal pairs in the CACSA format, built {report['built_utc'][:10]} with cacsa2 {cacsa2.__version__}.
Each scenario: context, question, two options; arms O (affirmative), P (paraphrase), N (one negation cue on the decisive
words; answer flips), S (one negation cue on an irrelevant detail; answer unchanged). Families syntactic and quantifier
are seen (train/dev/test, 60/15/25 by scenario); affixal and lexical are unseen (test only).

Generated by {GEN_MODEL} and verified arm by arm, blind, by {VERIFY_MODEL}; a seeded {int(AUDIT_FRACTION*100)}% sample
was re-verified by {AUDIT_MODEL} ({agree}/{len(audit)} scenarios agreed on all arms). Not human-verified.
Kept {len(kept)} scenarios, {len(items)} items; cost ${report['cost_usd']}. Prompts are in cacsa2/negpairs_build.py
(SHA-256 in build_report.json). Files: train/dev/test.jsonl (one row per arm), scenarios.jsonl, build_state.json
(every candidate and the reason it was discarded), opus_audit.json, build_report.json, FILE_MANIFEST_SHA256.csv.
''')
print(json.dumps({k: report[k] for k in ('funnel', 'kept_scenarios', 'items', 'scenarios_by_split_and_family', 'opus_audit', 'cost_usd')}, indent=1))
print('\\nEXAMPLES (first kept scenario of each family):')
for fam_name in nb.FAMILIES:
    ex = next((r for r in kept if r['meta']['family'] == fam_name), None)
    if ex:
        g = ex['generated']
        print(f"\\n[{fam_name}] Q: {g['question']}  A) {g['options'][0]}  B) {g['options'][1]}  answer_O={g['answer_O']}")
        for arm in 'OPNS':
            print(f"   {arm}: {g['context_' + arm]}")
n = provenance.write_file_manifest(OUT, OUT / 'FILE_MANIFEST_SHA256.csv')
ZIP = Path('/content') / f'{OUT_NAME}.zip'
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(x for x in OUT.rglob('*') if x.is_file()):
        z.write(p, arcname=str(p.relative_to(OUT.parent)))
print(f'\\n{n} files manifested -> {ZIP} ({ZIP.stat().st_size // 1024} kB)')
from google.colab import files
files.download(str(ZIP))
""")

nb_json = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                        "language_info": {"name": "python"}, "colab": {"provenance": []}},
           "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "NegPairs_build_colab_v1.ipynb"
out.write_text(json.dumps(nb_json, indent=1, ensure_ascii=False))
print(out, len(cells), "cells")
