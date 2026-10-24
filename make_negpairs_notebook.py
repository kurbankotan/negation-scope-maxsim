"""Generate the Colab notebook that builds the NegPairs dataset, construction v2 (CPU runtime + ANTHROPIC_API_KEY).

v2 continues in the same Drive folder as the v1 run: candidates that v1 already generated are re-checked under the v2
filters and verifier (cheap verifier calls only); candidates that v1 lost to empty or cut-off generator responses, and
the rest of the schedule, are generated with the unchanged generation prompt at effort "low" and a 4,000-token limit.
"""
import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# NegPairs builder, construction v2

**Runtime:** CPU is enough. `ANTHROPIC_API_KEY` in the Colab **Secrets** panel with *Notebook access* on.
`cacsa2_project.zip` (0.2.4 or later) in `MyDrive/CACSA_v2/`. Then `Runtime > Run all`.

**Why v2.** The first run kept only 7% of the candidates. Most losses were pipeline problems, not bad scenarios:
generator answers cut off by its thinking (token limit too low), two filter bugs, an edit limit of three words for N
that most natural negations exceed, and a verifier wording that rejected clear items. v2 fixes these and keeps the
generation prompt unchanged.

**What it does.** Phase A re-checks every candidate the first run already generated (verifier calls only, cheap).
Phase B generates the rest of the schedule. Everything is saved in Drive after every few candidates; re-running
continues where it stopped and never pays twice. **The run stops by itself at the session budget** below.

| Step | Expected |
|---|---|
| setup + preflight | 1–2 min |
| phase A (re-check of the first run's candidates) | 10–20 min, about 1–2 USD |
| phase B (new candidates) | 1–2 h; a line every few seconds with kept counts, cost and projection |
| splits, Opus audit, report, zip | 5–10 min |

**Status of the data.** Model-generated and model-verified, **not human-verified**.
""")

code("""# ============================== CONFIGURATION ==============================
PROJECT_ROOT  = "/content/drive/MyDrive/CACSA_v2"
PACKAGE_ZIP   = f"{PROJECT_ROOT}/cacsa2_project.zip"
OUT_NAME      = "negpairs_v1"          # same folder as the first run: its record is reused, not paid again
TARGETS       = {"syntactic": 500, "quantifier": 500, "affixal": 250, "lexical": 250}   # kept scenarios per family
OVERSAMPLE    = 2.2                    # same candidate schedule as the first run
SESSION_BUDGET_USD = 22.00             # hard stop for NEW spending in this session (the first run's cost is not included)
AUDIT_RESERVE_USD  = 3.00              # part of the session budget kept for the Opus audit at the end
GEN_MODEL, GEN_EFFORT, GEN_MAX_TOKENS = "claude-sonnet-5", "low", 4000
VERIFY_MODEL  = "claude-haiku-4-5-20251001"
AUDIT_MODEL, AUDIT_MAX_TOKENS = "claude-opus-5", 2000
AUDIT_FRACTION = 0.10
MAX_WORKERS   = 6
SEED          = 20261003
# ===========================================================================
""")

code("""%pip install -q anthropic wordfreq
from google.colab import drive
drive.mount('/content/drive')
import os, sys, json, zipfile, time, random, threading, hashlib, shutil, importlib
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
ROOT = Path(PROJECT_ROOT)
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
print('cacsa2', cacsa2.__version__, '| zip sha256', _zip_sha[:12])
assert _v >= (0, 2, 4), f'cacsa2_project.zip on Drive is version {cacsa2.__version__}; upload the newest cacsa2_project.zip (>= 0.2.4)'
from cacsa2 import negpairs_build as nb, provenance
OUT = ROOT / 'data' / OUT_NAME; OUT.mkdir(parents=True, exist_ok=True)
STATE = OUT / 'build_state.json'
API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
if not API_KEY:
    try:
        from google.colab import userdata
        API_KEY = userdata.get('ANTHROPIC_API_KEY')
    except Exception:
        pass
assert API_KEY, 'No API key: Secrets panel -> add ANTHROPIC_API_KEY -> switch on Notebook access.'
print('output folder', OUT, '| existing record:', STATE.exists())
""")

code("""# ---- record (first run kept as 'v1', this run as 'v2'), metered API calls, session budget, preflight -----------
state = json.loads(STATE.read_text()) if STATE.exists() else {}
if 'done' in state and 'v1' not in state:                       # first v2 start: archive the v1 record unchanged
    backup = OUT / 'build_state_v1_backup.json'
    if not backup.exists():
        backup.write_text(STATE.read_text())
    state['v1'] = {'done': state.pop('done'), 'settings': {'generator': 'claude-sonnet-5', 'generator_effort': 'medium',
                   'generator_max_tokens': 1200, 'verifier': 'claude-haiku-4-5-20251001', 'verify_prompt': 'VERIFY_PROMPT',
                   'filter': 'structural_filter (N edit <= 3, S edit <= 8)'}}
state.setdefault('v1', {'done': {}})
state.setdefault('v2', {'done': {}})
state['v2']['settings'] = {'generator': GEN_MODEL, 'generator_effort': GEN_EFFORT, 'generator_max_tokens': GEN_MAX_TOKENS,
                           'verifier': VERIFY_MODEL, 'verifier_temperature': 0.0, 'verify_prompt': 'VERIFY_PROMPT_V2',
                           'filter': f"structural_filter_v2 {nb.V2_SETTINGS}", 'prompt_sha256': nb.prompt_hashes()}
METER = {'usd': 0.0, 'calls': 0, 'in': 0, 'out': 0}
METER.update(state.get('meter', {}))
def save_state():
    state['meter'] = METER
    tmp = STATE.with_suffix('.tmp'); tmp.write_text(json.dumps(state, ensure_ascii=False)); tmp.replace(STATE)
save_state()
print(f"first run record: {len(state['v1']['done'])} candidates | v2 record: {len(state['v2']['done'])} | metered so far ${METER['usd']:.2f}")

from anthropic import Anthropic
PRICES = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0), "claude-haiku-4-5-20251001": (1.0, 5.0)}
SUPPORTS_EFFORT = {"claude-opus-5", "claude-sonnet-5"}
_client = Anthropic(api_key=API_KEY, max_retries=6, timeout=180.0)
_lock = threading.Lock()
SESSION_START_USD = METER['usd']
CAP_USD = SESSION_START_USD + SESSION_BUDGET_USD           # audit may spend up to here
BUILD_CAP_USD = CAP_USD - AUDIT_RESERVE_USD                # generation and verification stop here

class BudgetExceeded(Exception):
    pass
class CreditExhausted(Exception):
    pass

def _raw(model, prompt, max_tokens, effort, temperature):
    kw = dict(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
    if model in SUPPORTS_EFFORT and effort:
        kw["output_config"] = {"effort": effort}
    if temperature is not None:
        kw["temperature"] = temperature
    for _ in range(3):
        try:
            return _client.messages.create(**kw)
        except Exception as e:
            msg = str(e).lower()
            if 'credit balance' in msg or 'billing' in msg:
                raise CreditExhausted(str(e)[:200])
            if ('output_config' in msg or 'effort' in msg) and 'output_config' in kw:
                kw.pop('output_config'); continue
            if 'temperature' in msg and 'temperature' in kw:
                kw.pop('temperature'); continue
            raise
    return _client.messages.create(**kw)

def make_call(effort, temperature=None, cap=None):
    def call(model, prompt, max_tokens):
        with _lock:
            if METER['usd'] >= (cap if cap is not None else BUILD_CAP_USD):
                raise BudgetExceeded()
        r = _raw(model, prompt, max_tokens, effort, temperature)
        pin, pout = PRICES.get(model, (5.0, 25.0))
        with _lock:
            METER['usd'] += (r.usage.input_tokens * pin + r.usage.output_tokens * pout) / 1e6
            METER['calls'] += 1; METER['in'] += r.usage.input_tokens; METER['out'] += r.usage.output_tokens
        text = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
        if not text.strip():
            raise ValueError(f"no text block in response (stop_reason={getattr(r, 'stop_reason', '?')})")
        return text
    return call

call_gen = make_call(GEN_EFFORT)                  # generator
call_ver = make_call(None, temperature=0.0)       # verifier (deterministic)
call_audit = make_call("low", cap=CAP_USD)        # Opus audit (may use the reserve)

# preflight: one generation under v2 settings, inspected before the batch
work = nb.schedule(TARGETS, OVERSAMPLE, SEED)
BY_UID = {c.uid: c for c in work}
c0 = next(c for c in work if c.uid not in state['v1']['done']) if len(state['v1']['done']) < len(work) else work[0]
usd0 = METER['usd']
r0 = nb.process_v2(c0, call_gen, call_ver, GEN_MODEL, VERIFY_MODEL, None, GEN_MAX_TOKENS)
print('PREFLIGHT', c0.uid, '->', r0.get('stage'), r0.get('reason', ''))
if r0.get('generated'):
    print(json.dumps(r0['generated'], indent=1, ensure_ascii=False))
print(f"preflight cost ${METER['usd'] - usd0:.4f} | session budget ${SESSION_BUDGET_USD:.2f}: building stops at metered "
      f"${BUILD_CAP_USD:.2f}, the audit at ${CAP_USD:.2f}")
""")

code("""# ---- phase A: re-check under v2 every candidate the first run already generated (verifier calls only) ----------
v1d, v2d = state['v1']['done'], state['v2']['done']
dup_index = nb.NearDuplicateIndex(threshold=0.6)
kept_count = Counter()
for uid, res in sorted(v2d.items()):
    if res.get('stage') == 'kept':
        dup_index.add(res['meta']['domain'], res['generated']['context_O'], res['generated']['options'])
        kept_count[res['meta']['family']] += 1
dup_lock = threading.Lock()
def is_dup(domain, ctx, opts):
    with dup_lock:
        return dup_index.is_duplicate(domain, ctx, opts)

def record(c, res):
    if res.get('stage') == 'kept':
        with dup_lock:
            if dup_index.is_duplicate(c.domain, res['generated']['context_O'], res['generated']['options']):
                res = {**res, 'stage': 'duplicate', 'reason': 'near-duplicate (post-verification)'}
            else:
                dup_index.add(c.domain, res['generated']['context_O'], res['generated']['options'])
                kept_count[c.family] += 1
    v2d[c.uid] = res

def run_chunks(items, task, label, project=False):
    \"\"\"items: list of Candidate; task(c) -> record. Saves after every chunk; stops on budget, credit or repeated errors.\"\"\"
    t0, done_here, kept_here, halted = time.time(), 0, 0, None
    usd_start = METER['usd']
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        i = 0
        while i < len(items) and not halted:
            if all(kept_count[f] >= n for f, n in TARGETS.items()):
                halted = 'targets reached'; break
            chunk = []
            while i < len(items) and len(chunk) < MAX_WORKERS * 2:
                c = items[i]; i += 1
                if kept_count[c.family] < TARGETS[c.family] and c.uid not in v2d:
                    chunk.append(c)
            errors = []
            for c, f in [(c, pool.submit(task, c)) for c in chunk]:
                try:
                    res = f.result()
                except BudgetExceeded:
                    halted = 'session budget reached'; continue
                except CreditExhausted as e:
                    halted = f'API credit exhausted: {e}'; continue
                except Exception as e:                       # not recorded: retried on the next run
                    errors.append(f'{type(e).__name__}: {str(e)[:100]}'); continue
                before = kept_count[c.family]
                record(c, res)
                done_here += 1; kept_here += kept_count[c.family] - before
            save_state()
            if len(errors) >= max(3, len(chunk) // 2):
                halted = f'repeated API errors, e.g. {errors[0]}'
            spent = METER['usd'] - usd_start
            need = sum(max(0, n - kept_count[f]) for f, n in TARGETS.items())
            rate = kept_here / done_here if done_here else 0.0
            proj = f" | to reach all targets ~${spent / kept_here * need:.0f} more" if (project and kept_here) else ''
            print(f"[{label}] kept {dict(kept_count)} ({sum(kept_count.values())}/{sum(TARGETS.values())}) | checked {done_here} "
                  f"| yield {100*rate:.0f}% | +${spent:.2f} (session ${METER['usd'] - SESSION_START_USD:.2f} of {SESSION_BUDGET_USD:.0f})"
                  f"{proj} | {(time.time()-t0)/60:.1f} min" + (f" | errors {len(errors)}" if errors else ''), flush=True)
    return halted

todo = [c for c in work if c.uid in v1d and c.uid not in v2d and v1d[c.uid].get('generated')]   # schedule order: families interleaved
print(f'phase A: {len(todo)} candidates from the first run to re-check')
halted_A = run_chunks(todo, lambda c: nb.check_and_verify_v2(v1d[c.uid]['generated'], c, call_ver, VERIFY_MODEL, is_dup,
                                                             source='v1-generated'), 'A')
print('phase A ended:', halted_A or 'all checked', '| stages:', dict(Counter(r['stage'] for r in v2d.values())))
""")

code("""# ---- phase B: generate the rest of the schedule (and the first run's empty or cut-off generations) --------------
queue = [c for c in work if c.uid not in v2d and not (v1d.get(c.uid) or {}).get('generated')]   # first-run texts stay in phase A
print(f'phase B: {len(queue)} candidates in the queue')
halted_B = None
if halted_A in (None, 'targets reached'):
    halted_B = run_chunks(queue, lambda c: nb.process_v2(c, call_gen, call_ver, GEN_MODEL, VERIFY_MODEL, is_dup, GEN_MAX_TOKENS),
                          'B', project=True)
else:
    print('phase B skipped because phase A stopped:', halted_A)
if halted_A in (None, 'targets reached'):
    print('\\nphase B ended:', halted_B or 'queue finished')
print('v2 stages:', dict(Counter(r['stage'] for r in v2d.values())))
print('top v2 discard reasons:', Counter(r.get('reason') for r in v2d.values() if r.get('stage') != 'kept').most_common(12))
print(f"metered total ${METER['usd']:.2f} (this session ${METER['usd'] - SESSION_START_USD:.2f})")
""")

code("""# ---- splits, item files, Opus audit on a seeded 10% sample -----------------------------------------------
kept_all = sorted([r for r in v2d.values() if r.get('stage') == 'kept'], key=lambda r: r['meta']['uid'])
kept, per_fam = [], Counter()
for r in kept_all:                                   # parallel workers can overshoot a target by a few
    f = r['meta']['family']
    if per_fam[f] < TARGETS[f]:
        kept.append(r); per_fam[f] += 1
print('kept per family:', dict(per_fam), '| overshoot discarded:', len(kept_all) - len(kept))
splits = nb.assign_splits(kept, SEED)
items = nb.to_items(kept, splits, SEED)
for split in ('train', 'dev', 'test'):
    with open(OUT / f'{split}.jsonl', 'w', encoding='utf-8') as fh:
        for it in items:
            if it['split'] == split:
                fh.write(json.dumps(it, ensure_ascii=False) + '\\n')
with open(OUT / 'scenarios.jsonl', 'w', encoding='utf-8') as fh:
    for r in kept:
        uid = r['meta']['uid']
        fh.write(json.dumps({**r['meta'], 'split': splits[uid], 'source': r.get('source'), **r['generated'],
                             'verdicts_v2': r['verdicts'], 'v1_outcome': (v1d.get(uid) or {}).get('stage')}, ensure_ascii=False) + '\\n')

rng = random.Random(SEED + 7)
audit_ids = sorted(rng.sample(sorted(r['meta']['uid'] for r in kept), max(1, int(round(AUDIT_FRACTION * len(kept)))))) if kept else []
AUDIT = OUT / 'opus_audit_v2.json'
audit = json.loads(AUDIT.read_text()) if AUDIT.exists() else {}
by_uid = {r['meta']['uid']: r for r in kept}
for uid in audit_ids:
    if uid in audit:
        continue
    try:
        reason, verdicts = nb.verify_arms_v2(by_uid[uid]['generated'], call_audit, AUDIT_MODEL, max_tokens=AUDIT_MAX_TOKENS)
    except (BudgetExceeded, CreditExhausted) as e:
        print('audit stopped:', type(e).__name__); break
    except Exception as e:
        print('audit call failed for', uid, type(e).__name__, str(e)[:100]); continue
    audit[uid] = {'agrees': reason is None, 'reason': reason, 'verdicts': verdicts}
    AUDIT.write_text(json.dumps(audit, indent=1, ensure_ascii=False)); save_state()
audit_cur = {u: audit[u] for u in audit_ids if u in audit}        # the current seeded sample only
agree = sum(v['agrees'] for v in audit_cur.values())
print(f'Opus audit (v2 prompt): {agree}/{len(audit_cur)} sampled scenarios agree on all four arms '
      f'({100*agree/max(1,len(audit_cur)):.1f}%; sample size {len(audit_ids)})')
""")

code("""# ---- report, dataset card, examples, manifest, zip --------------------------------------------------------
counts = Counter((it['split'], it['family']) for it in items if it['arm'] == 'O')
src = Counter((r.get('source'), r.get('stage')) for r in v2d.values())
report = {
    'built_utc': datetime.now(timezone.utc).isoformat(), 'cacsa2_version': cacsa2.__version__, 'construction': 'v2',
    'v1': {'settings': state['v1'].get('settings'), 'funnel': dict(Counter(r.get('stage') for r in v1d.values())),
           'candidates': len(v1d)},
    'v2': {'settings': state['v2']['settings'], 'funnel': dict(Counter(r.get('stage') for r in v2d.values())),
           'funnel_by_source': {f'{s}/{t}': n for (s, t), n in sorted(src.items(), key=str)},
           'discard_reasons': dict(Counter(r.get('reason') for r in v2d.values() if r.get('stage') != 'kept').most_common(40))},
    'models': {'generator': GEN_MODEL, 'verifier': VERIFY_MODEL, 'audit': AUDIT_MODEL},
    'prompt_sha256': nb.prompt_hashes(), 'targets': TARGETS, 'seed': SEED,
    'kept_scenarios': len(kept), 'items': len(items),
    'scenarios_by_split_and_family': {f'{s}/{f}': n for (s, f), n in sorted(counts.items())},
    'label_balance_by_split_and_arm': {f'{s}/{a}': dict(Counter(it['label'] for it in items if it['split'] == s and it['arm'] == a))
                                       for s in ('train', 'dev', 'test') for a in 'OPNS'},
    'opus_audit_v2': {'sample': len(audit_ids), 'checked': len(audit_cur), 'agree_all_arms': agree},
    'cost_usd_total_metered': round(METER['usd'], 2), 'cost_usd_this_session': round(METER['usd'] - SESSION_START_USD, 2),
    'api_calls': METER['calls'], 'stopped': {'phase_A': halted_A, 'phase_B': halted_B},
}
(OUT / 'build_report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
(OUT / 'README.md').write_text(f'''# NegPairs (folder {OUT_NAME}), construction v2

Controlled negation minimal pairs in the CACSA format, built {report['built_utc'][:10]} with cacsa2 {cacsa2.__version__}.
Each scenario: context, question, two options; arms O (affirmative), P (paraphrase), N (one negation cue on the decisive
words; answer flips), S (one negation cue on an irrelevant detail; answer unchanged). Families syntactic and quantifier
are seen (train/dev/test, 60/15/25 by scenario); affixal and lexical are unseen (test only).

Generation: {GEN_MODEL} with an unchanged prompt (first run at effort medium and a 1,200-token limit; this run at effort
{GEN_EFFORT} and {GEN_MAX_TOKENS} tokens). Every kept scenario passed the v2 filters and was verified arm by arm, blind,
by {VERIFY_MODEL} (temperature 0) with the v2 verifier prompt, including the scenarios generated in the first run (their
first-run verdicts are archived, not reused). A seeded {int(AUDIT_FRACTION*100)}% sample was re-verified by {AUDIT_MODEL}
({agree}/{len(audit_cur)} checked scenarios agreed on all arms). Not human-verified.
Kept {len(kept)} scenarios, {len(items)} items; metered cost ${report['cost_usd_total_metered']} in total. Prompts are in
cacsa2/negpairs_build.py (SHA-256 in build_report.json). Files: train/dev/test.jsonl (one row per arm), scenarios.jsonl,
build_state.json (every candidate of both runs and the reason it was discarded), build_state_v1_backup.json (the first
run's record as it was), opus_audit_v2.json, build_report.json, FILE_MANIFEST_SHA256.csv.
''')
print(json.dumps({k: report[k] for k in ('kept_scenarios', 'items', 'scenarios_by_split_and_family', 'opus_audit_v2',
                                         'cost_usd_total_metered', 'cost_usd_this_session', 'stopped')}, indent=1))
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
    for p in sorted(x for x in OUT.rglob('*') if x.is_file() and x.suffix != '.tmp'):
        z.write(p, arcname=str(p.relative_to(OUT.parent)))
print(f'\\n{n} files manifested -> {ZIP} ({ZIP.stat().st_size // 1024} kB)')
from google.colab import files
files.download(str(ZIP))
""")

nb_json = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                        "language_info": {"name": "python"}, "colab": {"provenance": []}},
           "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "NegPairs_build_colab.ipynb"
out.write_text(json.dumps(nb_json, indent=1, ensure_ascii=False))
print(out, len(cells), "cells")
