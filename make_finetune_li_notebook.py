"""Generate NegPairs_finetune_li_colab.ipynb: released late-interaction checkpoints fine-tuned on minimal pairs
(NegPairs v1 train or NevIR train) and scored like the released systems on NegPairs-Perm, NegPairs-Fresh, NevIR test
and BEIR SciFact (revision step 3, October 2026). The cacsa2 package and the NegPairs v1 splits are embedded in the
notebook; nothing is uploaded.
"""
import base64
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
DATA_V3 = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "np_data_v3" / "negpairs_v3"
DATA_V1 = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "np_data" / "negpairs_v1"
V3_SHA = {r["path"]: r["sha256"] for r in csv.DictReader(open(DATA_V3 / "FILE_MANIFEST_SHA256.csv", encoding="utf-8"))}
NEVIR_REVISION = "6263585072ce3b435ed09658613553fbf4e74184"
NEVIR_COUNTS = {"instances": 1383, "nested": 184, "equal_bags": 36}
EXCLUDE_TOP = ("paper", "human_check", "indep", "data")


def runtime_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for q in sorted(HERE.rglob("*")):
            rel = q.relative_to(HERE)
            if not q.is_file() or "__pycache__" in q.parts or ".pytest_cache" in q.parts or q.suffix in (".pyc", ".ipynb"):
                continue
            if rel.parts[0] in EXCLUDE_TOP:
                continue
            z.write(q, f"cacsa2_project/{rel}")
        for q in sorted(DATA_V1.iterdir()):                       # NegPairs v1 splits travel with the notebook
            if q.is_file() and q.suffix in (".jsonl", ".csv", ".md", ".json"):
                z.write(q, f"cacsa2_project/data/negpairs_v1/{q.name}")
    return buf.getvalue()


PKG = runtime_zip_bytes()
PKG_SHA = hashlib.sha256(PKG).hexdigest()
PKG_B64 = base64.b64encode(PKG).decode("ascii")
PKG_B64_WRAPPED = "\n".join(PKG_B64[i:i + 120] for i in range(0, len(PKG_B64), 120))

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# Late-interaction checkpoints fine-tuned on minimal pairs (revision step 3)

**Runtime:** **L4 GPU** is enough (about 2 hours); an **A100** takes about half. Nothing to upload: the `cacsa2` package and the
NegPairs v1 splits are embedded (cell 2). Drive is used for `data/negpairs_v3/` (Perm and Fresh, already there) and for the
results folder `runs/finetune_li/`, which is reused across sessions (finished runs are skipped).

What is done. Two released ColBERT checkpoints (colbert-ir/colbertv2.0 and answerdotai/answerai-colbert-small-v1) are
trained end to end with their own scoring (query = option or query, document = context + question or document; sum over
query vectors of the maximum cosine) on either NegPairs v1 train (1,692 items of the two seen families; selection on the
v1 dev accuracy) or NevIR train (selection on the NevIR validation paired accuracy), three seeds each. Every trained model
and the released checkpoint are then scored exactly like the released systems on NegPairs-Perm (274 scenarios), NegPairs-Fresh
(240) and NevIR test (1,383 instances, nested subset 184), and on BEIR SciFact (300 test queries, 5,183 documents, brute-force
MaxSim, nDCG@10) as a standard-retrieval check. Metrics are the article's (SS_perm_bal with scenario-cluster intervals;
paired accuracy with strict comparisons).

| Phase | L4 |
|---|---|
| setup, data, NevIR, SciFact download | 5 min |
| baselines (2 models: NegPairs, NevIR, SciFact) | 8 min |
| 12 fine-tuning runs (2 models x 2 training sets x 3 seeds), each with its evaluation | 90 to 110 min |
| metrics, tables, zip | 3 min |

At the end `finetune_li_results.zip` is downloaded.
""")

code(f"""# ============================== CONFIGURATION ==============================
PROJECT_ROOT  = "/content/drive/MyDrive/CACSA_v3"
PACKAGE_SHA256 = "{PKG_SHA}"            # the cacsa2 package embedded in the next cell (nothing to upload)
DATA_V3_DIR   = f"{{PROJECT_ROOT}}/data/negpairs_v3"
DATA_V3_SHA256 = {json.dumps({k: V3_SHA[k] for k in ("perm.jsonl", "fresh.jsonl")}, indent=1)}
RUN_NAME      = "finetune_li"
NEVIR         = {{"dataset": "orionweller/NevIR", "revision": "{NEVIR_REVISION}"}}
NEVIR_COUNTS  = {json.dumps(NEVIR_COUNTS)}
MODELS = [
    {{"name": "colbertv2", "repo": "colbert-ir/colbertv2.0", "revision": "c1e84128e85ef755c096a95bdb06b47793b13acf", "lr": 1e-5, "batch": 16, "label": "ColBERTv2"}},
    {{"name": "answerai-colbert-small-v1", "repo": "answerdotai/answerai-colbert-small-v1", "lr": 2e-5, "batch": 16, "label": "answerai-colbert-small-v1"}},
]
TRAIN_SETS    = ["negpairs", "nevir"]
SEEDS         = [13, 42, 71]
EPOCHS, PATIENCE, WARMUP = 5, 2, 0.06
SCIFACT       = True
N_BOOT        = 2000           # per run; the seed means are the quantities of interest
SKIP          = []             # run names "model/train_set/seedN" to skip
# ===========================================================================
""")

code('# ============================== EMBEDDED PACKAGE (cacsa2 ' + PKG_SHA[:12] + ' with NegPairs v1; generated, do not edit) ==============================\n'
     'PACKAGE_B64 = """\n' + PKG_B64_WRAPPED + '\n"""\n')

code(r"""from google.colab import drive
drive.mount('/content/drive')
import os, sys, json, zipfile, time, csv, shutil, importlib, hashlib, subprocess, traceback, gc, base64, io
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import torch
ROOT = Path(PROJECT_ROOT)
PKG_BYTES = base64.b64decode(''.join(PACKAGE_B64.split()))
ZIP_SHA = hashlib.sha256(PKG_BYTES).hexdigest()
assert ZIP_SHA == PACKAGE_SHA256, 'the embedded package does not match its recorded SHA-256'
PKG_DIR = Path('/content') / f'cacsa2_pkg_{ZIP_SHA[:12]}'
shutil.rmtree(PKG_DIR, ignore_errors=True)
with zipfile.ZipFile(io.BytesIO(PKG_BYTES)) as z:
    z.extractall(PKG_DIR)
for _m in [m for m in list(sys.modules) if m == 'cacsa2' or m.startswith('cacsa2.')]:
    del sys.modules[_m]
sys.path = [p for p in sys.path if 'cacsa2_pkg_' not in p]
PKG_ROOT = next(PKG_DIR.rglob('cacsa2/__init__.py')).parent.parent
sys.path.insert(0, str(PKG_ROOT))
importlib.invalidate_caches()
import cacsa2
assert tuple(int(x) for x in cacsa2.__version__.split('.')) >= (0, 3, 5), f'package {cacsa2.__version__}'
from cacsa2 import negpairs as npx, nevir, exact, provenance, public_systems as ps, finetune_li as ft
RUN = ROOT / 'runs' / RUN_NAME
SCORES = RUN / 'scores'; SCORES.mkdir(parents=True, exist_ok=True)
if not (RUN / f'cacsa2_runtime_{ZIP_SHA[:12]}.zip').exists():
    (RUN / f'cacsa2_runtime_{ZIP_SHA[:12]}.zip').write_bytes(PKG_BYTES)
GPU = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip() if shutil.which('nvidia-smi') else 'none'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
def save_json(name, obj):
    (RUN / name).write_text(json.dumps(obj, indent=1, default=str))
env = provenance.run_environment({'run_name': RUN_NAME, 'cacsa2_version': cacsa2.__version__, 'package_zip_sha256': ZIP_SHA,
                                  'config': {'models': MODELS, 'train_sets': TRAIN_SETS, 'seeds': SEEDS, 'epochs': EPOCHS, 'patience': PATIENCE, 'warmup': WARMUP,
                                             'scifact': SCIFACT, 'n_boot': N_BOOT, 'nevir': NEVIR}})
ENVF = RUN / 'run_environment.json'
sessions = json.loads(ENVF.read_text()) if ENVF.exists() else []
sessions.append(env); ENVF.write_text(json.dumps(sessions, indent=1))
print('cacsa2', cacsa2.__version__, '| package', ZIP_SHA[:12], '| GPU:', GPU, '| run folder', RUN, '| session', len(sessions))
""")

code(r"""# ---- data: NegPairs v1 train/dev (embedded), Perm and Fresh (Drive, hashed), NevIR train/validation/test, SciFact ------
V1 = PKG_ROOT / 'data' / 'negpairs_v1'
for row in csv.DictReader(open(V1 / 'FILE_MANIFEST_SHA256.csv', encoding='utf-8')):
    assert provenance.sha256_file(V1 / row['path']) == row['sha256'], f"NegPairs v1 file changed: {row['path']}"
TRAIN, DEV = npx.load_split(V1, 'train'), npx.load_split(V1, 'dev')
D3 = Path(DATA_V3_DIR)
for name, sha in DATA_V3_SHA256.items():
    assert provenance.sha256_file(D3 / name) == sha, f'{name} differs from the audited NegPairs v3.0 file'
TESTS = {'perm': npx.load_split(D3, 'perm'), 'fresh': npx.load_split(D3, 'fresh')}
BY_ID = {name: {r.sample_id: r for r in rows} for name, rows in TESTS.items()}
NV = {split: nevir.load_test(NEVIR['dataset'], split, revision=NEVIR['revision']) for split in ('train', 'validation', 'test')}
NESTED = exact.nested_instances(NV['test']); NESTED_IDS = [i['id'] for i in NESTED]; EQUAL_IDS = [i['id'] for i in NESTED if i['contained'] == 'equal']
got = {'instances': len(NV['test']), 'nested': len(NESTED), 'equal_bags': len(EQUAL_IDS)}
assert got == NEVIR_COUNTS, f'NevIR test differs from the main run: {got} vs {NEVIR_COUNTS}'
COMBOS = nevir.four_combos(NV['test'])
NP_ITEMS = {name: [(o, r.context, r.question) for r in rows for o in r.options] for name, rows in TESTS.items()}
SF = ft.load_scifact('/content/beir') if SCIFACT else None
audit = {'v1': {'train_items': len(TRAIN), 'train_scenarios': len({r.passage_id for r in TRAIN}), 'dev_items': len(DEV)},
         'perm': {'items': len(TESTS['perm'])}, 'fresh': {'items': len(TESTS['fresh'])},
         'nevir': {**got, 'train': len(NV['train']), 'validation': len(NV['validation']), 'revision': NEVIR['revision']},
         'scifact': (SF[3] if SF else None)}
save_json('data_audit.json', audit)
print(json.dumps(audit, indent=1))
def score_all(sc):
    res = {'negpairs': {'score': {}}, 'nevir': {'score': {}}}
    for sname, rows in TESTS.items():
        vals = sc.score_negpairs(NP_ITEMS[sname], batch_size=32)
        res['negpairs']['score'][sname] = {r.sample_id: [vals[2 * k], vals[2 * k + 1]] for k, r in enumerate(rows)}
    vals = sc.score_nevir([(c[2], c[3]) for c in COMBOS], batch_size=32)
    res['nevir']['score'] = nevir.to_score_dict(COMBOS, vals)
    if SF:
        sfr = ft.scifact_ndcg(sc, SF[0], SF[1], SF[2])
        res['scifact'] = {'ndcg@10': sfr['ndcg@10'], 'n_queries': sfr['n_queries'], 'n_docs': sfr['n_docs'], 'per_query': sfr['per_query']}
    return res
""")

code(r"""# ---- baselines: the released checkpoints scored like the public systems (resumable) --------------------------------------
for m in MODELS:
    out = SCORES / m['name'] / 'baseline.json'
    if out.exists():
        print(m['name'], 'baseline done'); continue
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sc = ps.build_scorer({'name': m['name'], 'type': 'colbert_checkpoint', 'repo': m['repo'], 'revision': m.get('revision')}, device=DEVICE)
    res = score_all(sc)
    res.update({'name': m['name'], 'kind': 'baseline', 'settings': sc.settings, 'resolved_revision': ps.resolve_revision(m['repo'], m.get('revision')),
                'seconds': round(time.time() - t0, 1)})
    out.write_text(json.dumps(res)); sc.close(); del sc; gc.collect(); torch.cuda.empty_cache()
    print(f"{m['name']} baseline: SciFact nDCG@10 {res.get('scifact', {}).get('ndcg@10')} ({res['seconds']:.0f}s)", flush=True)
""")

code(r"""# ---- fine-tuning runs: model x training set x seed; each run scored immediately; failures recorded ----------------------
FAILF = RUN / 'failures.json'
FAIL = json.loads(FAILF.read_text()) if FAILF.exists() else {}
TRAIN_DATA = {'negpairs': (TRAIN, DEV), 'nevir': (NV['train'], NV['validation'])}
for m in MODELS:
    for ts in TRAIN_SETS:
        for seed in SEEDS:
            rn = f"{m['name']}/{ts}/seed{seed}"
            out = SCORES / m['name'] / ts / f'seed{seed}.json'
            if out.exists() or rn in SKIP:
                print(rn, 'done' if out.exists() else 'skipped'); continue
            out.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            try:
                sc = ps.build_scorer({'name': m['name'], 'type': 'colbert_checkpoint', 'repo': m['repo'], 'revision': m.get('revision')}, device=DEVICE)
                tr_data, dev_data = TRAIN_DATA[ts]
                summ = ft.finetune(sc, ts, tr_data, dev_data, seed=seed, lr=m['lr'], epochs=EPOCHS, batch=m.get('batch', 16), patience=PATIENCE, warmup_frac=WARMUP, verbose=True)
                res = score_all(sc)
                res.update({'name': m['name'], 'kind': f'finetuned/{ts}', 'train_set': ts, 'seed': seed, 'training': summ, 'settings': sc.settings,
                            'resolved_revision': ps.resolve_revision(m['repo'], m.get('revision')), 'seconds': round(time.time() - t0, 1), 'gpu': GPU})
                out.write_text(json.dumps(res))
                FAIL.pop(rn, None); FAILF.write_text(json.dumps(FAIL, indent=1))
                print(f"{rn}: selected epoch {summ['selected_epoch']} (dev {summ['selected_dev']:.2f}, before {summ['dev_before']:.2f}); "
                      f"SciFact {res.get('scifact', {}).get('ndcg@10')} ({res['seconds']:.0f}s)", flush=True)
                sc.close(); del sc
            except Exception as e:
                FAIL[rn] = {'error': repr(e)[:500], 'traceback': traceback.format_exc()[-4000:], 'utc': datetime.now(timezone.utc).isoformat()}
                FAILF.write_text(json.dumps(FAIL, indent=1))
                print(f'{rn} FAILED: {repr(e)[:200]}', flush=True)
            gc.collect(); torch.cuda.empty_cache()
print('failures:', list(FAIL))
""")

code(r"""# ---- metrics (the article's functions), seed means, tables, zip ------------------------------------------------------------
ROWS = []
for f in sorted(SCORES.rglob('*.json')):
    res = json.loads(f.read_text())
    row = {'name': res['name'], 'kind': res['kind'], 'train_set': res.get('train_set'), 'seed': res.get('seed'), 'training': res.get('training'),
           'seconds': res.get('seconds'), 'settings': res.get('settings'), 'resolved_revision': res.get('resolved_revision')}
    recs = {s: ps.negpairs_records(TESTS[s], res['negpairs']['score'][s]) for s in TESTS}
    row['perm'] = ps.perm_metrics(recs['perm'], BY_ID['perm'], N_BOOT)
    row['fresh'] = ps.fresh_metrics(recs['fresh'], BY_ID['fresh'], N_BOOT)
    row['nevir'] = ps.nevir_metrics(res['nevir']['score'], NV['test'], NESTED_IDS, EQUAL_IDS, N_BOOT)
    row['scifact'] = res.get('scifact', {}).get('ndcg@10')
    row['perm_predictions'] = {r['sample_id']: r['pred'] for r in recs['perm']}
    ROWS.append(row)
def val(r, path):
    cur = r
    for k in path:
        cur = cur[k]
    return cur
QUANT = {'perm.SS_bal': ('perm', 'flat', 'all.SS_bal'), 'perm.SS_bal_seen': ('perm', 'flat', 'seen.SS_bal'), 'perm.SS_bal_unseen': ('perm', 'flat', 'unseen.SS_bal'),
         'perm.flip_O_N': ('perm', 'flat', 'all.pooled.flip_O_N'), 'perm.flip_O_S': ('perm', 'flat', 'all.pooled.flip_O_S'), 'perm.joint_NS_acc': ('perm', 'flat', 'all.pooled.joint_NS_acc'),
         'fresh.SS': ('fresh', 'report', 'all', 'SS'), 'fresh.SS_seen': ('fresh', 'report', 'seen', 'SS'), 'fresh.SS_unseen': ('fresh', 'report', 'unseen', 'SS'),
         'nevir.all': ('nevir', 'all', 'paired_accuracy'), 'nevir.nested': ('nevir', 'nested', 'paired_accuracy'), 'nevir.complement': ('nevir', 'complement', 'paired_accuracy'),
         'nevir.equal': ('nevir', 'equal_bags', 'paired_accuracy'), 'nevir.ties': ('nevir', 'instances_with_tie')}
groups = defaultdict(list)
for r in ROWS:
    groups[(r['name'], r['kind'])].append(r)
SUMMARY = {}
for (name, kind), rs in sorted(groups.items()):
    g = {'n_runs': len(rs), 'seeds': [r['seed'] for r in rs]}
    for q, path in QUANT.items():
        vals = [val(r, path) for r in rs]
        g[q] = {'mean': sum(vals) / len(vals), 'values': vals, 'min': min(vals), 'max': max(vals)}
    sf = [r['scifact'] for r in rs if r['scifact'] is not None]
    g['scifact.ndcg10'] = {'mean': (sum(sf) / len(sf)) if sf else None, 'values': sf}
    if rs[0]['training']:
        g['selected_epoch'] = [r['training']['selected_epoch'] for r in rs]
        g['selected_dev'] = [r['training']['selected_dev'] for r in rs]
        g['dev_before'] = rs[0]['training']['dev_before']
        g['train_time_s'] = [r['training']['train_time_s'] for r in rs]
    SUMMARY[f'{name}|{kind}'] = g
TAB = RUN / 'tables'; TAB.mkdir(exist_ok=True)
head = ['model', 'condition', 'runs', 'Perm SS_bal', 'seen', 'unseen', 'flip_O_N', 'flip_O_S', 'Fresh SS', 'seen', 'unseen', 'NevIR all', 'nested', 'complement', 'equal', 'SciFact nDCG@10']
lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
f1 = lambda x: '' if x is None else f'{x:.1f}'
for key, g in SUMMARY.items():
    name, kind = key.split('|')
    cells = [name, kind, str(g['n_runs'])] + [f1(g[q]['mean']) for q in ('perm.SS_bal', 'perm.SS_bal_seen', 'perm.SS_bal_unseen', 'perm.flip_O_N', 'perm.flip_O_S',
                                                                        'fresh.SS', 'fresh.SS_seen', 'fresh.SS_unseen', 'nevir.all', 'nevir.nested', 'nevir.complement', 'nevir.equal')]
    cells.append(f1(g['scifact.ndcg10']['mean']))
    lines.append('| ' + ' | '.join(cells) + ' |')
(TAB / 'finetune.md').write_text('\n'.join(lines)); print('\n'.join(lines))
save_json('summary.json', {'groups': SUMMARY, 'failures': FAIL, 'gpu': GPU, 'n_boot': N_BOOT, 'data': json.loads((RUN / 'data_audit.json').read_text())})
save_json('metrics_full.json', [{k: v for k, v in r.items() if k != 'perm_predictions'} for r in ROWS])
save_json('predictions.json', [{'name': r['name'], 'kind': r['kind'], 'seed': r['seed'], 'perm': r['perm_predictions']} for r in ROWS])
n = provenance.write_file_manifest(RUN, RUN / 'FILE_MANIFEST_SHA256.csv')
RES_ZIP = ROOT / f'{RUN_NAME}_results.zip'
with zipfile.ZipFile(RES_ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(x for x in RUN.rglob('*') if x.is_file()):
        z.write(p, arcname=str(p.relative_to(RUN.parent)))
print(f'{n} files manifested | {RES_ZIP} ({RES_ZIP.stat().st_size // 1024 // 1024} MB)')
from google.colab import files
files.download(str(RES_ZIP))
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}},
      "nbformat": 4, "nbformat_minor": 5}
out = HERE / "NegPairs_finetune_li_colab.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(out, len(cells), "cells | package", PKG_SHA[:12], len(PKG) // 1024, "kB")
