"""Generate NegPairs_v3_followup_colab.ipynb: float64 NegPairs-Perm predictions for the static encoders E0 and E3
(manifest v3.0, section 10: every reported E0/E3 metric on NegPairs-Perm comes from float64 scores). The main run
computed the Perm tables of E0 and E3 from the GPU (bfloat16) predictions like every other encoder; this notebook
re-scores every trained E0/E3 condition and M0 on NegPairs-Perm in float64 on the CPU from the archived module states,
writes `perm_predictions_f64.jsonl` beside the GPU predictions, recomputes the Perm metrics of E0 and E3 from them and
lists every difference. No model is trained; no rule changes (R6 was already evaluated in float64)."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
MANIFEST = HERE / "CACSA_manifest_v3_0.md"
MAN_SHA = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# NegPairs v3.0 follow-up: float64 Perm predictions for E0 and E3

**Runtime:** CPU is enough (no GPU needed); about 10 to 15 minutes. **High-RAM: not needed.**
Needs the finished run folder `MyDrive/CACSA_v3/runs/np_v3_main/` (module states on Drive), `cacsa2_project.zip`
(0.3.2 or later) and `data/negpairs_v3/`. `Runtime > Run all`.

Manifest v3.0, section 10, asks that every reported metric of the static encoders E0 and E3 on NegPairs-Perm come from
float64 scores. The main notebook evaluated the exact rule R6 in float64 but built the Perm tables of E0 and E3 from
the GPU (bfloat16) predictions like every other encoder. This notebook re-scores every E0 and E3 condition on
NegPairs-Perm in float64 from the archived module states and recomputes their Perm metrics. Nothing is trained.

At the end `np_v3_main_followup.zip` is downloaded. Send it back.
""")

code(f"""# ============================== CONFIGURATION ==============================
PROJECT_ROOT  = "/content/drive/MyDrive/CACSA_v3"
PACKAGE_ZIP   = f"{{PROJECT_ROOT}}/cacsa2_project.zip"
MANIFEST_NAME, MANIFEST_SHA256 = "CACSA_manifest_v3_0.md", "{MAN_SHA}"
DATA_V3_DIR   = f"{{PROJECT_ROOT}}/data/negpairs_v3"
RUN_NAME      = "np_v3_main"
SEEDS         = [13, 42, 71, 7595, 1676, 101, 202, 303, 404, 505]
CONDITIONS    = ["max", "mean", "lse", "signed_max", "signed_max_gate", "signed_max_sentence", "signed_max_oracle", "m1", "head_only", "option_only"]
STATIC        = ["E0-hash", "E3-glove"]
# ===========================================================================
""")

code(r"""%pip install -q "transformers>=4.45" "spacy>=3.8,<3.9" wordfreq==3.1.1 scipy
%pip install -q https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
from google.colab import drive
drive.mount('/content/drive')
import os, sys, json, zipfile, time, csv, shutil, importlib, hashlib
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import torch
ROOT = Path(PROJECT_ROOT)
ZIP_SHA = hashlib.sha256(open(PACKAGE_ZIP, 'rb').read()).hexdigest()
PKG_DIR = Path('/content') / f'cacsa2_pkg_{ZIP_SHA[:12]}'
shutil.rmtree(PKG_DIR, ignore_errors=True)
with zipfile.ZipFile(PACKAGE_ZIP) as z:
    z.extractall(PKG_DIR)
for _m in [m for m in list(sys.modules) if m == 'cacsa2' or m.startswith('cacsa2.')]:
    del sys.modules[_m]
sys.path = [p for p in sys.path if 'cacsa2_pkg_' not in p]
PKG_ROOT = next(PKG_DIR.rglob('cacsa2/__init__.py')).parent.parent
sys.path.insert(0, str(PKG_ROOT)); importlib.invalidate_caches()
import cacsa2
assert tuple(int(x) for x in cacsa2.__version__.split('.')) >= (0, 3, 2), f'package {cacsa2.__version__}; upload cacsa2_project.zip 0.3.2 or later'
from cacsa2 import negpairs as npx, np_analysis as npa, np_analysis_v3 as v3, encoders, scope, model as cmodel, train as ctrain, provenance, packs, worker, exact
RUN = ROOT / 'runs' / RUN_NAME
assert (RUN / 'rules.json').exists(), 'the main run has not finished'
assert json.loads((RUN / 'manifest_lock.json').read_text())['sha256'] == MANIFEST_SHA256
RUNS_DIR = RUN / 'runs'
V1 = PKG_ROOT / 'data' / 'negpairs_v1'
D3 = Path(DATA_V3_DIR)
for d in (V1, D3):
    for row in csv.DictReader(open(d / 'FILE_MANIFEST_SHA256.csv', encoding='utf-8')):
        assert provenance.sha256_file(d / row['path']) == row['sha256'], f"file changed: {row['path']}"
PERM = npx.load_split(D3, 'perm')
PERM_BY_ID = {r.sample_id: r for r in PERM}
print('cacsa2', cacsa2.__version__, '| run', RUN, '| Perm rows', len(PERM))
""")

code(r"""# ---- static encoders, masks, items ------------------------------------------------------------------------------------
blob = torch.load(ROOT / 'cache' / 'glove.6B.300d.pt', weights_only=False)
ENC = {'E0-hash': encoders.HashEncoder(dim=384), 'E3-glove': encoders.GloveEncoder(dim=300, vocab=blob['vocab'], vectors=blob['vectors'])}
for n, e in ENC.items():
    e.name = n
resolver = scope.ScopeResolver(scope.CueLexicon())
texts = sorted({r.text for r in PERM} | {o for r in PERM for o in r.options})
ITEMS = {}
for name, e in ENC.items():
    st = encoders.EncodedStore(e); st.get_or_encode(texts)
    ITEMS[name] = {'scope': npx.build_items(PERM, st, resolver, mask_mode='scope'), 'sentence': npx.build_items(PERM, st, resolver, mask_mode='sentence'),
                   'oracle': npx.build_items(PERM, st, None, mask_mode='oracle'), 'zero': npx.build_items(PERM, st, None)}
    assert all(int(it.polarity.sum()) == 0 for it in ITEMS[name]['zero'])
DIM = {n: e.dim for n, e in ENC.items()}
print({n: {k: len(v) for k, v in d.items()} for n, d in ITEMS.items()})
""")

code(r"""# ---- float64 predictions for every E0/E3 condition on NegPairs-Perm --------------------------------------------------
def records_f64(model, items):
    sc = exact.scores_f64(model, items)
    out = []
    for it, s in zip(items, sc):
        r = it.row
        z = torch.softmax(torch.tensor(s, dtype=torch.float64), dim=0).tolist()
        out.append({'sample_id': r.sample_id, 'passage_id': r.passage_id, 'question_id': r.question_id, 'edit_id': r.edit_id, 'label': r.label,
                    'pred': npx.LABELS2[0 if s[0] >= s[1] else 1], 'probs': [round(x, 6) for x in z], 'scores': [round(x, 9) for x in s],
                    'cue': r.cue, 'unseen_cue': not r.seen_family, 'dtype': 'float64'})
    return out
t0 = time.time(); n_done = 0
for name in STATIC:
    for cond in CONDITIONS + ['m0']:
        mode = worker.MASK_OF.get(cond, 'scope')
        kind = worker.MODULE_OF.get(cond, cond)
        items = ITEMS[name]['zero' if kind in ('max', 'mean', 'lse', 'signed_max_gate', 'm1', 'head_only', 'option_only', 'm0') else mode]
        for seed in ([0] if cond == 'm0' else SEEDS):
            out = RUNS_DIR / name / cond / f'seed{seed}'
            if (out / 'perm_predictions_f64.jsonl').exists():
                continue
            assert (out / 'done.json').exists(), f'missing run {out}'
            model = cmodel.ZeroShotMaxSim(DIM[name]) if cond == 'm0' else exact.load_condition(out / 'module_state.pt', DIM[name], kind)
            ctrain._write_records(out / 'perm_predictions_f64.jsonl', records_f64(model, items))
            n_done += 1
    print(f'{name}: done ({(time.time() - t0) / 60:.1f} min)', flush=True)
print('float64 prediction files written now:', n_done)
""")

code(r"""# ---- Perm metrics of E0 and E3 from float64 predictions; differences from the GPU-based values -------------------------
def load(split_file):
    out = {}
    for d in RUNS_DIR.glob('*/*/seed*'):
        f = d / split_file
        if (d / 'done.json').exists() and f.exists() and d.parent.parent.name in STATIC:
            out[(d.parent.parent.name, d.parent.name, int(d.name[4:]))] = [json.loads(l) for l in open(f) if l.strip()]
    return out
RUNS64, RUNSGPU = load('perm_predictions_f64.jsonl'), load('perm_predictions.jsonl')
assert set(RUNS64) == set(RUNSGPU), 'prediction sets differ'
EQ = json.loads((RUN / 'perm_token_bag_check.json').read_text())
rows64 = v3.per_run_perm(RUNS64, PERM_BY_ID)                       # E0/E3 tokenizers keep every scenario (n_equal = 274)
rowsgpu = v3.per_run_perm(RUNSGPU, PERM_BY_ID)
sm64, smgpu = npa.seed_means(rows64), npa.seed_means(rowsgpu)
item_dis = {}
for key, recs in RUNS64.items():
    g = {r['sample_id']: r['pred'] for r in RUNSGPU[key]}
    item_dis['/'.join(map(str, key))] = sum(1 for r in recs if g[r['sample_id']] != r['pred'])
diffs = []
for a, b in zip(sm64, smgpu):
    assert (a['encoder'], a['aggregator']) == (b['encoder'], b['aggregator'])
    for k in ('seen.SS_bal', 'seen.pooled.SS', 'seen.first.SS', 'seen.second.SS', 'seen.pooled.joint_NS_acc', 'unseen.SS_bal', 'unseen.pooled.joint_NS_acc', 'all.pooled.flip_O_P'):
        if a.get(k) is not None and b.get(k) is not None and abs(a[k] - b[k]) > 1e-9:
            diffs.append({'encoder': a['encoder'], 'aggregator': a['aggregator'], 'metric': k, 'float64': a[k], 'gpu': b[k]})
report = {'computed_utc': datetime.now(timezone.utc).isoformat(), 'cacsa2_version': cacsa2.__version__, 'runs_rescored': len(RUNS64),
          'item_decision_disagreements_gpu_vs_f64': {'total': sum(item_dis.values()), 'items_per_run': len(PERM), 'by_run_nonzero': {k: v for k, v in item_dis.items() if v}},
          'seed_mean_differences': diffs, 'n_differences': len(diffs)}
(RUN / 'per_run_perm_f64_static.json').write_text(json.dumps(rows64, indent=1))
(RUN / 'seed_means_perm_f64_static.json').write_text(json.dumps(sm64, indent=1))
(RUN / 'table_perm_f64_static.md').write_text(npa.markdown(sm64, v3.PERM_TABLE_COLS))
(RUN / 'perm_f64_vs_gpu.json').write_text(json.dumps(report, indent=1))
print((RUN / 'table_perm_f64_static.md').read_text())
print(f"item-level GPU vs float64 decision disagreements: {report['item_decision_disagreements_gpu_vs_f64']['total']} | seed-mean metric differences: {len(diffs)}")
for d in diffs[:20]:
    print('  ', d)
""")

code(r"""# ---- file manifest update and zip -------------------------------------------------------------------------------------------
n = provenance.write_file_manifest(RUN, RUN / 'FILE_MANIFEST_SHA256.csv', exclude=('logs/',))
ZIP = ROOT / f'{RUN_NAME}_followup.zip'
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for name in ('per_run_perm_f64_static.json', 'seed_means_perm_f64_static.json', 'table_perm_f64_static.md', 'perm_f64_vs_gpu.json', 'FILE_MANIFEST_SHA256.csv'):
        z.write(RUN / name, arcname=f'{RUN_NAME}/{name}')
    for p in sorted(RUNS_DIR.glob('*/*/seed*/perm_predictions_f64.jsonl')):
        z.write(p, arcname=str(p.relative_to(RUN.parent)))
print(f'{n} files manifested | {ZIP} {ZIP.stat().st_size // 1024} kB')
from google.colab import files
files.download(str(ZIP))
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
out = HERE / "NegPairs_v3_followup_colab.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(out, len(cells), "cells")
