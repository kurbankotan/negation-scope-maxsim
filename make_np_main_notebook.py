"""Generate the confirmatory NegPairs notebook (manifest v2.0)."""
import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# NegPairs confirmatory run (manifest v2.0, frozen 4 October 2026)

**Runtime:** Runtime > Change runtime type > **A100 GPU** (a T4 also works, about 2 to 3 times slower).
**High-RAM: not needed.** Needs `cacsa2_project.zip` (0.2.5 or later; the frozen manifest is inside it), the dataset
`MyDrive/CACSA_v2/data/negpairs_v1/` and the pilot run folder `MyDrive/CACSA_v2/runs/np_pilot_20261004T143238Z/`
(learning rate). `Runtime > Run all`; resume-safe (finished runs are skipped).

**Primary estimand:** scope sensitivity SS = (share of scenarios whose decision changes from O to N) minus (share whose
decision changes from O to S). Accuracy-based results are reported as secondary.

Grid: encoders E1, E2, E0 × aggregators A1 max, A2 mean, A3 log-sum-exp, A4 signed max with dependency scope,
A5 signed max with a learned gate, A6 signed max with a sentence-wide (scope-blind) mask, and D1 signed max with the
dataset-supplied polarity (descriptive) × 5 seeds; controls C0 (head-only), C2 (option-only) and C1 (fine-tuned
DeBERTa-v3-base multiple choice); Proposition 1 in float64; gate G and rules R1 to R5.

| Cell | Expected on an A100 |
|---|---|
| setup, data, resolver diagnostics | 2–3 min |
| caches and items | 2–5 min |
| main grid (105 runs) | about 40 s per run, about 70 min; ETA printed |
| C0 and C2 (30 runs), C1 (5 seeds) | about 5 min and about 1.5 min per seed |
| Proposition 1, analysis, zip | a few minutes |
""")

code("""# ============================== CONFIGURATION (manifest v2.0) ==============================
PROJECT_ROOT = "/content/drive/MyDrive/CACSA_v2"
PACKAGE_ZIP  = f"{PROJECT_ROOT}/cacsa2_project.zip"
MANIFEST_NAME = "CACSA_manifest_v2_0.md"   # frozen manifest shipped inside the package zip
DATA_NAME    = "negpairs_v1"
PILOT_RUN_ID = "np_pilot_20261004T143238Z"   # the learning rate is read from its lr_grid.json
SEEDS        = [13, 42, 71, 7595, 1676]
AGGREGATORS  = ["max", "mean", "lse", "signed_max", "signed_max_gate", "signed_max_sentence", "signed_max_oracle"]
ENCODERS = {
    "E0-hash":    {"kind": "hash", "dim": 384},
    "E1-minilm":  {"kind": "hf", "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "max_length": 128, "normalize": True},
    "E2-deberta": {"kind": "hf", "model": "microsoft/deberta-v3-base", "max_length": 128, "normalize": True},
}
GRID_ORDER   = ["E1-minilm", "E2-deberta", "E0-hash"]
EPOCHS, PATIENCE, BATCH = 30, 5, 16
THRESHOLDS   = {"gate_SS_seen": 20.0, "R1_max_unseen_SS": 5.0, "R2_min_gain_unseen_SS": 15.0,      # primary (manifest 8)
                "R3_min_gain_flipS": 10.0, "R3_min_items": 100}
THRESHOLDS_ACC_SECONDARY = {"gate_acc_OP": 70.0, "gate_incongruent_acc_OP": 50.0, "gate_margin_fallback": 10.0,
                "min_stratum_scenarios": 30, "R1_min_unseen_negation_gap": 20.0, "R2_min_gain_N": 15.0, "R2_max_loss_OP": -2.0,
                "R3_min_gain_S": 10.0, "R3_min_items": 100}     # draft-1 accuracy view, reported as secondary only
C1_MODEL, C1_KW, RUN_C1 = "microsoft/deberta-v3-base", dict(lr=2e-5, epochs=5, batch_size=16, max_length=128), True
N_BOOT       = 10000
# ==========================================================================================
""")

code("""%pip install -q "transformers>=4.45" "spacy>=3.8,<3.9" wordfreq scipy sentencepiece tiktoken protobuf
%pip install -q https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
from google.colab import drive
drive.mount('/content/drive')
import os, sys, json, zipfile, time, csv, random
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
ROOT = Path(PROJECT_ROOT)
import torch
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
assert _v >= (0, 2, 5), f'cacsa2_project.zip on Drive is version {cacsa2.__version__}; upload the newest cacsa2_project.zip (>= 0.2.5)'
from cacsa2 import negpairs as npx, np_analysis as npa, encoders, scope, model as cmodel, train as ctrain, prop1, provenance, controls
assert PILOT_RUN_ID, 'set PILOT_RUN_ID to the pilot run folder name'
MANIFEST_MD = next(PKG_DIR.rglob(MANIFEST_NAME), None)
assert MANIFEST_MD is not None, f'{MANIFEST_NAME} not found in the package zip'
assert 'Status: FROZEN' in MANIFEST_MD.read_text(encoding='utf-8'), 'the manifest in the package is not frozen'
assert (ROOT / 'runs' / PILOT_RUN_ID / 'lr_grid.json').exists(), f'pilot folder runs/{PILOT_RUN_ID} not found on Drive'
LR = json.loads((ROOT / 'runs' / PILOT_RUN_ID / 'lr_grid.json').read_text())['chosen_lr']
RUN_ID = datetime.now(timezone.utc).strftime('np_main_%Y%m%dT%H%M%SZ')
RUN = ROOT / 'runs' / RUN_ID; RUN.mkdir(parents=True, exist_ok=True)
(RUN / 'run_environment.json').write_text(json.dumps(provenance.run_environment({
    'run_id': RUN_ID, 'cacsa2_version': cacsa2.__version__, 'package_zip_sha256': provenance.sha256_file(PACKAGE_ZIP),
    'pilot_run_id': PILOT_RUN_ID, 'lr': LR, 'config': {'seeds': SEEDS, 'aggregators': AGGREGATORS, 'encoders': ENCODERS,
    'epochs': EPOCHS, 'patience': PATIENCE, 'batch': BATCH, 'thresholds': THRESHOLDS,
    'thresholds_accuracy_secondary': THRESHOLDS_ACC_SECONDARY, 'c1': {'model': C1_MODEL, **C1_KW}}}), indent=2))
shutil.copy(MANIFEST_MD, RUN / MANIFEST_NAME)
lock = provenance.lock_record(RUN / MANIFEST_NAME, note='recorded before any training in this run (copied from the package)'); print(lock)
(RUN / 'manifest_lock.json').write_text(json.dumps(lock, indent=2))
print('cacsa2', cacsa2.__version__, '| lr from pilot', LR, '| run', RUN)
""")

code("""# ---- dataset integrity, splits, audit -------------------------------------------------------------------------
DATA = ROOT / 'data' / DATA_NAME
man = list(csv.DictReader(open(DATA / 'FILE_MANIFEST_SHA256.csv')))
bad = [m['path'] for m in man if (DATA / m['path']).exists() and provenance.sha256_file(DATA / m['path']) != m['sha256']]
assert not bad, f'dataset files changed since they were built: {bad}'
TRAIN, DEV, TEST = (npx.load_split(DATA, s) for s in ('train', 'dev', 'test'))
ids = [{r.passage_id for r in x} for x in (TRAIN, DEV, TEST)]
assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), 'scenario overlap between splits'
assert all(r.seen_family for r in TRAIN + DEV), 'unseen-family items in train/dev'
ROWS = {r.sample_id: r for r in TRAIN + DEV + TEST}
audit = {s: {'items': len(x), 'scenarios': len({r.passage_id for r in x}), 'by_family_arm': dict(Counter(f'{r.family}/{r.arm}' for r in x)),
             'labels': dict(Counter(r.label for r in x))} for s, x in (('train', TRAIN), ('dev', DEV), ('test', TEST))}
(RUN / 'data_audit.json').write_text(json.dumps({'dataset_manifest_files': len(man), **audit}, indent=1))
print(json.dumps({s: {k: v for k, v in a.items() if k != 'by_family_arm'} for s, a in audit.items()}, indent=1))
""")

code("""# ---- frozen scope resolver on test: cue detection, false positives, scope over decisive words; R3 subset -----------
resolver = scope.ScopeResolver(scope.CueLexicon())
resolver.prefetch([r.text for r in TRAIN + DEV + TEST])
DIAG_TEST = npx.arm_diagnostics(TEST, resolver)
S_SAME = npx.s_same_ids(DIAG_TEST)
rr = {'by_seen_arm': npx.summarize_diagnostics(DIAG_TEST, by=('seen_family', 'arm')),
      'by_family_arm': npx.summarize_diagnostics(DIAG_TEST, by=('family', 'arm')),
      'S_same_items_test': len(S_SAME), 'spacy': f'{resolver.spacy_model} {resolver.spacy_version}'}
with open(RUN / 'arm_diagnostics_test.jsonl', 'w') as fh:
    for k, d in sorted(DIAG_TEST.items()):
        fh.write(json.dumps({'sample_id': k, **d}) + '\\n')
(RUN / 'S_same_ids_test.json').write_text(json.dumps(S_SAME))
(RUN / 'scope_resolver_test_report.json').write_text(json.dumps(rr, indent=1))
print(json.dumps({k: rr[k] for k in ('by_seen_arm', 'S_same_items_test')}, indent=1))
""")

code("""# ---- caches and items (scope masks for A4/A5 runs, sentence-wide masks for A6) -----------------------------------
from huggingface_hub import HfApi
STORES, ENC_META = {}, {}
texts = sorted({r.text for r in ROWS.values()} | {o for r in ROWS.values() for o in r.options})
for name, spec in ENCODERS.items():
    t0 = time.time()
    if spec['kind'] == 'hash':
        enc = encoders.HashEncoder(dim=spec['dim']); enc.name = name; ENC_META[name] = dict(spec)
    else:
        sha = None if Path(spec['model']).exists() else HfApi().model_info(spec['model']).sha
        enc = encoders.HFFrozenEncoder(spec['model'], max_length=spec['max_length'], name=name, revision=sha,
                                       dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, normalize=spec['normalize'])
        ENC_META[name] = {**spec, 'revision': sha, 'dim': enc.dim}
    store = encoders.EncodedStore(enc); store.get_or_encode(texts, batch_size=128)
    ENC_META[name]['truncated_texts'] = int(sum(store.get(t).truncated for t in texts)); STORES[name] = store
    if spec['kind'] == 'hf':
        del enc.model; torch.cuda.empty_cache()
    print(f'{name}: {len(store)} texts in {time.time()-t0:.0f}s', flush=True)
(RUN / 'encoder_meta.json').write_text(json.dumps(ENC_META, indent=1))
ITEMS = {n: {s: npx.build_items(x, st, resolver) for s, x in (('train', TRAIN), ('dev', DEV), ('test', TEST))} for n, st in STORES.items()}
ITEMS_SENT = {n: {s: npx.build_items(x, st, resolver, mask_mode='sentence') for s, x in (('train', TRAIN), ('dev', DEV), ('test', TEST))} for n, st in STORES.items()}
ITEMS_ORACLE = {n: {s: npx.build_items(x, st, None, mask_mode='oracle') for s, x in (('train', TRAIN), ('dev', DEV), ('test', TEST))} for n, st in STORES.items()}
MASK_OF = {'signed_max_sentence': ('sentence', ITEMS_SENT), 'signed_max_oracle': ('oracle', ITEMS_ORACLE)}
DIM = {n: st.encoder.dim for n, st in STORES.items()}
""")

code("""# ---- main grid -----------------------------------------------------------------------------------------------
RUNS_DIR = RUN / 'runs'
total = len(GRID_ORDER) * len(AGGREGATORS) * len(SEEDS); t_grid, done_now = time.time(), 0
for enc_name in GRID_ORDER:
    for agg in AGGREGATORS:
        kind = 'signed_max' if agg in MASK_OF else agg
        if agg in MASK_OF:
            mask_mode, its = MASK_OF[agg][0], MASK_OF[agg][1][enc_name]
        else:
            mask_mode, its = 'scope', ITEMS[enc_name]
        for seed in SEEDS:
            out_dir = RUNS_DIR / enc_name / agg / f'seed{seed}'
            if (out_dir / 'run_summary.json').exists():
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / 'arm.json').write_text(json.dumps({'directory': agg, 'module_aggregator': kind, 'polarity_mask': mask_mode,
                                                           'mask_read_by_module': kind == 'signed_max'}))
            t0 = time.time()
            cfg = ctrain.TrainConfig(aggregator=kind, lr=LR, epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH, seed=seed)
            res = ctrain.train_run(cfg, DIM[enc_name], its['train'], its['dev'], its['test'], out_dir=out_dir, label_set=npx.LABELS2, verbose=False)
            rep = npx.report(res['test']['records'], ROWS)['all']
            done_now += 1
            left = total - sum(1 for _ in RUNS_DIR.glob('E*/*/seed*/run_summary.json') if _.parts[-3] in AGGREGATORS)
            eta = (time.time() - t_grid) / done_now * left / 60
            print(f"{enc_name:11s} {agg:20s} seed {seed:5d} | test O {rep['acc_O']:.1f} P {rep['acc_P']:.1f} S {rep['acc_S']:.1f} N {rep['acc_N']:.1f} "
                  f"| ep {res['selected_epoch']} {time.time()-t0:.0f}s | {left} left ~{eta:.0f} min", flush=True)
""")

code("""# ---- controls C0 (head-only, pooled states) and C2 (option-only, context-blind) ------------------------------------
for enc_name in GRID_ORDER:
    for agg, factory in (('head_only', controls.head_only_factory), ('option_only', controls.option_only_factory)):
        for seed in SEEDS:
            out_dir = RUNS_DIR / enc_name / agg / f'seed{seed}'
            if (out_dir / 'run_summary.json').exists():
                continue
            cfg = ctrain.TrainConfig(aggregator=agg, lr=LR, epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH, seed=seed)
            res = ctrain.train_run(cfg, DIM[enc_name], ITEMS[enc_name]['train'], ITEMS[enc_name]['dev'], ITEMS[enc_name]['test'],
                                   out_dir=out_dir, label_set=npx.LABELS2, verbose=False, model_factory=factory)
            rep = npx.report(res['test']['records'], ROWS)['all']
            print(f"{enc_name:11s} {agg:11s} seed {seed}: test acc_OP {rep['acc_OP']:.1f} acc_N {rep['acc_N']:.1f}", flush=True)
""")

code("""# ---- control C1: fine-tuned multiple-choice cross-encoder --------------------------------------------------------
if RUN_C1:
    from huggingface_hub import HfApi
    sha = None if Path(C1_MODEL).exists() else HfApi().model_info(C1_MODEL).sha
    for seed in SEEDS:
        out_dir = RUNS_DIR / 'C1-mcft' / (C1_MODEL.split('/')[-1] + '-mc-ft') / f'seed{seed}'
        if (out_dir / 'run_summary.json').exists():
            continue
        s = npx.finetune_mc_run(C1_MODEL, TRAIN, DEV, TEST, seed=seed, out_dir=out_dir, revision=sha, **C1_KW)
        print(f"C1 seed {seed}: dev {s['dev']['accuracy']:.1f} test {s['test']['accuracy']:.1f} epoch {s['selected_epoch']}", flush=True)
""")

code("""# ---- R5: Proposition 1 in float64 on test insertion pairs (E0; trained A1 and A3, every seed) -----------------------
pairs = npx.insertion_pairs(TEST); s0 = STORES['E0-hash']
print(f'test insertion pairs: {len(pairs)} by arm {dict(Counter(x.arm for _, x in pairs))}')
r5 = {}
for agg in ('max', 'lse'):
    for seed in SEEDS:
        m = prop1.load_module(str(RUNS_DIR / 'E0-hash' / agg / f'seed{seed}' / 'module_state.pt'), DIM['E0-hash'], agg)
        sc = [(prop1.option_scores(m, s0.get(x.text).states, [s0.get(o).states for o in o_.options]),
               prop1.option_scores(m, s0.get(o_.text).states, [s0.get(o).states for o in o_.options])) for o_, x in pairs]
        r5[f'{agg}/seed{seed}'] = prop1.violations(sc, tol=1e-9); print(agg, seed, r5[f'{agg}/seed{seed}'], flush=True)
r5['met'] = all(v['violations'] == 0 for k, v in r5.items() if k != 'met')
(RUN / 'R5_proposition1_float64.json').write_text(json.dumps(r5, indent=1)); print('R5 met:', r5['met'])
""")

code("""# ---- analysis: primary rules (scope sensitivity), secondary accuracy view, tables -------------------------------------
runs = npa.load_runs(RUNS_DIR, split='test')
S_SAME_SCEN = sorted({ROWS[i].passage_id for i in S_SAME})
STRATA = npx.option_prior_strata([v for (e, g, sd), v in runs.items() if g == 'option_only'], ROWS)
STRATA_IDS = npx.strata_item_ids(STRATA, TEST)
(RUN / 'option_prior_strata_test.json').write_text(json.dumps(STRATA, indent=0))
rows = npa.per_run(runs, ROWS, extra_subsets={'S_same': S_SAME, **STRATA_IDS}, s_same_scenarios=S_SAME_SCEN)
sm = npa.seed_means(rows)
(RUN / 'per_run.json').write_text(json.dumps(rows, indent=1)); (RUN / 'seed_means.json').write_text(json.dumps(sm, indent=1))
cols_flip = ['encoder', 'aggregator', 'n_seeds', 'seen.SS', 'unseen.SS', 'seen.flip_O_N', 'seen.flip_O_S', 'unseen.flip_O_N',
             'unseen.flip_O_S', 'S_same_scen.flip_O_S', 'samecue_not.flip_O_N', 'samecue_not.flip_O_S', 'all.flip_O_P']
cols_acc = ['encoder', 'aggregator', 'n_seeds', 'all.acc', 'all.acc_O', 'all.acc_P', 'all.acc_S', 'all.acc_N', 'unseen.acc_N',
            'seen.acc_N', 'all.acc_N_given_O', 'all.consistency', 'prior_incongruent.acc_OP', 'prior_incongruent.acc_N']
(RUN / 'table_seed_means_flip.md').write_text(npa.markdown(sm, cols_flip))
(RUN / 'table_seed_means_accuracy.md').write_text(npa.markdown(sm, cols_acc))
rules = npa.evaluate_rules_flip(rows, runs, ROWS, GRID_ORDER, THRESHOLDS, n_boot=N_BOOT, s_same_scenarios=S_SAME_SCEN)
rules['R5'] = {'met': r5['met']}
(RUN / 'rules.json').write_text(json.dumps(rules, indent=1, default=str))
rules_acc = npa.evaluate_rules_accuracy(rows, runs, ROWS, GRID_ORDER, THRESHOLDS_ACC_SECONDARY, n_boot=N_BOOT, s_same=S_SAME,
                                        strata_ids=STRATA_IDS)
(RUN / 'rules_accuracy_secondary.json').write_text(json.dumps(rules_acc, indent=1, default=str))
mc = []
for enc in GRID_ORDER:
    mc += npa.mcnemar_by_seed(runs, ROWS, enc, 'signed_max', 'max', ('N',), 'all')
    mc += npa.mcnemar_by_seed(runs, ROWS, enc, 'signed_max', 'max', ('N',), 'unseen')
    mc += npa.mcnemar_by_seed(runs, ROWS, enc, 'signed_max', 'signed_max_sentence', ('S',), 'all', ids=S_SAME, tag='S_same')
(RUN / 'mcnemar_accuracy_secondary.json').write_text(json.dumps(mc, indent=1))
print('scope sensitivity (seed means, percentage points)'); print((RUN / 'table_seed_means_flip.md').read_text())
print('accuracy (secondary)'); print((RUN / 'table_seed_means_accuracy.md').read_text())
summary = {'G': {e: v['passed'] for e, v in rules['G'].items()}, 'R1': {e: v['met'] for e, v in rules['R1'].items()},
           'R2': {e: v['met'] for e, v in rules['R2'].items()}, 'R3': {e: v['met'] for e, v in rules['R3'].items()},
           'R5': rules['R5']['met']}
print('PRIMARY RULES:', json.dumps(summary, indent=1))
""")

code("""# ---- file manifest, zip, download -----------------------------------------------------------------------------------
n = provenance.write_file_manifest(RUN, RUN / 'FILE_MANIFEST_SHA256.csv')
ZIP = Path('/content') / f'{RUN_ID}.zip'
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(x for x in RUN.rglob('*') if x.is_file() and x.suffix != '.pt'):
        z.write(p, arcname=str(p.relative_to(RUN.parent)))
print('manifested', n, 'files ->', ZIP, ZIP.stat().st_size // 1024, 'kB')
from google.colab import files
files.download(str(ZIP))
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "NegPairs_main_colab.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(out, len(cells), "cells")
