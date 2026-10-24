"""Generate the Colab notebook for the main CACSA v2 experiment grid (manifest v0.1 → frozen version)."""
import json
from pathlib import Path

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})


md("""# CACSA v2 — Main experiment grid (CondaQA, frozen encoders, five aggregators)

**Runtime:** Colab A100 (high RAM). `Runtime > Run all`. The notebook is resume-safe: every finished run leaves a
`run_summary.json`, and re-running skips those runs.

**Before running**
1. Upload `cacsa2_project.zip` to Google Drive at `MyDrive/CACSA_v2/cacsa2_project.zip` (configurable below).
2. Nothing else: CondaQA is cloned from GitHub at a pinned commit; encoders come from the Hugging Face Hub with their
   resolved commit recorded.

**Package version:** requires `cacsa2` ≥ 0.1.1 (0.1.0 could hang in the scope-resolver cell).

**What it does** (manifest sections 3–9): data audit and hashes → scope resolver and its cue recall → token-state caches
for E0/E1/E2 → learning-rate mini-grid (A1, E1, seed 13, dev only) → grid encoders × aggregators × seeds → controls C0
(head-only) and C1 (full fine-tuning) → Proposition-1 cue-deletion test on E0 → tables, rules R1–R5, intervals, McNemar →
`FILE_MANIFEST_SHA256.csv` and a zip of the run.

The manifest hash is recorded in the run directory before any training starts; if the manifest changes afterwards, the
run is labelled as executed under a different manifest.

**Expected durations** (so that a stall is visible). "Measured" = measured on 3 October 2026 when this exact notebook was
executed on all CondaQA rows on a 2-core CPU machine with encoder E0; the A100 figures are estimates.

| Cell | Expected |
|---|---|
| install + Drive | 1–2 min |
| CondaQA clone + hash check | < 1 min |
| scope resolver, cue recall | < 1 min (measured 8 s) |
| token caches | E0 < 1 min (measured 11 s); E1 and E2 a few minutes each, with a progress counter |
| items (scope analysis of all 13,967 contexts, then masks) | about 2–4 min (measured 116 s for the analysis), progress every 1,000 texts |
| learning-rate grid | 5–15 min, one line per epoch |
| main grid (75 runs) | about 2–4 min per run; every run prints a line with the number of runs left and an ETA |
| head-only control (15 runs) | 15–30 min |
| full fine-tuning (5 seeds) | about 10–15 min per seed |
| Proposition-1 test, analysis, zip | a few minutes |

If a cell runs far beyond its expected time **without printing anything new**, stop it and send the output.
""")

code("""# ============================== CONFIGURATION ==============================
PROJECT_ROOT   = "/content/drive/MyDrive/CACSA_v2"
PACKAGE_ZIP    = f"{PROJECT_ROOT}/cacsa2_project.zip"
MANIFEST_MD    = f"{PROJECT_ROOT}/CACSA_manifest_v1_1.md"      # the frozen manifest; its SHA-256 is recorded
CACHE_DIR      = "/content/cacsa_caches"      # token-state caches on the runtime's local disk (not Drive: size/quota)
RUN_ID         = None            # None -> new timestamped run; set to an existing id to resume
CONDAQA_REPO   = "https://github.com/AbhilashaRavichander/CondaQA.git"
CONDAQA_COMMIT = "bd4857f1f2819937f1505925ad8c809544e8e5ef"   # pinned (manifest v1.1, section 3.1)
CONDAQA_SHA256 = {                                               # expected file hashes at that commit
    "condaqa_train.json": "f5f8a9d1b64a6c3b111b00d2e125f1d7f4da8b50acfdbc788d2901e2e43d79ec",
    "condaqa_dev.json":   "b9ec9ab7453fbdc61fbed988119c03f507341ac495f1f5f95b13b27f59e639a2",
    "condaqa_test.json":  "ffd7f4ad85689b2317283430a226cd759c3ccc40070e6a32652e7ae84ae90e8e",
}

SEEDS          = [13, 42, 71, 7595, 1676]
AGGREGATORS    = ["max", "mean", "lse", "signed_max", "signed_max_gate"]
ENCODERS = {
    "E0-hash":    {"kind": "hash", "dim": 384},
    "E1-minilm":  {"kind": "hf", "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "max_length": 512},
    "E2-deberta": {"kind": "hf", "model": "microsoft/deberta-v3-base", "max_length": 512},
}
CONTEXTUAL     = ["E1-minilm", "E2-deberta"]
GRID_ORDER     = CONTEXTUAL + ["E0-hash"]       # contextual encoders first, so the main results exist earliest
LR_GRID        = [3e-4, 1e-3, 3e-3]   # dev-only mini-grid on A1 / E1 / seed 13, then fixed for every run
LR_GRID_ENCODER = "E1-minilm"
EPOCHS, PATIENCE, BATCH = 20, 5, 16
FULLFT_MODEL   = "microsoft/deberta-v3-base"
FULLFT_SEEDS   = SEEDS
RUN_FULLFT     = True
N_BOOT         = 10000
# ===========================================================================
""")

code("""%pip install -q "transformers>=4.45" "spacy>=3.8,<3.9" wordfreq scipy sentencepiece tiktoken protobuf
%pip install -q https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
from google.colab import drive
drive.mount('/content/drive')
import os, sys, json, zipfile, time, subprocess, hashlib, random
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')   # deterministic cuBLAS, set before torch touches CUDA
from pathlib import Path
from datetime import datetime, timezone
ROOT = Path(PROJECT_ROOT); ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(PACKAGE_ZIP) as z: z.extractall('/content/cacsa2_unzipped')
# the zip may or may not carry a top-level folder: locate the directory that contains cacsa2/__init__.py
pkg_init = next(Path('/content/cacsa2_unzipped').rglob('cacsa2/__init__.py'))
PKG_ROOT = str(pkg_init.parent.parent)
sys.path.insert(0, PKG_ROOT)
print('package root:', PKG_ROOT)
import torch
import cacsa2
from cacsa2 import condaqa, encoders, scope, model as cmodel, train as ctrain, metrics, stats, analysis, provenance, controls
PACKAGE_SHA256 = provenance.sha256_file(PACKAGE_ZIP)
print('cacsa2', cacsa2.__version__, '| zip sha256', PACKAGE_SHA256[:16], '| torch', torch.__version__, '| cuda', torch.cuda.is_available())
assert cacsa2.__version__ >= '0.1.1', 'old package: upload the new cacsa2_project.zip (v0.1.1 fixes the scope-resolver hang)'

""")

code("""# ---- immutable run directory + environment + manifest lock ------------------------------------
RUN_ID = RUN_ID or datetime.now(timezone.utc).strftime('main_%Y%m%dT%H%M%SZ')
RUN = ROOT / 'runs' / RUN_ID
RUN.mkdir(parents=True, exist_ok=True)
env = provenance.run_environment({'run_id': RUN_ID, 'cacsa2_version': cacsa2.__version__, 'package_zip_sha256': PACKAGE_SHA256, 'config': {
    'seeds': SEEDS, 'aggregators': AGGREGATORS, 'encoders': ENCODERS, 'lr_grid': LR_GRID,
    'epochs': EPOCHS, 'patience': PATIENCE, 'batch': BATCH, 'fullft_model': FULLFT_MODEL}})
(RUN / 'run_environment.json').write_text(json.dumps(env, indent=2))
if Path(MANIFEST_MD).exists():
    lock = provenance.lock_record(MANIFEST_MD, note='recorded before any training in this run')
    (RUN / 'manifest_lock.json').write_text(json.dumps(lock, indent=2)); print(lock)
else:
    print('WARNING: manifest file not found; the run will be labelled as executed without a locked manifest')
print(RUN)
""")

code("""# ---- CondaQA at a pinned commit; file hashes; data summary ------------------------------------
DATA_DIR = Path('/content/CondaQA')
if not DATA_DIR.exists():
    subprocess.run(['git', 'clone', '--quiet', CONDAQA_REPO, str(DATA_DIR)], check=True)
    if CONDAQA_COMMIT:
        subprocess.run(['git', '-C', str(DATA_DIR), 'checkout', '--quiet', CONDAQA_COMMIT], check=True)
commit = subprocess.run(['git', '-C', str(DATA_DIR), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
hashes = condaqa.file_hashes(DATA_DIR / 'data')
mismatch = {k: v for k, v in hashes.items() if CONDAQA_SHA256.get(k) not in (None, v)}
assert not mismatch, f'CondaQA files differ from the manifest: {mismatch}'
data = condaqa.load_condaqa(DATA_DIR / 'data')
summ = condaqa.summary(data)
(RUN / 'data_provenance.json').write_text(json.dumps({'repo': CONDAQA_REPO, 'commit': commit, 'file_sha256': hashes, 'summary': summ}, indent=2))
print(commit); print(json.dumps(summ, indent=1))
TRAIN, DEV, TEST = data['train'], data['dev'], data['test']
""")

code("""# ---- scope resolver (core + affixal + CondaQA *train* cues) and its intrinsic cue recall -----------
train_cues = set(condaqa.cue_inventory(TRAIN))
lexicon = scope.CueLexicon(extra_cues=train_cues)
resolver = scope.ScopeResolver(lexicon)
t0 = time.time()
rec = {split: scope.cue_recall(resolver, [(r.passage, r.cue) for r in rows if r.edit_id == 0], progress=True)
       for split, rows in (('dev', DEV), ('test', TEST))}
print(f'cue recall computed in {time.time()-t0:.0f}s (expected: well under a minute)')
rec['spacy_model'] = resolver.spacy_model; rec['spacy_version'] = resolver.spacy_version
rec['lexicon'] = {'core_syntactic': sorted(lexicon.core_syntactic), 'core_lexical': sorted(lexicon.core_lexical),
                  'n_train_cues': len(train_cues), 'affixal': lexicon.affixal, 'base_zipf_min': lexicon.base_zipf_min, 'combine': resolver.combine}
(RUN / 'scope_resolver_report.json').write_text(json.dumps(rec, indent=2))
print({k: {kk: vv for kk, vv in v.items() if kk != 'missed_cues'} for k, v in rec.items() if k in ('dev', 'test')})
""")

code("""# ---- token-state caches per encoder (resume-safe) -------------------------------------------------
from huggingface_hub import HfApi
CACHE = Path(CACHE_DIR); CACHE.mkdir(parents=True, exist_ok=True)
all_rows = [r for s in (TRAIN, DEV, TEST) for r in s if r.eligible]
texts = list(dict.fromkeys([condaqa.context_text(r.question, r.passage) for r in all_rows] + condaqa.render_candidates()))
print(len(texts), 'unique texts to encode')
STORES, ENC_META = {}, {}
for name, spec in ENCODERS.items():
    if spec['kind'] == 'hash':
        enc = encoders.HashEncoder(dim=spec['dim']); enc.name = name
        ENC_META[name] = {'kind': 'hash', 'dim': spec['dim']}
    else:
        sha = HfApi().model_info(spec['model']).sha
        enc = encoders.HFFrozenEncoder(spec['model'], max_length=spec['max_length'], name=name, revision=sha,
                                       dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        ENC_META[name] = {'kind': 'hf', 'model': spec['model'], 'revision': sha, 'max_length': spec['max_length'], 'dim': enc.dim}
    store = encoders.EncodedStore(enc, path=CACHE / f'{name}.pt')
    t0 = time.time()
    print(f'{name}: encoding...', flush=True)
    store.get_or_encode(texts, batch_size=64, progress=True)
    store.save()
    truncated = sum(store.get(t).truncated for t in texts)
    ENC_META[name]['truncated_texts'] = int(truncated); ENC_META[name]['cache_sha256'] = provenance.sha256_file(CACHE / f'{name}.pt')
    STORES[name] = store
    print(f'{name}: {len(store)} texts cached in {time.time()-t0:.0f}s; truncated {truncated}')
    if spec['kind'] == 'hf':
        del enc.model; torch.cuda.empty_cache()
(RUN / 'encoder_meta.json').write_text(json.dumps(ENC_META, indent=2))
""")

code("""# ---- items (encoded rows + polarity masks) per encoder -------------------------------------------
t0 = time.time()
resolver.prefetch([condaqa.context_text(r.question, r.passage) for r in all_rows], progress=True)   # ~2-4 min, once
print(f'scope analysis of all contexts: {time.time()-t0:.0f}s')
ITEMS = {}
pol_cache = {}
for name, store in STORES.items():
    ITEMS[name] = {
        'train': ctrain.build_items(TRAIN, store, resolver, train_rows_for_cues=TRAIN, polarity_cache=pol_cache),
        'dev':   ctrain.build_items(DEV,   store, resolver, train_rows_for_cues=TRAIN, polarity_cache=pol_cache),
        'test':  ctrain.build_items(TEST,  store, resolver, train_rows_for_cues=TRAIN, polarity_cache=pol_cache),
    }
    n_tok = sum(it.ctx.shape[0] for it in ITEMS[name]['train']); n_neg = sum(int(it.polarity.sum()) for it in ITEMS[name]['train'])
    print(f"{name}: items {[len(v) for v in ITEMS[name].values()]}; fraction of train context tokens inside a negation scope {n_neg/n_tok:.3f}")
DIM = {name: STORES[name].encoder.dim for name in STORES}
""")

code("""# ---- learning-rate mini-grid: A1 (max) / E1 / seed 13, dev accuracy only; then FIXED for every run ----
grid_dir = RUN / 'lr_grid'
best_lr, grid = None, []
for lr in LR_GRID:
    cfg = ctrain.TrainConfig(aggregator='max', lr=lr, epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH, seed=13)
    print(f'lr {lr:g}:', flush=True)
    res = ctrain.train_run(cfg, DIM[LR_GRID_ENCODER], ITEMS[LR_GRID_ENCODER]['train'], ITEMS[LR_GRID_ENCODER]['dev'], None,
                           out_dir=grid_dir / f'lr{lr:g}', verbose=True)
    grid.append({'lr': lr, 'dev_acc': res['dev']['accuracy'], 'selected_epoch': res['selected_epoch']})
    print(grid[-1])
best_lr = max(grid, key=lambda g: (g['dev_acc'], -g['lr']))['lr']
(RUN / 'lr_grid.json').write_text(json.dumps({'grid': grid, 'chosen_lr': best_lr, 'rule': 'highest dev accuracy; tie -> smaller lr'}, indent=2))
print('chosen lr', best_lr)
""")

code("""# ---- main grid: encoders x aggregators x seeds (resume-safe) ---------------------------------------
RUNS_DIR = RUN / 'runs'
grid_start, done_now = time.time(), 0
total_runs = len(GRID_ORDER) * len(AGGREGATORS) * len(SEEDS)
for enc_name in GRID_ORDER:
    for agg in AGGREGATORS:
        for seed in SEEDS:
            out_dir = RUNS_DIR / enc_name / agg / f'seed{seed}'
            if (out_dir / 'run_summary.json').exists():
                continue
            cfg = ctrain.TrainConfig(aggregator=agg, lr=best_lr, epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH, seed=seed)
            print(f"start {enc_name} {agg} seed {seed} ...", flush=True)
            t0 = time.time()
            res = ctrain.train_run(cfg, DIM[enc_name], ITEMS[enc_name]['train'], ITEMS[enc_name]['dev'], ITEMS[enc_name]['test'],
                                   out_dir=out_dir, verbose=False)
            done_now += 1
            left = total_runs - sum(1 for _ in RUNS_DIR.glob('*/*/seed*/run_summary.json') if _.parts[-4] in GRID_ORDER and _.parts[-3] in AGGREGATORS)
            eta_min = (time.time() - grid_start) / done_now * left / 60
            print(f"{enc_name:11s} {agg:16s} seed {seed:5d}  dev {res['dev']['accuracy']:.2f}  test {res['test']['accuracy']:.2f}  "
                  f"epoch {res['selected_epoch']}  {time.time()-t0:.0f}s  | {left} runs left, ~{eta_min:.0f} min", flush=True)
""")

code("""# ---- control C0: head-only scorer on frozen pooled states ---------------------------------------
for enc_name in GRID_ORDER:
    for seed in SEEDS:
        out_dir = RUNS_DIR / enc_name / 'head_only' / f'seed{seed}'
        if (out_dir / 'run_summary.json').exists():
            continue
        cfg = ctrain.TrainConfig(aggregator='head_only', lr=best_lr, epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH, seed=seed)
        res = ctrain.train_run(cfg, DIM[enc_name], ITEMS[enc_name]['train'], ITEMS[enc_name]['dev'], ITEMS[enc_name]['test'],
                               out_dir=out_dir, verbose=False, model_factory=controls.head_only_factory)
        print(f"{enc_name:11s} head_only        seed {seed:5d}  dev {res['dev']['accuracy']:.2f}  test {res['test']['accuracy']:.2f}")
""")

code("""# ---- control C1: full fine-tuning reference (DeBERTa-v3-base, 3-class) ---------------------------
if RUN_FULLFT:
    from cacsa2 import finetune
    from huggingface_hub import HfApi
    sha = HfApi().model_info(FULLFT_MODEL).sha
    for seed in FULLFT_SEEDS:
        out_dir = RUNS_DIR / 'C1-fullft' / (FULLFT_MODEL.split('/')[-1] + '-ft') / f'seed{seed}'
        if (out_dir / 'run_summary.json').exists():
            continue
        print(f"start full fine-tuning seed {seed} ...", flush=True)
        s = finetune.finetune_run(FULLFT_MODEL, TRAIN, DEV, TEST, seed=seed, out_dir=out_dir, revision=sha,
                                  state_dir='/content/fullft_states')   # weights hashed on local disk, never written to Drive
        print(f"fullft seed {seed}: dev {s['dev']['accuracy']:.2f} test {s['test']['accuracy']:.2f} epoch {s['selected_epoch']} ({s['train_time_s']}s)", flush=True)
""")

code("""# ---- Proposition-1 exact test: cue deletion under E0 for the token-addition-monotone aggregators -----
prop1 = {}
store0 = STORES['E0-hash']
for agg in ['max', 'lse']:
    for seed in SEEDS:
        out_dir = RUNS_DIR / 'E0-hash' / agg / f'seed{seed}'
        state = torch.load(out_dir / 'module_state.pt', map_location='cpu')
        cfg = ctrain.TrainConfig(aggregator=agg, device='cuda' if torch.cuda.is_available() else 'cpu')
        m = cmodel.CACSAModule(DIM['E0-hash'], agg).to(cfg.device); m.load_state_dict(state); m.eval()
        pairs, n_passages = [], 0
        for r in [x for x in TEST if x.edit_id == 0 and x.eligible and x.cue]:
            edited, n = condaqa.delete_cue(r.passage, r.cue)
            if n == 0:
                continue
            n_passages += 1
            full = store0.get_or_encode([condaqa.context_text(r.question, r.passage)])[0].states
            dele = store0.get_or_encode([condaqa.context_text(r.question, edited)])[0].states
            cands = [store0.get(c).states for c in condaqa.render_candidates()]
            pairs.append((ctrain.candidate_scores(m, cfg, full, cands), ctrain.candidate_scores(m, cfg, dele, cands)))
        prop1[f'{agg}/seed{seed}'] = {**metrics.proposition1_violations(pairs), 'passages': n_passages}
        print(agg, seed, prop1[f'{agg}/seed{seed}'])
(RUN / 'proposition1_cue_deletion.json').write_text(json.dumps(prop1, indent=2))
""")

code("""# ---- analysis: per-run metrics, seed means, rules R1-R3, intervals, McNemar ----------------------
runs = analysis.load_runs(RUNS_DIR, split='test')
rows = analysis.per_run_metrics(runs)
seed_means = analysis.seed_mean_table(rows)
cols = ['encoder', 'aggregator', 'seed', 'selected_epoch', 'params_total', 'acc', 'acc_original', 'acc_paraphrase', 'acc_scope',
        'acc_affirmative', 'consistency', 'flip_paraphrase', 'flip_scope', 'flip_affirmative', 'polarity_cost', 'scope_cost',
        'acc_seen_cue', 'acc_unseen_cue', 'unseen_cue_gap']
(RUN / 'per_run_metrics.json').write_text(json.dumps(rows, indent=1))
(RUN / 'seed_means.json').write_text(json.dumps(seed_means, indent=1))
(RUN / 'table_per_run.md').write_text(analysis.markdown_table(rows, cols))
(RUN / 'table_seed_means.md').write_text(analysis.markdown_table(seed_means, ['encoder', 'aggregator', 'n_seeds', 'acc', 'acc_sd', 'acc_original',
    'acc_paraphrase', 'acc_scope', 'acc_affirmative', 'consistency', 'flip_paraphrase', 'flip_scope', 'flip_affirmative', 'polarity_cost', 'scope_cost', 'unseen_cue_gap']))
rules = analysis.evaluate_rules(rows, runs, contextual_encoders=CONTEXTUAL, n_boot=N_BOOT)
(RUN / 'rules_R1_R4.json').write_text(json.dumps(rules, indent=1, default=str))
mcn = []
for enc in CONTEXTUAL:
    for a, b in (('signed_max', 'max'), ('signed_max', 'mean'), ('signed_max_gate', 'signed_max'), ('mean', 'max'), ('lse', 'max')):
        mcn += analysis.mcnemar_table(runs, enc, a, b)
(RUN / 'mcnemar.json').write_text(json.dumps(mcn, indent=1))
print((RUN / 'table_seed_means.md').read_text())
print(json.dumps({k: {e: v.get('met') for e, v in d.items()} for k, d in rules.items() if k != 'R4_descriptive'}, indent=1))
""")

code("""# ---- file manifest, zip, download -----------------------------------------------------------------
n = provenance.write_file_manifest(RUN, RUN / 'FILE_MANIFEST_SHA256.csv')
print('manifested files:', n)
ZIP = Path('/content') / f'{RUN_ID}.zip'
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(x for x in RUN.rglob('*') if x.is_file() and x.suffix not in ('.pt',)):
        z.write(p, arcname=str(p.relative_to(RUN.parent)))
print(ZIP, ZIP.stat().st_size // 1024, 'kB')
from google.colab import files
files.download(str(ZIP))
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "CACSA_v2_main_experiment_colab.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(out, len(cells), "cells")
