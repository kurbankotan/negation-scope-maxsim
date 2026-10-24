"""Generate NegPairs_v3_main_colab.ipynb: the extended experiment of manifest v3.0 (frozen 5 October 2026).

The frozen manifest's SHA-256 and the SHA-256 values of the delivered NegPairs v3.0 data files are embedded; the
notebook stops before any training if the packaged manifest or the data differ.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
MANIFEST = HERE / "CACSA_manifest_v3_0.md"
MAN_SHA = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
assert "Status: FROZEN" in MANIFEST.read_text(encoding="utf-8")
DATA_V3 = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "np_data_v3" / "negpairs_v3"
V3_SHA = {r["path"]: r["sha256"] for r in csv.DictReader(open(DATA_V3 / "FILE_MANIFEST_SHA256.csv", encoding="utf-8"))}
V3_SHA["FILE_MANIFEST_SHA256.csv"] = hashlib.sha256((DATA_V3 / "FILE_MANIFEST_SHA256.csv").read_bytes()).hexdigest()
assert V3_SHA["CACSA_manifest_v3_0.md"] == MAN_SHA

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# NegPairs v3.0 extended experiment (manifest v3.0, frozen 5 October 2026)

**Runtime:** Runtime > Change runtime type > **A100 GPU** (the A100 runtime comes with High-RAM; nothing else to
select). An L4 also works, roughly twice as slow. A T4 cannot run the Qwen audit (last phase).
**Files on Drive (`MyDrive/CACSA_v3/`):** `cacsa2_project.zip` (0.3.1 or later; NegPairs v1 and the frozen manifest
are inside) and the folder `data/negpairs_v3/` with the contents of `negpairs_v3.zip` (perm.jsonl, fresh.jsonl, ...).
**Secrets:** `HF_TOKEN` optional. No Anthropic key is needed.

`Runtime > Run all`. The run folder `MyDrive/CACSA_v3/runs/np_v3_main/` is reused across sessions: if the session
stops, run all cells again and finished work is skipped (every phase writes its result files and checks for them).

Phases (manifest v3.0): data and encoder checks; pilot on train and dev only (exact checks with random
initializations, C1 check, learning-rate grid per encoder); test sets parsed and encoded; main grid of 7 encoders x
8 conditions x 10 seeds (560 runs) plus controls, run by 4 worker processes on the GPU; gate on dev; exact rules
R5, R6, R7 in float64; NevIR; analysis and rules; probing; Qwen audit of NegPairs v1; zip.

| Phase | Expected on an A100 |
|---|---|
| setup, checks, encoders, GloVe | 10 to 15 min (GloVe 822 MB is downloaded once and cached on Drive) |
| pilot: exact checks, C1 check, 63 learning-rate runs | 20 to 30 min |
| test sets, caches, NevIR encoding, packs | 10 min |
| main grid 560 runs + 140 control runs (4 workers) | 2 to 2.5 h |
| C1, 10 seeds | 20 to 40 min |
| exact rules, NevIR, analysis, probing | 20 to 30 min |
| Qwen audit of NegPairs v1 (8,704 prompts) | 10 to 15 min |

At the end two zips are downloaded: `np_v3_main_results.zip` (everything except raw predictions; send this one) and
`np_v3_main_predictions.zip` (raw predictions, compact; larger; keep it on Drive and send it if asked).
""")

code(f"""# ============================== CONFIGURATION (manifest v3.0) ==============================
PROJECT_ROOT  = "/content/drive/MyDrive/CACSA_v3"
PACKAGE_ZIP   = f"{{PROJECT_ROOT}}/cacsa2_project.zip"
MANIFEST_NAME = "CACSA_manifest_v3_0.md"
MANIFEST_SHA256 = "{MAN_SHA}"
DATA_V3_DIR   = f"{{PROJECT_ROOT}}/data/negpairs_v3"
DATA_V3_SHA256 = {json.dumps({k: V3_SHA[k] for k in ("perm.jsonl", "fresh.jsonl", "build_report.json", "FILE_MANIFEST_SHA256.csv")}, indent=1)}
RUN_NAME      = "np_v3_main"                  # fixed name: the folder is reused across sessions (resume)
SEEDS         = [13, 42, 71, 7595, 1676, 101, 202, 303, 404, 505]
LR_GRID, LR_SEEDS = [3e-4, 1e-3, 3e-3], [13, 42, 71]
CONDITIONS    = ["max", "mean", "lse", "signed_max", "signed_max_gate", "signed_max_sentence", "signed_max_oracle", "m1"]
CONTROLS      = ["head_only", "option_only"]
ENCODERS = {{
    "E0-hash":     {{"kind": "hash",  "dim": 384}},
    "E3-glove":    {{"kind": "glove", "dim": 300, "repo": "stanfordnlp/glove", "file": "glove.6B.zip", "member": "glove.6B.300d.txt",
                    "revision": "1db2080b2d94def6e5b0386a523102f9d8849e9d"}},
    "E1-minilm":   {{"kind": "hf", "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"}},
    "E2-deberta":  {{"kind": "hf", "model": "microsoft/deberta-v3-base", "revision": "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"}},
    "E4-mpnet":    {{"kind": "hf", "model": "sentence-transformers/all-mpnet-base-v2", "revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130"}},
    "E5-bge":      {{"kind": "hf", "model": "BAAI/bge-base-en-v1.5", "revision": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"}},
    "E6-colbertv2": {{"kind": "colbert", "model": "colbert-ir/colbertv2.0", "revision": "c1e84128e85ef755c096a95bdb06b47793b13acf"}},
}}
GRID_ORDER    = ["E1-minilm", "E2-deberta", "E4-mpnet", "E5-bge", "E6-colbertv2", "E0-hash", "E3-glove"]
CONTEXTUAL    = ["E1-minilm", "E2-deberta", "E4-mpnet", "E5-bge", "E6-colbertv2"]
STATIC        = ["E0-hash", "E3-glove"]
MAX_LEN_NEGPAIRS, MAX_LEN_NEVIR = 128, 512
NEVIR         = {{"dataset": "orionweller/NevIR", "split": "test"}}
QWEN_MODEL, QWEN_REVISION, QWEN_BATCH = "Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28", 16
C1_MODEL, C1_KW = "microsoft/deberta-v3-base", dict(lr=2e-5, epochs=5, batch_size=16, max_length=128)
EPOCHS, PATIENCE, BATCH = 30, 5, 16
N_WORKERS     = 4
THRESHOLDS = {{"gate_SS_seen_dev": 20.0, "R2_min_gain_unseen_SS": 15.0, "R3_min_gain_flipS": 10.0, "R8_min_SS_perm_bal_seen": 10.0,
              "min_positive_seeds": 9, "n_seeds_required": 10, "family_alpha": 0.05, "min_scenarios_R2": 80,
              "min_scenarios_R3": 100, "min_scenarios_R8_per_order": 40, "exact_tolerance": 1e-9,
              "R1_max_unseen_SS_descriptive": 5.0}}
THRESHOLDS_ACC_SECONDARY = {{"gate_acc_OP": 70.0, "gate_incongruent_acc_OP": 50.0, "gate_margin_fallback": 10.0,
              "min_stratum_scenarios": 30, "R1_min_unseen_negation_gap": 20.0, "R2_min_gain_N": 15.0, "R2_max_loss_OP": -2.0,
              "R3_min_gain_S": 10.0, "R3_min_items": 100}}      # draft-1 accuracy view (v2.0), descriptive only
N_BOOT        = 10000
PILOT_INIT_SEEDS = [1, 2, 3, 4, 5]
PROBE_SEED    = 20261031
# ==========================================================================================
""")

code(r"""%pip install -q "transformers>=4.45" "spacy>=3.8,<3.9" wordfreq==3.1.1 scipy sentencepiece tiktoken protobuf safetensors datasets scikit-learn accelerate
%pip install -q https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
from google.colab import drive
drive.mount('/content/drive')
import os, sys, json, zipfile, time, csv, random, shutil, importlib, hashlib, gzip, subprocess
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import torch
ROOT = Path(PROJECT_ROOT)
assert Path(PACKAGE_ZIP).exists(), f'upload cacsa2_project.zip to {PROJECT_ROOT}/'
ZIP_SHA = hashlib.sha256(open(PACKAGE_ZIP, 'rb').read()).hexdigest()
PKG_DIR = Path('/content') / f'cacsa2_pkg_{ZIP_SHA[:12]}'
shutil.rmtree(PKG_DIR, ignore_errors=True)
with zipfile.ZipFile(PACKAGE_ZIP) as z:
    z.extractall(PKG_DIR)
for _m in [m for m in list(sys.modules) if m == 'cacsa2' or m.startswith('cacsa2.')]:
    del sys.modules[_m]
sys.path = [p for p in sys.path if 'cacsa2_pkg_' not in p]
PKG_ROOT = next(PKG_DIR.rglob('cacsa2/__init__.py')).parent.parent
sys.path.insert(0, str(PKG_ROOT))
importlib.invalidate_caches()
import cacsa2
assert tuple(int(x) for x in cacsa2.__version__.split('.')) >= (0, 3, 1), f'package {cacsa2.__version__}; upload cacsa2_project.zip 0.3.1 or later'
from cacsa2 import (negpairs as npx, np_analysis as npa, np_analysis_v3 as v3, encoders, scope, model as cmodel, train as ctrain,
                    provenance, controls, packs, worker, exact, nevir, probing, colbert, qwen_verify as qv)
try:
    from google.colab import userdata
    _hf = userdata.get('HF_TOKEN')
    if _hf:
        os.environ['HF_TOKEN'] = _hf
except Exception:
    pass
RUN = ROOT / 'runs' / RUN_NAME
RUN.mkdir(parents=True, exist_ok=True)
LOCAL = Path('/content/np_v3_local'); LOCAL.mkdir(exist_ok=True)       # caches and packs (local disk; rebuilt per session)
MAN = PKG_ROOT / MANIFEST_NAME
assert 'Status: FROZEN' in MAN.read_text(encoding='utf-8'), 'the manifest in the package is not frozen'
assert provenance.sha256_file(MAN) == MANIFEST_SHA256, 'the packaged manifest differs from the frozen manifest'
LOCK = RUN / 'manifest_lock.json'
if LOCK.exists():
    assert json.loads(LOCK.read_text())['sha256'] == MANIFEST_SHA256, 'the run folder belongs to another manifest'
else:
    shutil.copy(MAN, RUN / MANIFEST_NAME)
    LOCK.write_text(json.dumps(provenance.lock_record(RUN / MANIFEST_NAME, 'recorded before any training in this run'), indent=1))
GPU = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip() if shutil.which('nvidia-smi') else 'none'
env = provenance.run_environment({'run_name': RUN_NAME, 'cacsa2_version': cacsa2.__version__, 'package_zip_sha256': ZIP_SHA,
      'config': {'seeds': SEEDS, 'conditions': CONDITIONS, 'controls': CONTROLS, 'encoders': ENCODERS, 'epochs': EPOCHS, 'patience': PATIENCE,
                 'batch': BATCH, 'lr_grid': LR_GRID, 'lr_seeds': LR_SEEDS, 'thresholds': THRESHOLDS, 'thresholds_accuracy_secondary': THRESHOLDS_ACC_SECONDARY,
                 'c1': {'model': C1_MODEL, **C1_KW}, 'nevir': NEVIR, 'qwen': {'model': QWEN_MODEL, 'revision': QWEN_REVISION}, 'n_workers': N_WORKERS,
                 'max_len_negpairs': MAX_LEN_NEGPAIRS, 'max_len_nevir': MAX_LEN_NEVIR}})
ENVF = RUN / 'run_environment.json'
sessions = json.loads(ENVF.read_text()) if ENVF.exists() else []
sessions.append(env); ENVF.write_text(json.dumps(sessions, indent=1))
def save_json(name, obj):
    (RUN / name).write_text(json.dumps(obj, indent=1, default=str))
def have(name):
    return (RUN / name).exists()
print('cacsa2', cacsa2.__version__, '| zip', ZIP_SHA[:12], '| GPU:', GPU, '| run folder', RUN, '| session', len(sessions))
""")

code(r"""# ---- phase P1.1: data integrity (train and dev parsed; test files only hashed) ---------------------------------------
V1 = PKG_ROOT / 'data' / 'negpairs_v1'
for row in csv.DictReader(open(V1 / 'FILE_MANIFEST_SHA256.csv', encoding='utf-8')):
    assert provenance.sha256_file(V1 / row['path']) == row['sha256'], f"NegPairs v1 file changed: {row['path']}"
D3 = Path(DATA_V3_DIR)
assert D3.exists(), f'extract negpairs_v3.zip so that {D3}/perm.jsonl exists'
for name, sha in DATA_V3_SHA256.items():
    assert provenance.sha256_file(D3 / name) == sha, f'{name} differs from the audited NegPairs v3.0 file'
for row in csv.DictReader(open(D3 / 'FILE_MANIFEST_SHA256.csv', encoding='utf-8')):
    assert provenance.sha256_file(D3 / row['path']) == row['sha256'], f"NegPairs v3 file changed: {row['path']}"
BUILD = json.loads((D3 / 'build_report.json').read_text())
assert BUILD['manifest']['sha256'] == MANIFEST_SHA256, 'the v3 data were built under another manifest'
assert BUILD['perm']['bag_assertion']['equal_E0_bags_N_S'] == BUILD['perm']['bag_assertion']['scenarios'] > 0, 'S1: Perm bag assertion missing'
TRAIN, DEV = npx.load_split(V1, 'train'), npx.load_split(V1, 'dev')
assert all(r.seen_family for r in TRAIN + DEV), 'S1: unseen-family items in train/dev'
assert not ({r.passage_id for r in TRAIN} & {r.passage_id for r in DEV}), 'S1: scenario in train and dev'
DEV_BY_ID = {r.sample_id: r for r in DEV}
audit = {'v1_files': len(list(csv.DictReader(open(V1 / 'FILE_MANIFEST_SHA256.csv')))), 'v3_files': len(list(csv.DictReader(open(D3 / 'FILE_MANIFEST_SHA256.csv')))),
         'train': {'items': len(TRAIN), 'scenarios': len({r.passage_id for r in TRAIN})}, 'dev': {'items': len(DEV), 'scenarios': len({r.passage_id for r in DEV})},
         'v3_build': {'perm_scenarios': BUILD['perm']['bag_assertion']['scenarios'], 'fresh_scenarios': len(BUILD['fresh']['accepted_by_family']) and sum(BUILD['fresh']['accepted_by_family'].values()),
                      'built_utc': BUILD['built_utc'], 'manifest_sha256': BUILD['manifest']['sha256']},
         'test_files_parsed_in_pilot': False}
save_json('data_audit.json', audit)
print(json.dumps(audit, indent=1))
""")

code(r"""# ---- phase P1.2: encoders (loaded once; HF models stay on the GPU until every text is encoded) ------------------------
from huggingface_hub import hf_hub_download
ENC, STORES, ENC_META = {}, {}, {}
def hf_dtype():
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32
def load_glove_table():
    cache = ROOT / 'cache' / 'glove.6B.300d.pt'
    if cache.exists():
        blob = torch.load(cache, weights_only=False)
        return blob['vocab'], blob['vectors'], blob['source_sha256']
    spec = ENCODERS['E3-glove']
    t0 = time.time()
    path = hf_hub_download(spec['repo'], spec['file'], revision=spec['revision'])
    sha = provenance.sha256_file(path)
    vocab, vecs = [], []
    with zipfile.ZipFile(path) as z, z.open(spec['member']) as fh:
        for raw in fh:
            k = raw.find(b' ')
            vocab.append(raw[:k].decode('utf-8', 'replace')); vecs.append([float(x) for x in raw[k + 1:].split()])
    vectors = torch.tensor(vecs, dtype=torch.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'vocab': vocab, 'vectors': vectors, 'source_sha256': sha, 'member': spec['member'], 'revision': spec['revision']}, cache)
    print(f'GloVe parsed in {(time.time() - t0) / 60:.1f} min and cached on Drive')
    return vocab, vectors, sha
for name, spec in ENCODERS.items():
    t0 = time.time()
    if spec['kind'] == 'hash':
        e = encoders.HashEncoder(dim=spec['dim']); e.name = name; ENC_META[name] = dict(spec)
    elif spec['kind'] == 'glove':
        vocab, vectors, sha = load_glove_table()
        e = encoders.GloveEncoder(dim=spec['dim'], vocab=vocab, vectors=vectors); e.name = name
        ENC_META[name] = {**spec, 'vocabulary': len(vocab), 'source_zip_sha256': sha}
    elif spec['kind'] == 'hf':
        e = encoders.HFFrozenEncoder(spec['model'], max_length=MAX_LEN_NEGPAIRS, name=name, revision=spec['revision'], dtype=hf_dtype(), normalize=True)
        ENC_META[name] = {**spec, 'dim': e.dim, 'normalize': True}
    else:
        W = encoders.ColBERTDocEncoder.load_projection(spec['model'], spec['revision'])
        e = encoders.ColBERTDocEncoder(spec['model'], W, max_length=MAX_LEN_NEGPAIRS, name=name, revision=spec['revision'], dtype=hf_dtype())
        ENC_META[name] = {**spec, 'dim': e.dim, 'projection_shape': list(W.shape), 'doc_marker': '[unused1]', 'normalize': True}
    ENC[name] = e
    STORES[name] = encoders.EncodedStore(e)
    print(f'{name}: loaded in {time.time() - t0:.0f}s (dim {e.dim})', flush=True)
DIM = {n: e.dim for n, e in ENC.items()}
texts_td = sorted({r.text for r in TRAIN + DEV} | {o for r in TRAIN + DEV for o in r.options})
for name, st in STORES.items():
    st.get_or_encode(texts_td, batch_size=128)
    ENC_META[name]['truncated_train_dev'] = int(sum(st.get(t).truncated for t in texts_td))
if 'E3-glove' in ENC:
    ENC_META['E3-glove']['oov_rate_train_dev'] = ENC['E3-glove'].oov_rate()
    ENC_META['E3-glove']['oov_examples'] = sorted(ENC['E3-glove'].oov)[:30]
SANITY = {}
if not have('hf_pipeline_sanity.json'):
    sample = [r.text for r in DEV][:200]
    for name, spec in ENCODERS.items():
        if spec['kind'] in ('hf', 'colbert'):
            SANITY[name] = encoders.hf_load_report(spec['model'], spec['revision'], sample, head='base')
    SANITY['C1_mc_head'] = encoders.hf_load_report(C1_MODEL, ENCODERS['E2-deberta']['revision'], [], head='mc')
    save_json('hf_pipeline_sanity.json', SANITY)
else:
    SANITY = json.loads((RUN / 'hf_pipeline_sanity.json').read_text())
bad = [n for n, s in SANITY.items() if s.get('load_error') or (s.get('unk_rate') or 0) > 0.01]
assert not bad, f'S2: tokenizer or weight-loading problem: {bad}'
save_json('encoder_meta.json', ENC_META)
print({n: (s.get('unk_rate'), s.get('tokenizer_class')) for n, s in SANITY.items()})
""")

code(r"""# ---- phase P1.3: exact checks with random initializations on dev (E0, E3) ----------------------------------------
resolver = scope.ScopeResolver(scope.CueLexicon())
resolver.prefetch([r.text for r in TRAIN + DEV])
if not have('pilot_exact_checks.json'):
    PX = {}
    for name in STATIC:
        st = STORES[name]
        dev_zero = npx.build_items(DEV, st, None)                     # all-zero masks
        dev_scope = npx.build_items(DEV, st, resolver, mask_mode='scope')
        z = {it.row.sample_id: it for it in dev_zero}; s = {it.row.sample_id: it for it in dev_scope}
        t0 = time.time()
        PX[name] = exact.pilot_exact_checks(DIM[name], dev_zero, DEV, lambda r, z=z: z[r.sample_id], PILOT_INIT_SEEDS,
                                            item_of_scope=lambda r, s=s: s[r.sample_id])
        pv = {k: v['violations'] for k, v in PX[name]['permutation'].items()}; iv = {k: v['violations'] for k, v in PX[name]['insertion'].items()}
        print(f"{name}: passed={PX[name]['passed']} | permutation violations {pv} | insertion violations {iv} | {time.time() - t0:.0f}s", flush=True)
    save_json('pilot_exact_checks.json', PX)
else:
    PX = json.loads((RUN / 'pilot_exact_checks.json').read_text())
assert all(v['passed'] for v in PX.values()), 'S4: a Proposition A or B check failed with random initializations'
""")

code(r"""# ---- phase P1.4: C1 check, two epochs on dev (S3) ---------------------------------------------------------------------
if not have('c1_check.json'):
    c1 = npx.finetune_mc_run(C1_MODEL, TRAIN, DEV, [], seed=13, out_dir=RUN / 'pilot' / 'c1_check', revision=ENCODERS['E2-deberta']['revision'],
                             lr=C1_KW['lr'], epochs=2, batch_size=C1_KW['batch_size'], max_length=C1_KW['max_length'], verbose=True)
    recs = [json.loads(l) for l in open(RUN / 'pilot' / 'c1_check' / 'dev_predictions.jsonl')]
    b = npx.report(recs, DEV_BY_ID)['all']
    save_json('c1_check.json', {'dev_accuracy': c1['dev']['accuracy'], 'dev_block': b})
C1C = json.loads((RUN / 'c1_check.json').read_text())
print('C1 check: dev acc_OP', C1C['dev_block']['acc_OP'], 'acc_N', C1C['dev_block']['acc_N'], 'SS', C1C['dev_block']['SS'])
assert C1C['dev_block']['acc_OP'] is not None and C1C['dev_block']['acc_OP'] >= 75, 'S3: C1 check below 75% dev acc_OP'
""")

code(r"""# ---- phase P1.5: learning-rate grid per encoder (A1, seeds 13/42/71, train and dev only; 4 workers) -------------------
PACK_PILOT = LOCAL / 'packs_pilot'; PACK_PILOT.mkdir(exist_ok=True)
if not have('lr_grid.json'):
    for name, st in STORES.items():
        if not (PACK_PILOT / f'{name}_records.json').exists():
            packs.save_pack(PACK_PILOT / name, st, {'train': TRAIN, 'dev': DEV}, resolver)
    tasks = [worker.task_spec(name, 'max', seed, lr, str(RUN / 'pilot' / 'lr_grid' / name / f'lr{lr:g}' / f'seed{seed}'), [],
                              epochs=EPOCHS, patience=PATIENCE, batch=BATCH) for name in GRID_ORDER for lr in LR_GRID for seed in LR_SEEDS]
    worker.run_tasks(tasks, PACK_PILOT, RUN / 'logs', n_workers=N_WORKERS, label='lr_grid')
    grid = {}
    for name in GRID_ORDER:
        rows = []
        for lr in LR_GRID:
            accs = [json.loads((RUN / 'pilot' / 'lr_grid' / name / f'lr{lr:g}' / f'seed{s}' / 'done.json').read_text())['dev_acc'] for s in LR_SEEDS]
            rows.append({'lr': lr, 'dev_acc_by_seed': accs, 'dev_acc_mean': sum(accs) / len(accs)})
        best = max(rows, key=lambda r: (round(r['dev_acc_mean'], 9), -r['lr']))     # highest mean dev accuracy; tie -> smaller lr
        grid[name] = {'grid': rows, 'chosen_lr': best['lr']}
    save_json('lr_grid.json', {'rule': 'A1, seeds 13/42/71: highest mean dev accuracy over all dev items; tie -> smaller lr',
                               'by_encoder': grid, 'test_files_parsed_before': False})
LRG = json.loads((RUN / 'lr_grid.json').read_text())
LR = {n: LRG['by_encoder'][n]['chosen_lr'] for n in GRID_ORDER}
print('chosen learning rates:', LR)
save_json('pilot_report.json', {'exact_checks_passed': {k: v['passed'] for k, v in PX.items()}, 'c1_check_dev_acc_OP': C1C['dev_block']['acc_OP'],
                                'learning_rates': LR, 'stop_conditions_met': []})
""")

code(r"""# ---- phase 2: test sets parsed, integrity, resolver diagnostics, caches, NevIR, packs ------------------------------------
V1TEST, FRESH, PERM = npx.load_split(V1, 'test'), npx.load_split(D3, 'fresh'), npx.load_split(D3, 'perm')
TESTS = {'v1test': V1TEST, 'fresh': FRESH, 'perm': PERM}
BY_ID = {name: {r.sample_id: r for r in rows} for name, rows in TESTS.items()}
ids = [{r.passage_id for r in x} for x in (TRAIN, DEV, V1TEST, FRESH, PERM)]
assert all(not (ids[i] & ids[j]) for i in range(5) for j in range(i + 1, 5)), 'S1: a scenario occurs in two sets'
assert all(exact.bag(n.text) == exact.bag(s.text) for n, s in exact.permutation_pairs(PERM)) and len(exact.permutation_pairs(PERM)) == len({r.passage_id for r in PERM}), 'S1: Perm bags'
all_rows = TRAIN + DEV + V1TEST + FRESH + PERM
texts_all = sorted({r.text for r in all_rows} | {o for r in all_rows for o in r.options})
# neutral twins (probing control): syntactic Perm N and S arms with "also" in place of "not"
TWINS = {}
for r in PERM:
    if r.family == 'syntactic' and r.arm in ('N', 'S'):
        ctx, span = probing.neutral_twin(r)
        TWINS[r.sample_id] = {'context': ctx, 'text': f'{ctx.strip()} {r.question.strip()}', 'cue_span': list(span)}
twin_texts = sorted({t['text'] for t in TWINS.values()})
# NevIR
from huggingface_hub import HfApi
NEVIR_REV = HfApi().dataset_info(NEVIR['dataset']).sha
NEVIR_INST = nevir.load_test(NEVIR['dataset'], NEVIR['split'], revision=NEVIR_REV)
NESTED = exact.nested_instances(NEVIR_INST)
NESTED_IDS = [i['id'] for i in NESTED]
nevir_docs = sorted({i[k] for i in NEVIR_INST for k in ('doc1', 'doc2')}); nevir_qs = sorted({i[k] for i in NEVIR_INST for k in ('q1', 'q2')})
print(f'NevIR test: {len(NEVIR_INST)} instances (revision {NEVIR_REV[:12]}); nested {len(NESTED)} (equal bags {sum(i["contained"] == "equal" for i in NESTED)})')
NEVIR_STORES = {}
for name, st in STORES.items():
    t0 = time.time()
    st.get_or_encode(texts_all + twin_texts, batch_size=128)
    e = ENC[name]
    if name == 'E0-hash':
        ne = encoders.HashEncoder(dim=e.dim, max_tokens=None)
    elif name == 'E3-glove':
        ne = e.with_max_tokens(None)
    else:
        ne = e
    nst = encoders.EncodedStore(ne)
    kw = {} if name in STATIC else {'max_length': MAX_LEN_NEVIR}
    nst.get_or_encode(nevir_docs, batch_size=32, **kw); nst.get_or_encode(nevir_qs, batch_size=128)
    NEVIR_STORES[name] = nst
    ENC_META[name]['nevir_truncated_docs'] = int(sum(nst.get(t).truncated for t in nevir_docs))
    ENC_META[name]['truncated_texts_all'] = int(sum(st.get(t).truncated for t in texts_all))
    print(f'{name}: {len(st)} texts, NevIR {len(nst)} texts ({time.time() - t0:.0f}s)', flush=True)
if 'E3-glove' in ENC:
    ENC_META['E3-glove']['oov_rate_all'] = ENC['E3-glove'].oov_rate()
assert all(ENC_META[n]['nevir_truncated_docs'] == 0 for n in STATIC), 'static encoders must see every NevIR token'
# M0-native (ColBERTv2's own scoring) on NegPairs test sets and NevIR, while the E6 model is loaded
if not have('m0_native_scores.json'):
    CN = colbert.ColBERTNative(ENCODERS['E6-colbertv2']['model'], ENC['E6-colbertv2'].projection.float().cpu(), revision=ENCODERS['E6-colbertv2']['revision'],
                               dtype=hf_dtype(), tokenizer=ENC['E6-colbertv2'].tokenizer, model=ENC['E6-colbertv2'].model)
    native = {'settings': CN.settings, 'negpairs': {}, 'nevir': {}}
    for sname, rows in TESTS.items():
        qs = [o for r in rows for o in r.options]; ds = [r.text for r in rows for _ in r.options]
        sc = CN.score_pairs(qs, ds)
        native['negpairs'][sname] = {r.sample_id: [sc[2 * k], sc[2 * k + 1]] for k, r in enumerate(rows)}
    combos = nevir.four_combos(NEVIR_INST)
    native['nevir'] = nevir.to_score_dict(combos, CN.score_pairs([c[2] for c in combos], [c[3] for c in combos]))
    native['nevir_truncated_docs'] = int(sum(sum(CN.encode_docs(nevir_docs[i:i + 32])[1]) for i in range(0, len(nevir_docs), 32)))
    save_json('m0_native_scores.json', native)
    del CN
for name, spec in ENCODERS.items():                     # free the GPU; every text is cached
    if spec['kind'] in ('hf', 'colbert'):
        del ENC[name].model
torch.cuda.empty_cache()
# resolver diagnostics (Fresh and Perm), S-same ids, Perm bag check per tokenizer
resolver.prefetch(texts_all)
DIAG = {name: npx.arm_diagnostics(rows, resolver) for name, rows in TESTS.items()}
S_SAME = {name: npx.s_same_ids(DIAG[name]) for name in TESTS}
S_SAME_SCEN = {name: sorted({BY_ID[name][i].passage_id for i in S_SAME[name]}) for name in TESTS}
for name in TESTS:
    with open(RUN / f'arm_diagnostics_{name}.jsonl', 'w') as fh:
        for k, d in sorted(DIAG[name].items()):
            fh.write(json.dumps({'sample_id': k, **d}) + '\n')
    save_json(f'S_same_ids_{name}.json', S_SAME[name])
save_json('scope_resolver_report.json', {name: {'by_seen_arm': npx.summarize_diagnostics(DIAG[name], by=('seen_family', 'arm')),
                                                'by_family_arm': npx.summarize_diagnostics(DIAG[name], by=('family', 'arm')),
                                                'S_same_items': len(S_SAME[name]), 'S_same_scenarios': len(S_SAME_SCEN[name])} for name in TESTS}
                                        | {'spacy': f'{resolver.spacy_model} {resolver.spacy_version}'})
PERM_BY_SCEN = defaultdict(dict)
for r in PERM:
    PERM_BY_SCEN[r.passage_id][r.arm] = r
BAGCHECK = {}
for name, st in STORES.items():
    eq = sorted(sid for sid, a in PERM_BY_SCEN.items() if Counter(st.get(a['N'].text).tokens) == Counter(st.get(a['S'].text).tokens))
    BAGCHECK[name] = {'equal': eq, 'n_equal': len(eq), 'n_scenarios': len(PERM_BY_SCEN), 'unequal': sorted(set(PERM_BY_SCEN) - set(eq))}
save_json('perm_token_bag_check.json', {n: {k: v for k, v in b.items() if k != 'equal'} for n, b in BAGCHECK.items()})
PERM_EQ = {n: b['equal'] for n, b in BAGCHECK.items()}
print('Perm scenarios with equal token multisets per encoder:', {n: b['n_equal'] for n, b in BAGCHECK.items()})
audit.update({'test_files_parsed_in_pilot': False, 'v1test': {'items': len(V1TEST)}, 'fresh': {'items': len(FRESH)}, 'perm': {'items': len(PERM)},
              'nevir': {'instances': len(NEVIR_INST), 'nested': len(NESTED), 'revision': NEVIR_REV}})
save_json('data_audit.json', audit); save_json('encoder_meta.json', ENC_META)
PACK_DIR = LOCAL / 'packs'; PACK_DIR.mkdir(exist_ok=True)
for name, st in STORES.items():
    if not (PACK_DIR / f'{name}_records.json').exists():
        t0 = time.time()
        packs.save_pack(PACK_DIR / name, st, {'train': TRAIN, 'dev': DEV, **TESTS}, resolver)
        print(f'pack {name} written ({time.time() - t0:.0f}s)', flush=True)
PACKS = {name: packs.Pack(PACK_DIR / name) for name in STORES}
""")

code(r"""# ---- phase 3: main grid (560 runs), controls (140 runs), M0 (scoring only), C1 (10 seeds) -----------------------------
RUNS_DIR = RUN / 'runs'
TEST_NAMES = list(TESTS)
tasks = [worker.task_spec(name, cond, seed, LR[name], str(RUNS_DIR / name / cond / f'seed{seed}'), TEST_NAMES, epochs=EPOCHS, patience=PATIENCE, batch=BATCH)
         for name in GRID_ORDER for cond in CONDITIONS for seed in SEEDS]
worker.run_tasks(tasks, PACK_DIR, RUN / 'logs', n_workers=N_WORKERS, label='grid')
ctasks = [worker.task_spec(name, cond, seed, LR[name], str(RUNS_DIR / name / cond / f'seed{seed}'), TEST_NAMES, epochs=EPOCHS, patience=PATIENCE, batch=BATCH)
          for name in GRID_ORDER for cond in CONTROLS for seed in SEEDS]
worker.run_tasks(ctasks, PACK_DIR, RUN / 'logs', n_workers=N_WORKERS, label='controls')
for name in GRID_ORDER:                                 # M0: zero-shot MaxSim, scored once (seed0), predictions only
    out = RUNS_DIR / name / 'm0' / 'seed0'
    if (out / 'done.json').exists():
        continue
    out.mkdir(parents=True, exist_ok=True)
    m0 = cmodel.ZeroShotMaxSim(DIM[name]).to('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = ctrain.TrainConfig(aggregator='max', seed=0, batch_size=BATCH, label_set=npx.LABELS2)
    for sname in TEST_NAMES + ['dev']:
        recs, _ = ctrain.predict(m0, PACKS[name].zero_mask_items(sname), cfg, npx.LABELS2)
        ctrain._write_records(out / f'{sname}_predictions.jsonl', recs)
    (out / 'arm.json').write_text(json.dumps({'directory': 'm0', 'module_kind': 'm0', 'polarity_mask': 'none', 'trained': False}))
    (out / 'run_summary.json').write_text(json.dumps({'config': {'aggregator': 'm0', 'seed': 0}, 'parameter_counts': m0.parameter_counts(), 'trained': False}))
    (out / 'done.json').write_text(json.dumps({'encoder': name, 'condition': 'm0', 'seed': 0, 'tests': TEST_NAMES}))
native = json.loads((RUN / 'm0_native_scores.json').read_text())
out = RUNS_DIR / 'E6-colbertv2' / 'm0_native' / 'seed0'
if not (out / 'done.json').exists():
    out.mkdir(parents=True, exist_ok=True)
    for sname, rows in TESTS.items():
        recs = []
        for r in rows:
            s0, s1 = native['negpairs'][sname][r.sample_id]
            z = torch.softmax(torch.tensor([s0, s1], dtype=torch.float64), dim=0).tolist()
            recs.append({'sample_id': r.sample_id, 'passage_id': r.passage_id, 'question_id': r.question_id, 'edit_id': r.edit_id, 'label': r.label,
                         'pred': npx.LABELS2[0 if s0 >= s1 else 1], 'probs': [round(x, 6) for x in z], 'scores': [round(s0, 6), round(s1, 6)], 'cue': r.cue, 'unseen_cue': not r.seen_family})
        ctrain._write_records(out / f'{sname}_predictions.jsonl', recs)
    (out / 'arm.json').write_text(json.dumps({'directory': 'm0_native', 'module_kind': 'colbert_native', 'settings': native['settings'], 'trained': False}))
    (out / 'run_summary.json').write_text(json.dumps({'config': {'aggregator': 'm0_native', 'seed': 0}, 'trained': False}))
    (out / 'done.json').write_text(json.dumps({'encoder': 'E6-colbertv2', 'condition': 'm0_native', 'seed': 0, 'tests': TEST_NAMES}))
print('M0 scored for every encoder; M0-native for E6')
for seed in SEEDS:                                      # C1: fine-tuned reference; raw predictions on the three test sets
    out = RUNS_DIR / 'C1-mcft' / 'deberta-v3-base-mc-ft' / f'seed{seed}'
    if (out / 'done.json').exists():
        continue
    t0 = time.time()
    s = npx.finetune_mc_run(C1_MODEL, TRAIN, DEV, [], seed=seed, out_dir=out, revision=ENCODERS['E2-deberta']['revision'], extra_tests=TESTS, verbose=False, **C1_KW)
    (out / 'done.json').write_text(json.dumps({'encoder': 'C1-mcft', 'condition': 'deberta-v3-base-mc-ft', 'seed': seed, 'dev_acc': s['dev']['accuracy'],
                                               'selected_epoch': s['selected_epoch'], 'tests': TEST_NAMES}))
    print(f"C1 seed {seed}: dev {s['dev']['accuracy']:.1f} epoch {s['selected_epoch']} ({time.time() - t0:.0f}s)", flush=True)
""")

code(r"""# ---- phase 4: gate G on dev (written before any test metric is computed) ---------------------------------------------
def load_runs(split):
    out = {}
    for d in RUNS_DIR.glob('*/*/seed*'):
        f = d / f'{split}_predictions.jsonl'
        if (d / 'done.json').exists() and f.exists():
            out[(d.parent.parent.name, d.parent.name, int(d.name[4:]))] = [json.loads(l) for l in open(f) if l.strip()]
    return out
if not have('gate.json'):
    GATE = v3.dev_gate(load_runs('dev'), DEV_BY_ID, GRID_ORDER, CONTEXTUAL, THRESHOLDS)
    GATE['computed_utc'] = datetime.now(timezone.utc).isoformat()
    save_json('gate.json', GATE)
GATE = json.loads((RUN / 'gate.json').read_text())
print('gate G (dev):', {e: (round(v['A1_seed_mean_dev_SS'], 2), v['passed']) for e, v in GATE['encoders'].items()}, '| k_R2_R3', GATE['k_R2_R3'], 'k_R8', GATE['k_R8'])
""")

code(r"""# ---- phase 5: exact rules R5, R6, R7 in float64 (E0, E3), NevIR scoring ----------------------------------------------
TOL = THRESHOLDS['exact_tolerance']
ZERO = {name: {s: {it.row.sample_id: it for it in PACKS[name].zero_mask_items(s)} for s in TEST_NAMES} for name in STATIC}
SENT = {name: {it.row.sample_id: it for it in PACKS[name].items('perm', 'sentence')} for name in STATIC}
def item_of(name, sname):
    return lambda r: ZERO[name][sname][r.sample_id]
def module(name, cond, seed):
    kind = worker.MODULE_OF.get(cond, cond)
    return exact.load_condition(RUNS_DIR / name / cond / f'seed{seed}' / 'module_state.pt', DIM[name], kind)
if not have('exact_R5.json'):
    R5 = {}
    for name in STATIC:
        for sname, rows in TESTS.items():
            pairs = exact.insertion_pairs(rows)
            R5[f'{name}/{sname}/pairs'] = len(pairs)
            if not pairs:
                continue
            for cond in ('max', 'lse', 'm1'):
                for seed in SEEDS:
                    R5[f'{name}/{sname}/{cond}/seed{seed}'] = exact.check_r5(module(name, cond, seed), item_of(name, sname), pairs, TOL)
            R5[f'{name}/{sname}/m0'] = exact.check_r5(cmodel.ZeroShotMaxSim(DIM[name]), item_of(name, sname), pairs, TOL)
        print(name, 'R5 done', flush=True)
    checks = [v for k, v in R5.items() if isinstance(v, dict)]
    R5['summary'] = {'checks': len(checks), 'violations_total': sum(v['violations'] for v in checks),
                     'met': bool(checks) and all(v['violations'] == 0 for v in checks), 'evaluable': bool(checks)}
    save_json('exact_R5.json', R5)
R5 = json.loads((RUN / 'exact_R5.json').read_text()); print('R5:', R5['summary'])
if not have('exact_R6.json'):
    R6 = {}
    pairs = exact.permutation_pairs(PERM)
    for name in STATIC:
        for cond in ('max', 'mean', 'lse', 'signed_max_gate', 'm1', 'head_only'):
            for seed in SEEDS:
                R6[f'{name}/{cond}/seed{seed}'] = exact.check_r6(module(name, cond, seed), item_of(name, 'perm'), pairs, TOL)
        R6[f'{name}/m0'] = exact.check_r6(cmodel.ZeroShotMaxSim(DIM[name]), item_of(name, 'perm'), pairs, TOL)
        # A6 on the scenarios whose (token, sentence-mask) multisets agree in N and S (Corollary A.3)
        eq = []
        for n, s in pairs:
            tn, ts = STORES[name].get(n.text).tokens, STORES[name].get(s.text).tokens
            if Counter(zip(tn, SENT[name][n.sample_id].polarity.tolist())) == Counter(zip(ts, SENT[name][s.sample_id].polarity.tolist())):
                eq.append((n, s))
        R6[f'{name}/signed_max_sentence/equal_marked_multisets'] = len(eq)
        for seed in SEEDS:
            R6[f'{name}/signed_max_sentence/seed{seed}'] = exact.check_r6(module(name, 'signed_max_sentence', seed), lambda r, nm=name: SENT[nm][r.sample_id], eq, TOL) if eq else {'pairs': 0}
        # GPU (bfloat16) decisions against the float64 decisions, listed, never counted as violations
        dis = []
        RUNS_PERM_GPU = load_runs('perm')
        for cond in ('max', 'mean', 'lse', 'signed_max_gate', 'm1'):
            for seed in SEEDS:
                recs = {r['sample_id']: r for r in RUNS_PERM_GPU[(name, cond, seed)]}
                m = module(name, cond, seed)
                items = [ZERO[name]['perm'][r.sample_id] for n, s in pairs for r in (n, s)]
                sc = exact.scores_f64(m, items)
                for it, s64 in zip(items, sc):
                    g = recs[it.row.sample_id]['pred']
                    f = npx.LABELS2[0 if s64[0] >= s64[1] else 1]
                    if g != f:
                        dis.append({'cond': cond, 'seed': seed, 'item': it.row.sample_id, 'gpu': g, 'f64': f, 'f64_margin': abs(s64[0] - s64[1])})
        R6[f'{name}/gpu_vs_float64_decision_disagreements'] = {'n': len(dis), 'examples': dis[:50]}
        print(name, 'R6 done', flush=True)
    checks = [v for k, v in R6.items() if isinstance(v, dict) and 'score_violations' in v]
    R6['summary'] = {'checks': len(checks), 'score_violations_total': sum(v['score_violations'] for v in checks),
                     'decision_disagreements_total': sum(v['decision_disagreements'] for v in checks), 'near_ties_total': sum(v['near_ties'] for v in checks),
                     'met': bool(checks) and all(v['score_violations'] == 0 and v['decision_disagreements'] == 0 for v in checks), 'evaluable': bool(checks)}
    save_json('exact_R6.json', R6)
R6 = json.loads((RUN / 'exact_R6.json').read_text()); print('R6:', R6['summary'])
# NevIR: M0 for every encoder, M0-native, trained A1 / A3 / M1 of the static encoders (float64); R7 on the nested subset
def nevir_items(name):
    st = NEVIR_STORES[name]; out = []
    for ins in NEVIR_INST:
        for key, q, d in (('q1d1', 'q1', 'doc1'), ('q1d2', 'q1', 'doc2'), ('q2d1', 'q2', 'doc1'), ('q2d2', 'q2', 'doc2')):
            ctx = st.get(ins[d]); qry = st.get(ins[q])
            out.append((ins['id'], key, ctrain.Item(row=None, ctx=ctx.states, cands=[qry.states], polarity=torch.zeros(ctx.states.shape[0], dtype=torch.uint8), label=0)))
    return out
if not have('nevir_results.json') or not have('exact_R7.json'):
    NV = {'instances': len(NEVIR_INST), 'nested': len(NESTED), 'equal_bags': sum(i['contained'] == 'equal' for i in NESTED), 'revision': NEVIR_REV,
          'published_test_values': {'random': 25.0, 'ColBERTv1': 19.7, 'ColBERTv2': 13.0, 'source': 'Weller et al. (2024), Table 2; orientation only'}, 'scorers': {}}
    R7 = {}
    for name in GRID_ORDER:
        items = nevir_items(name)
        if name in STATIC:
            vals = [s[0] for s in exact.scores_f64(cmodel.ZeroShotMaxSim(DIM[name]), [it for _, _, it in items])]
            margin = TOL
        else:
            vals = nevir.maxsim_scores_gpu([it.ctx for _, _, it in items], [it.cands[0] for _, _, it in items], 'cuda' if torch.cuda.is_available() else 'cpu')
            margin = 0.0
        sc = nevir.to_score_dict([(i, k, None, None) for i, k, _ in items], vals)
        NV['scorers'][f'{name}/m0'] = nevir.evaluate_scorer(sc, NEVIR_INST, NESTED_IDS, margin, N_BOOT)
        print(f"NevIR {name} M0: paired {NV['scorers'][f'{name}/m0']['all']['paired_accuracy']:.2f} (nested {NV['scorers'][f'{name}/m0']['nested']['paired_accuracy']})", flush=True)
    native = json.loads((RUN / 'm0_native_scores.json').read_text())
    NV['scorers']['E6-colbertv2/m0_native'] = nevir.evaluate_scorer(native['nevir'], NEVIR_INST, NESTED_IDS, 0.0, N_BOOT)
    NV['scorers']['E6-colbertv2/m0_native']['settings'] = native['settings']; NV['scorers']['E6-colbertv2/m0_native']['truncated_docs'] = native.get('nevir_truncated_docs')
    # trained static modules and R7 (contained vs containing document, both queries)
    def r7_pairs(name, m):
        st = NEVIR_STORES[name]
        def doc_item(ins, q, which):
            ctx = st.get(ins[which]); qry = st.get(q)
            return ctrain.Item(row=None, ctx=ctx.states, cands=[qry.states], polarity=torch.zeros(ctx.states.shape[0], dtype=torch.uint8), label=0)
        items, idx = [], []
        for k, ins in enumerate(NESTED):
            for qi in ('q1', 'q2'):
                for which in ('doc1', 'doc2'):
                    items.append(doc_item(ins, ins[qi], which)); idx.append((k, qi, which))
        sc = exact.scores_f64(m, items)
        s = {}
        for (k, qi, which), v in zip(idx, sc):
            s.setdefault((k, qi), {})[which] = v[0]
        viol, worst, correct = 0, float('-inf'), 0
        for k, ins in enumerate(NESTED):
            for qi in ('q1', 'q2'):
                d1, d2 = s[(k, qi)]['doc1'], s[(k, qi)]['doc2']
                d = (d1 - d2) if ins['contained'] == 'doc1' else (d2 - d1) if ins['contained'] == 'doc2' else abs(d1 - d2)
                worst = max(worst, d); viol += d > TOL
            correct += (s[(k, 'q1')]['doc1'] > s[(k, 'q1')]['doc2'] + TOL) and (s[(k, 'q2')]['doc2'] > s[(k, 'q2')]['doc1'] + TOL)
        return {'instances': len(NESTED), 'violations': viol, 'largest_excess': worst if NESTED else None,
                'paired_accuracy_margin': (100.0 * correct / len(NESTED)) if NESTED else None, 'tolerance': TOL}
    for name in STATIC:
        R7[f'{name}/m0'] = r7_pairs(name, cmodel.ZeroShotMaxSim(DIM[name]))
        for cond in ('max', 'lse', 'm1'):
            for seed in SEEDS:
                m = module(name, cond, seed)
                items = nevir_items(name)
                vals = [s[0] for s in exact.scores_f64(m, [it for _, _, it in items])]
                sc = nevir.to_score_dict([(i, k, None, None) for i, k, _ in items], vals)
                NV['scorers'][f'{name}/{cond}/seed{seed}'] = nevir.evaluate_scorer(sc, NEVIR_INST, NESTED_IDS, TOL, 2000)
                R7[f'{name}/{cond}/seed{seed}'] = r7_pairs(name, m)
        print(name, 'NevIR trained modules and R7 done', flush=True)
    checks = [v for v in R7.values() if isinstance(v, dict) and 'violations' in v]
    R7['summary'] = {'checks': len(checks), 'violations_total': sum(v['violations'] for v in checks), 'nested_instances': len(NESTED),
                     'met': bool(NESTED) and bool(checks) and all(v['violations'] == 0 and v['paired_accuracy_margin'] == 0 for v in checks), 'evaluable': bool(NESTED)}
    save_json('exact_R7.json', R7); save_json('nevir_results.json', NV)
R7 = json.loads((RUN / 'exact_R7.json').read_text()); NV = json.loads((RUN / 'nevir_results.json').read_text())
print('R7:', R7['summary'])
assert R5['summary']['met'] or not R5['summary']['evaluable'], 'R5 violated: a defect in code or data; the study stops here'
assert R6['summary']['met'] or not R6['summary']['evaluable'], 'R6 violated: a defect in code or data; the study stops here'
assert R7['summary']['met'] or not R7['summary']['evaluable'], 'R7 violated: a defect in code or data; the study stops here'
""")

code(r"""# ---- phase 6: analysis (test metrics are computed only now, after gate.json) --------------------------------------------
RUNS = {s: load_runs(s) for s in TEST_NAMES}
ROWS_FRESH, ROWS_V1 = {}, {}
STRATA = npx.option_prior_strata([v for (e, g, sd), v in RUNS['fresh'].items() if g == 'option_only'], BY_ID['fresh'])
STRATA_IDS = npx.strata_item_ids(STRATA, FRESH)
save_json('option_prior_strata_fresh.json', STRATA)
rows_fresh = npa.per_run(RUNS['fresh'], BY_ID['fresh'], extra_subsets={'S_same': S_SAME['fresh'], **STRATA_IDS}, s_same_scenarios=S_SAME_SCEN['fresh'])
rows_v1 = npa.per_run(RUNS['v1test'], BY_ID['v1test'], extra_subsets={'S_same': S_SAME['v1test']}, s_same_scenarios=S_SAME_SCEN['v1test'])
runs_perm = v3.filter_runs(RUNS['perm'], BY_ID['perm'], PERM_EQ)
rows_perm = v3.per_run_perm(runs_perm, BY_ID['perm'])
sm = {'fresh': npa.seed_means(rows_fresh), 'v1test': npa.seed_means(rows_v1), 'perm': npa.seed_means(rows_perm)}
save_json('per_run_fresh.json', rows_fresh); save_json('per_run_v1test.json', rows_v1); save_json('per_run_perm.json', rows_perm)
for s in sm:
    save_json(f'seed_means_{s}.json', sm[s])
(RUN / 'table_fresh_flip.md').write_text(npa.markdown(sm['fresh'], v3.FLIP_TABLE_COLS))
(RUN / 'table_fresh_accuracy.md').write_text(npa.markdown(sm['fresh'], v3.ACC_TABLE_COLS))
(RUN / 'table_v1test_flip.md').write_text(npa.markdown(sm['v1test'], v3.FLIP_TABLE_COLS))
(RUN / 'table_perm.md').write_text(npa.markdown(sm['perm'], v3.PERM_TABLE_COLS))
n_unseen_fresh = len({r.passage_id for r in FRESH if not r.seen_family})
perm_seen_counts = {name: {o: len({sid for sid in PERM_EQ[name] if PERM_BY_SCEN[sid]['O'].seen_family and bool(PERM_BY_SCEN[sid]['O'].perm['decisive_first']) == flag})
                           for o, flag in (('first', True), ('second', False))} for name in GRID_ORDER}
RULES = v3.evaluate_rules_v3(GATE, rows_fresh, RUNS['fresh'], BY_ID['fresh'], rows_perm, runs_perm, BY_ID['perm'], GRID_ORDER, CONTEXTUAL, THRESHOLDS,
                             S_SAME_SCEN['fresh'], n_unseen_fresh, perm_seen_counts, n_boot=N_BOOT)
RULES['R5'] = R5['summary']; RULES['R6'] = R6['summary']; RULES['R7'] = R7['summary']
RULES['subset_sizes'] = {'fresh_unseen_scenarios': n_unseen_fresh, 'fresh_S_same_scenarios': len(S_SAME_SCEN['fresh']), 'perm_seen_by_order': perm_seen_counts}
save_json('rules.json', RULES)
DESC = v3.descriptive_blocks(rows_fresh, RUNS['fresh'], BY_ID['fresh'], rows_perm, runs_perm, BY_ID['perm'], rows_v1, GRID_ORDER, N_BOOT)
save_json('descriptive_blocks.json', DESC)
rules_acc = npa.evaluate_rules_accuracy(rows_fresh, RUNS['fresh'], BY_ID['fresh'], GRID_ORDER, THRESHOLDS_ACC_SECONDARY, n_boot=N_BOOT,
                                        s_same=S_SAME['fresh'], strata_ids=STRATA_IDS)
save_json('rules_accuracy_secondary.json', rules_acc)
mc = []
for enc in GRID_ORDER:
    mc += npa.mcnemar_by_seed(RUNS['fresh'], BY_ID['fresh'], enc, 'signed_max', 'max', ('N',), 'all')
    mc += npa.mcnemar_by_seed(RUNS['fresh'], BY_ID['fresh'], enc, 'signed_max', 'max', ('N',), 'unseen')
    mc += npa.mcnemar_by_seed(RUNS['fresh'], BY_ID['fresh'], enc, 'signed_max', 'signed_max_sentence', ('S',), 'all', ids=S_SAME['fresh'], tag='S_same')
save_json('mcnemar_accuracy_secondary.json', mc)
print('NegPairs-Fresh, scope sensitivity (seed means):'); print((RUN / 'table_fresh_flip.md').read_text())
print('NegPairs-Perm:'); print((RUN / 'table_perm.md').read_text())
print('RULES:', json.dumps(v3.rule_summary(RULES), indent=1), '| exact:', {k: RULES[k]['met'] for k in ('R5', 'R6', 'R7')})
""")

code(r"""# ---- phase 7: probing (clause of the edit), neutral twins, A5 gate AUC ---------------------------------------------------
if not have('probing.json'):
    PR = {}
    for name in GRID_ORDER:
        st = STORES[name]
        feats, labels, groups, seen = [], [], [], []
        tw_feats, tw_labels, tw_groups = [], [], []
        gate_vals = {seed: [] for seed in SEEDS}; gate_labels = []
        a5 = {}
        for seed in SEEDS:
            p = RUNS_DIR / name / 'signed_max_gate' / f'seed{seed}' / 'module_state.pt'
            if p.exists():
                a5[seed] = exact.load_condition(p, DIM[name], 'signed_max_gate').float()
        for r in PERM:
            if r.arm not in ('N', 'S') or r.passage_id not in PERM_EQ[name]:
                continue
            e = st.get(r.text); words = probing.probe_words(r)
            for f, lab in probing.word_features(e.states, e.offsets, words):
                feats.append(f); labels.append(lab); groups.append(r.passage_id); seen.append(r.seen_family)
            if a5:
                ctx = e.states.float().unsqueeze(0)
                for seed, m in a5.items():
                    with torch.no_grad():
                        gv = m.agg.gate_values(ctx)[0]
                    for f, lab in probing.word_features(e.states, e.offsets, words, values=gv.unsqueeze(1)):
                        gate_vals[seed].append(float(f))
                for _, lab in probing.word_features(e.states, e.offsets, words):
                    gate_labels.append(lab)
            if r.sample_id in TWINS:
                tw = TWINS[r.sample_id]; et = st.get(tw['text'])
                twin_row = packs.row_from_dict({**packs.row_to_dict(r), 'context': tw['context']})
                tw_words = probing.probe_words(twin_row, twin_cue_span=tuple(tw['cue_span']))
                for f, lab in probing.word_features(et.states, et.offsets, tw_words):
                    tw_feats.append(f); tw_labels.append(lab); tw_groups.append(r.passage_id)
        res = probing.run_probe(feats, labels, groups, seen, seed=PROBE_SEED)
        syn = [i for i, g in enumerate(groups) if PERM_BY_SCEN[g]['O'].family == 'syntactic']
        res['syntactic_only'] = probing.run_probe([feats[i] for i in syn], [labels[i] for i in syn], [groups[i] for i in syn], [True] * len(syn), seed=PROBE_SEED)
        res['neutral_twins'] = probing.run_probe(tw_feats, tw_labels, tw_groups, [True] * len(tw_labels), seed=PROBE_SEED) if tw_feats else None
        if res['neutral_twins'] and res['syntactic_only']['cv_acc_mean'] is not None:
            res['selectivity_syntactic'] = res['syntactic_only']['cv_acc_mean'] - res['neutral_twins']['cv_acc_mean']
        res['A5_gate_auc_by_seed'] = {seed: probing.gate_auc(vals, gate_labels) for seed, vals in gate_vals.items() if vals}
        aucs = [v for v in res['A5_gate_auc_by_seed'].values() if v is not None]
        res['A5_gate_auc_mean'] = (sum(aucs) / len(aucs)) if aucs else None
        PR[name] = res
        print(f"{name}: probe CV {res['cv_acc_mean']:.1f} | transfer {res.get('transfer', {}).get('acc')} | twins {res['neutral_twins'] and round(res['neutral_twins']['cv_acc_mean'], 1)} "
              f"| selectivity {res.get('selectivity_syntactic')} | A5 gate AUC {res['A5_gate_auc_mean']}", flush=True)
    save_json('probing.json', PR)
PR = json.loads((RUN / 'probing.json').read_text())
""")

code(r"""# ---- phase 8: Qwen audit of NegPairs v1 (cross-family verification, descriptive) and sensitivity tables ----------------
if not have('qwen_audit_v1.json'):
    V1_SCEN = [json.loads(l) for l in open(V1 / 'scenarios.jsonl', encoding='utf-8')]
    QWEN = qv.QwenVerifier(QWEN_MODEL, revision=QWEN_REVISION, batch_size=QWEN_BATCH)
    QA = {'model': QWEN_MODEL, 'revision': QWEN.revision, 'scenarios': {}}
    texts, owners = [], []
    for s in V1_SCEN:
        texts.extend(QWEN.scenario_texts({a: s['context_' + a] for a in 'OPNS'}, s['question'], s['options'])); owners.append(s)
    t0 = time.time(); logits = QWEN.logits3(texts); print(f'Qwen: {len(texts)} prompts in {(time.time() - t0) / 60:.1f} min')
    for k, s in enumerate(owners):
        ok, det = qv.scenario_check(logits[8 * k: 8 * k + 8], s['answer_O'])
        QA['scenarios'][s['uid']] = {'pass': ok, 'split': s['split'], 'family': s['family'], 'verdicts': {a: det[a]['verdict'] for a in 'OPNS'}, 'expected': {a: det[a]['expected'] for a in 'OPNS'}}
    agg = defaultdict(lambda: [0, 0])
    for u, v in QA['scenarios'].items():
        for key in (f"split/{v['split']}", f"family/{v['family']}", 'all'):
            agg[key][0] += v['pass']; agg[key][1] += 1
        for a in 'OPNS':
            agg[f"arm/{a}"][0] += v['verdicts'][a] == v['expected'][a]; agg[f"arm/{a}"][1] += 1
    QA['agreement'] = {k: {'agree': a, 'n': n, 'pct': 100.0 * a / n} for k, (a, n) in sorted(agg.items())}
    save_json('qwen_audit_v1.json', QA)
    del QWEN; torch.cuda.empty_cache()
QA = json.loads((RUN / 'qwen_audit_v1.json').read_text())
print('Qwen audit of v1:', {k: round(v['pct'], 1) for k, v in QA['agreement'].items()})
passing = {u for u, v in QA['scenarios'].items() if v['pass']}
runs_v1_pass = v3.filter_runs(RUNS['v1test'], BY_ID['v1test'], {name: passing for name in list(GRID_ORDER) + ['C1-mcft']})
rows_v1_pass = npa.per_run(runs_v1_pass, BY_ID['v1test'])
save_json('per_run_v1test_qwen_passing.json', rows_v1_pass)
(RUN / 'table_v1test_flip_qwen_passing.md').write_text(npa.markdown(npa.seed_means(rows_v1_pass), v3.FLIP_TABLE_COLS))
print(f"v1 test scenarios passing Qwen: {len([u for u in passing if QA['scenarios'][u]['split'] == 'test'])} of {sum(v['split'] == 'test' for v in QA['scenarios'].values())}")
""")

code(r"""# ---- phase 9: file manifest, zips, download ------------------------------------------------------------------------------
summary = {'rules': v3.rule_summary(RULES), 'exact': {k: RULES[k] for k in ('R5', 'R6', 'R7')}, 'learning_rates': LR,
           'gate': {e: v['passed'] for e, v in GATE['encoders'].items()}, 'runs': len(list(RUNS_DIR.glob('*/*/seed*/done.json'))),
           'nevir': {k: v['all']['paired_accuracy'] for k, v in NV['scorers'].items() if '/seed' not in k},
           'probing_cv': {n: v['cv_acc_mean'] for n, v in PR.items()}, 'qwen_v1_pass_pct': QA['agreement']['all']['pct'], 'gpu': GPU}
save_json('summary.json', summary)
n = provenance.write_file_manifest(RUN, RUN / 'FILE_MANIFEST_SHA256.csv', exclude=('logs/',))
RES_ZIP, PRED_ZIP = ROOT / f'{RUN_NAME}_results.zip', ROOT / f'{RUN_NAME}_predictions.zip'
with zipfile.ZipFile(RES_ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(x for x in RUN.rglob('*') if x.is_file() and x.suffix != '.pt' and not x.name.endswith('_predictions.jsonl')):
        z.write(p, arcname=str(p.relative_to(RUN.parent)))
with zipfile.ZipFile(PRED_ZIP, 'w', zipfile.ZIP_DEFLATED) as z:        # compact raw predictions: id, pred, probs (4 decimals)
    for p in sorted(RUN.rglob('*_predictions.jsonl')):
        rows = [json.loads(l) for l in open(p) if l.strip()]
        compact = '\n'.join(json.dumps({'sample_id': r['sample_id'], 'pred': r['pred'], 'probs': [round(x, 4) for x in r['probs']]}) for r in rows)
        z.writestr(str(p.relative_to(RUN.parent)), compact)
print(f'{n} files manifested | results {RES_ZIP.stat().st_size // 1024 // 1024} MB | predictions {PRED_ZIP.stat().st_size // 1024 // 1024} MB')
print(json.dumps(summary, indent=1))
from google.colab import files
files.download(str(RES_ZIP))
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
out = HERE / "NegPairs_v3_main_colab.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(out, len(cells), "cells | manifest", MAN_SHA[:12])
