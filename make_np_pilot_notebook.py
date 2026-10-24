"""Generate the NegPairs pilot notebook (train and dev only; the test split is never read)."""
import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# NegPairs pilot (train and dev only)

**Runtime:** Colab A100 (a T4 also works). Needs `cacsa2_project.zip` (0.2.3 or later) and the built dataset folder
`MyDrive/CACSA_v2/data/negpairs_v1/`. `Runtime > Run all`.

**Purpose.** Before any confirmatory run: (1) check the dataset files against their SHA-256 manifest; (2) describe the
frozen scope resolver on dev: cue detection in N and S, false positives in O and P, whether its scope covers the
decisive words, and how often the S cue shares a sentence with the decisive words; (3) test Proposition 1 exactly
(float64) on dev insertion pairs with randomly initialised and trained modules; (4) choose the learning rate for A1
(max) on E1 by dev accuracy; (5) check that A1 learns the affirmative arms on each encoder and by how much it beats the
context-blind control C2 (option-only); (6) check the Hugging Face pipeline itself: tokenizer unknown-token rate, weight
keys reported as missing or unexpected, and a short fine-tuning run of the C1 reference on dev. Only A1, C2 and that C1
check are trained here, so no aggregator comparison is seen before the confirmatory protocol is frozen.
**The test split is not loaded.**

| Cell | Expected |
|---|---|
| setup, data audit, resolver diagnostics | 1–2 min |
| encoder caches (E0, E1, E2), pipeline sanity incl. short C1 check | 3–6 min |
| Proposition-1 checks | 1–2 min |
| learning-rate grid (3 runs), A1 and C2 on three encoders | 5–15 min |
| report, zip | < 1 min |
""")

code("""# ============================== CONFIGURATION ==============================
PROJECT_ROOT = "/content/drive/MyDrive/CACSA_v2"
PACKAGE_ZIP  = f"{PROJECT_ROOT}/cacsa2_project.zip"
DATA_NAME    = "negpairs_v1"
ENCODERS = {
    "E0-hash":    {"kind": "hash", "dim": 384},
    "E1-minilm":  {"kind": "hf", "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "max_length": 128, "normalize": True},
    "E2-deberta": {"kind": "hf", "model": "microsoft/deberta-v3-base", "max_length": 128, "normalize": True},
}
LR_GRID, LR_GRID_ENCODER = [3e-4, 1e-3, 3e-3], "E1-minilm"
EPOCHS, PATIENCE, BATCH, SEED = 30, 5, 16, 13
C1_MODEL, C1_CHECK_EPOCHS = "microsoft/deberta-v3-base", 2      # short pipeline check of the fine-tuned reference (dev only)
# ===========================================================================
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
assert _v >= (0, 2, 3), f'cacsa2_project.zip on Drive is version {cacsa2.__version__}; upload the newest cacsa2_project.zip (>= 0.2.3)'
from cacsa2 import negpairs as npx, encoders, scope, model as cmodel, train as ctrain, prop1, provenance, controls
DATA = ROOT / 'data' / DATA_NAME
RUN_ID = datetime.now(timezone.utc).strftime('np_pilot_%Y%m%dT%H%M%SZ')
RUN = ROOT / 'runs' / RUN_ID; RUN.mkdir(parents=True, exist_ok=True)
(RUN / 'run_environment.json').write_text(json.dumps(provenance.run_environment({
    'run_id': RUN_ID, 'cacsa2_version': cacsa2.__version__, 'package_zip_sha256': provenance.sha256_file(PACKAGE_ZIP),
    'purpose': 'NegPairs pilot on train/dev only', 'config': {'encoders': ENCODERS, 'lr_grid': LR_GRID, 'epochs': EPOCHS,
    'patience': PATIENCE, 'batch': BATCH, 'seed': SEED}}), indent=2))
print('cacsa2', cacsa2.__version__, '| run', RUN)
""")

code("""# ---- dataset integrity and audit (train/dev only) ----------------------------------------------------------
man = list(csv.DictReader(open(DATA / 'FILE_MANIFEST_SHA256.csv')))
bad = [m['path'] for m in man if (DATA / m['path']).exists() and provenance.sha256_file(DATA / m['path']) != m['sha256']]
assert not bad, f'dataset files changed since they were built: {bad}'
print(f'dataset manifest verified: {len(man)} files')
TRAIN, DEV = npx.load_split(DATA, 'train'), npx.load_split(DATA, 'dev')
audit = {s: {'items': len(rows), 'scenarios': len({r.passage_id for r in rows}),
             'by_family_arm': dict(Counter(f'{r.family}/{r.arm}' for r in rows)),
             'label_balance': dict(Counter(r.label for r in rows)),
             'label_balance_by_arm': {a: dict(Counter(r.label for r in rows if r.arm == a)) for a in 'OPSN'},
             'decisive_field_present': sum(bool(r.decisive) for r in rows)} for s, rows in (('train', TRAIN), ('dev', DEV))}
assert not ({r.passage_id for r in TRAIN} & {r.passage_id for r in DEV}), 'scenario overlap between train and dev'
assert all(r.seen_family for r in TRAIN + DEV), 'unseen-family items found in train/dev'
(RUN / 'data_audit.json').write_text(json.dumps(audit, indent=1))
print(json.dumps(audit, indent=1))
""")

code("""# ---- frozen scope resolver on dev: cue detection, false positives, scope over decisive words, sentence relation ----
resolver = scope.ScopeResolver(scope.CueLexicon())        # core list + multiword cues + affixal rule; no dataset cues
DIAG_DEV = npx.arm_diagnostics(DEV, resolver)
resolver.prefetch([r.text for r in TRAIN])
rep = {'by_arm': npx.summarize_diagnostics(DIAG_DEV, by=('arm',)),
       'by_family_arm': npx.summarize_diagnostics(DIAG_DEV, by=('family', 'arm')),
       'S_same_items_dev': len(npx.s_same_ids(DIAG_DEV)),
       'spacy': f'{resolver.spacy_model} {resolver.spacy_version}',
       'reading': {'cue_detected': 'N/S items whose declared cue overlaps a detected cue',
                   'any_cue_detected (O/P)': 'false positives: any cue detected in an item without negation',
                   'decisive_in_scope': 'N: should be high (scope reaches the decisive words); S: should be low',
                   'same_sentence (S)': 'share of S items on which sentence-wide and dependency scope can disagree (R3 subset)'}}
with open(RUN / 'arm_diagnostics_dev.jsonl', 'w') as fh:
    for k, d in sorted(DIAG_DEV.items()):
        fh.write(json.dumps({'sample_id': k, **d}) + '\\n')
(RUN / 'scope_resolver_dev_report.json').write_text(json.dumps(rep, indent=1))
print(json.dumps({k: rep[k] for k in ('by_arm', 'S_same_items_dev', 'spacy')}, indent=1))
""")

code("""# ---- token-state caches -------------------------------------------------------------------------------------
from huggingface_hub import HfApi
STORES, ENC_META = {}, {}
texts = sorted({r.text for r in TRAIN + DEV} | {o for r in TRAIN + DEV for o in r.options})
for name, spec in ENCODERS.items():
    t0 = time.time()
    if spec['kind'] == 'hash':
        enc = encoders.HashEncoder(dim=spec['dim']); enc.name = name; ENC_META[name] = dict(spec)
    else:
        sha = None if Path(spec['model']).exists() else HfApi().model_info(spec['model']).sha
        enc = encoders.HFFrozenEncoder(spec['model'], max_length=spec['max_length'], name=name, revision=sha,
                                       dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, normalize=spec['normalize'])
        ENC_META[name] = {**spec, 'revision': sha, 'dim': enc.dim}
    store = encoders.EncodedStore(enc)
    store.get_or_encode(texts, batch_size=128)
    ENC_META[name]['truncated_texts'] = int(sum(store.get(t).truncated for t in texts))
    STORES[name] = store
    if spec['kind'] == 'hf':
        del enc.model; torch.cuda.empty_cache()
    print(f'{name}: {len(store)} texts in {time.time()-t0:.0f}s; truncated {ENC_META[name]["truncated_texts"]}', flush=True)
(RUN / 'encoder_meta.json').write_text(json.dumps(ENC_META, indent=1))
ITEMS = {n: {'train': npx.build_items(TRAIN, s, resolver), 'dev': npx.build_items(DEV, s, resolver)} for n, s in STORES.items()}
DIM = {n: s.encoder.dim for n, s in STORES.items()}
dev_by_id = {r.sample_id: r for r in DEV}
""")

code("""# ---- Hugging Face pipeline sanity: tokenizer, weight keys, and a short C1 fine-tuning check on dev ----------------------
from huggingface_hub import HfApi
sample = [r.text for r in DEV][:400] + sorted({o for r in DEV for o in r.options})[:200]
SANITY = {}
for name, spec in ENCODERS.items():
    if spec['kind'] != 'hf':
        continue
    SANITY[name] = encoders.hf_load_report(spec['model'], ENC_META[name]['revision'], sample, head='base')
    print(name, {k: SANITY[name].get(k) for k in ('tokenizer_class', 'unk_rate', 'missing_keys', 'unexpected_keys_n', 'load_error')})
    print('   example tokens:', SANITY[name]['example_tokens'])
C1_REV = None if Path(C1_MODEL).exists() else HfApi().model_info(C1_MODEL).sha
SANITY['C1_mc_head'] = encoders.hf_load_report(C1_MODEL, C1_REV, [], head='mc')
print('C1 multiple-choice head:', {k: SANITY['C1_mc_head'].get(k) for k in ('missing_keys', 'unexpected_keys_n', 'load_error')})
c1 = npx.finetune_mc_run(C1_MODEL, TRAIN, DEV, [], seed=SEED, out_dir=RUN / 'c1_check', revision=C1_REV, lr=2e-5,
                         epochs=C1_CHECK_EPOCHS, batch_size=16, max_length=128, state_dir='/content/fullft_states')
c1_recs = [json.loads(l) for l in open(RUN / 'c1_check' / 'dev_predictions.jsonl')]
SANITY['C1_check_dev'] = {'selected_epoch': c1['selected_epoch'], 'history': c1['history'],
                          **npx.report(c1_recs, {r.sample_id: r for r in DEV})['all']}
b = SANITY['C1_check_dev']
print(f"C1 check (dev, {C1_CHECK_EPOCHS} epochs): acc {b['acc']:.1f} | O {b['acc_O']:.1f} P {b['acc_P']:.1f} S {b['acc_S']:.1f} N {b['acc_N']:.1f}")
STOPS = []                                                  # manifest v2.0, section 11.3
for name in [n for n in ENCODERS if n in SANITY]:
    r = SANITY[name]
    if r.get('load_error'): STOPS.append(f"S2 {name}: load error {r['load_error'][:120]}")
    if (r.get('unk_rate') or 0) > 0.01: STOPS.append(f"S2 {name}: unknown-token rate {r['unk_rate']:.3f} > 0.01")
    miss = [k for k in r.get('missing_keys', []) if 'pooler' not in k]   # an unused pooler may be absent from encoder checkpoints
    if miss: STOPS.append(f'S2 {name}: backbone weights missing, e.g. {miss[:3]}')
if b['acc_OP'] is None or b['acc_OP'] < 75: STOPS.append(f"S3 C1 check: dev acc_OP {b['acc_OP']} < 75")
SANITY['stop_conditions_so_far'] = STOPS
print('stop conditions S2/S3:', STOPS or 'none')
(RUN / 'hf_pipeline_sanity.json').write_text(json.dumps(SANITY, indent=1, default=str))
""")

code("""# ---- Proposition 1, exact (float64): dev insertion pairs, random modules -------------------------------------
pairs = npx.insertion_pairs(DEV)
print(f'dev insertion pairs (X = O plus inserted tokens): {len(pairs)}  by arm {dict(Counter(x.arm for _, x in pairs))}')
s0 = STORES['E0-hash']
p1 = {}
for agg in ('max', 'lse', 'mean', 'signed_max'):
    viol = []
    for k in range(5):
        torch.manual_seed(1000 + k)
        m = cmodel.CACSAModule(DIM['E0-hash'], agg).double().eval()
        for p_ in m.parameters():
            with torch.no_grad(): p_.add_(torch.randn_like(p_) * 0.3)
        sc = []
        for o_row, x_row in pairs:
            cands = [s0.get(o).states for o in o_row.options]
            pol_x = torch.tensor(resolver.polarity_mask(x_row.text, s0.get(x_row.text).offsets), dtype=torch.uint8)
            pol_o = torch.tensor(resolver.polarity_mask(o_row.text, s0.get(o_row.text).offsets), dtype=torch.uint8)
            sc.append((prop1.option_scores(m, s0.get(x_row.text).states, cands, pol_x), prop1.option_scores(m, s0.get(o_row.text).states, cands, pol_o)))
        viol.append(prop1.violations(sc, tol=1e-9))
    p1[agg] = {'random_inits': 5, 'violations_per_init': [v['violations'] for v in viol], 'n_scores_per_init': viol[0]['n_scores'],
               'max_increase_after_deletion': max(v['max_increase_after_deletion'] for v in viol)}
    print(agg, p1[agg])
(RUN / 'proposition1_dev_random.json').write_text(json.dumps(p1, indent=1))
print('expected: zero violations for max and lse (monotone under token addition); mean and signed_max may violate')
""")

code("""# ---- learning-rate grid: A1 (max) on LR_GRID_ENCODER, seed SEED, dev accuracy over all arms -------------------
grid = []
for lr in LR_GRID:
    cfg = ctrain.TrainConfig(aggregator='max', lr=lr, epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH, seed=SEED)
    res = ctrain.train_run(cfg, DIM[LR_GRID_ENCODER], ITEMS[LR_GRID_ENCODER]['train'], ITEMS[LR_GRID_ENCODER]['dev'], None,
                           out_dir=RUN / 'lr_grid' / f'lr{lr:g}', label_set=npx.LABELS2, verbose=False)
    rpt = npx.report(res['dev']['records'], dev_by_id)['all']
    grid.append({'lr': lr, 'dev_acc': res['dev']['accuracy'], 'selected_epoch': res['selected_epoch'], 'dev_acc_OP': rpt['acc_OP']})
    print(grid[-1], flush=True)
best_lr = max(grid, key=lambda g: (g['dev_acc'], -g['lr']))['lr']
(RUN / 'lr_grid.json').write_text(json.dumps({'grid': grid, 'chosen_lr': best_lr, 'rule': 'highest dev accuracy over all arms; tie -> smaller lr'}, indent=1))
print('chosen lr', best_lr)
""")

code("""# ---- learnability: A1 (max) and the context-blind control C2 on every encoder (dev, by arm) ------------------------
pilot = {}
f1 = lambda v: '—' if v is None else f'{v:.1f}'
def show(tag, rpt, ep):
    print(f"{tag:24s} dev acc {f1(rpt['acc'])} | O {f1(rpt['acc_O'])} P {f1(rpt['acc_P'])} S {f1(rpt['acc_S'])} N {f1(rpt['acc_N'])} | "
          f"N|O {f1(rpt['acc_N_given_O'])} S|O {f1(rpt['acc_S_given_O'])} | flip O->N {f1(rpt['flip_O_N'])} O->S {f1(rpt['flip_O_S'])} | epoch {ep}", flush=True)
RECS = {}
for name in ENCODERS:
    for tag, kw in (('A1', {}), ('C2', {'model_factory': controls.option_only_factory})):
        cfg = ctrain.TrainConfig(aggregator='max' if tag == 'A1' else 'option_only', lr=best_lr, epochs=EPOCHS,
                                 patience=PATIENCE, batch_size=BATCH, seed=SEED)
        res = ctrain.train_run(cfg, DIM[name], ITEMS[name]['train'], ITEMS[name]['dev'], None,
                               out_dir=RUN / ('a1' if tag == 'A1' else 'c2') / name, label_set=npx.LABELS2, verbose=False, **kw)
        RECS[(name, tag)] = res['dev']['records']
        rpt = npx.report(res['dev']['records'], dev_by_id)['all']
        pilot[f'{name}/{tag}'] = {'selected_epoch': res['selected_epoch'], 'train_acc_last': res['history'][-1]['train_acc'],
                                  'parameters': res['parameter_counts']['total'], **rpt}
        show(f'{name} {tag}', rpt, res['selected_epoch'])
    m_ = pilot[f'{name}/A1']['acc_OP'] - pilot[f'{name}/C2']['acc_OP']
    pilot[f'{name}/A1_minus_C2_acc_OP'] = m_
    print(f'   {name}: A1 - C2 on dev acc_OP = {m_:.1f} points (planned gate: A1 >= 70 and A1 - C2 >= 10 on test)', flush=True)
    # A1 within option-prior strata defined by this encoder's C2 (incongruent: the option-only prior misses the O answer)
    ids = npx.strata_item_ids(npx.option_prior_strata([RECS[(name, 'C2')]], dev_by_id), DEV)
    for k, v in ids.items():
        keep = set(v)
        b = npx.report([r for r in RECS[(name, 'A1')] if r['sample_id'] in keep], dev_by_id)['all'] if keep else {}
        pilot[f'{name}/A1/{k}'] = b
        if keep:
            print(f"   A1 on {k:17s} ({b['n_scenarios']:3d} scenarios): O {f1(b['acc_O'])} P {f1(b['acc_P'])} N {f1(b['acc_N'])} "
                  f"N|O {f1(b['acc_N_given_O'])}", flush=True)
pooled = npx.option_prior_strata([RECS[(n, 'C2')] for n in ENCODERS], dev_by_id)
pilot['option_prior_pooled_dev'] = {'scenarios': len(pooled), 'incongruent': sum(v == 'incongruent' for v in pooled.values())}
print('option-only prior pooled over encoders (dev):', pilot['option_prior_pooled_dev'])
m = prop1.load_module(str(RUN / 'a1' / 'E0-hash' / 'module_state.pt'), DIM['E0-hash'], 'max')
sc = [(prop1.option_scores(m, s0.get(x.text).states, [s0.get(o).states for o in o_.options]),
       prop1.option_scores(m, s0.get(o_.text).states, [s0.get(o).states for o in o_.options])) for o_, x in pairs]
pilot['prop1_trained_E0_max'] = prop1.violations(sc, tol=1e-9)
print('Proposition 1, trained E0 A1:', pilot['prop1_trained_E0_max'])
if pilot['prop1_trained_E0_max']['violations'] or any(sum(p1[a]['violations_per_init']) for a in ('max', 'lse')):
    STOPS.append('S4 Proposition-1 violation by A1 or A3 under E0')
pilot['stop_conditions'] = STOPS
(RUN / 'pilot_a1_c2_dev.json').write_text(json.dumps(pilot, indent=1))
print('\\nPILOT VERDICT:', 'no stop condition met (S1 is enforced by the asserts above)' if not STOPS else f'STOP: {STOPS}')
""")

code("""# ---- manifest, zip, download ------------------------------------------------------------------------------------
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
out = Path(__file__).parent / "NegPairs_pilot_colab.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(out, len(cells), "cells")
