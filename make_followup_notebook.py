"""Generate the follow-up notebook for the v1.1 run: (1) float64 re-check of the Proposition-1 test on the archived
E0 states, with a parity check against the archived scores; (2) re-execution of control C1 (full fine-tuning on CondaQA)
with the precision fix of cacsa2 0.1.2."""
import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# CACSA v1.1 follow-up: Proposition-1 re-check (float64) and C1 re-execution (float32)

**Runtime:** Colab A100. `Runtime > Run all`. Requires `cacsa2_project.zip` **0.1.2** in `MyDrive/CACSA_v2/`.

1. **Proposition-1 re-check.** The v1.1 run (`main_20261003T050337Z`) reported 5 to 28 score increases per seed (at most
   6.8e-5) for the max aggregator when cue tokens were deleted, against a tolerance of 1e-5; log-sum-exp had none. This
   notebook finds that run's E0 module states on Drive, first checks that recomputing the archived test scores reproduces
   them (parity), then repeats the cue-deletion test in float64 on CPU (no TF32), tolerance 1e-9.
2. **C1 re-execution.** In v1.1 every C1 step was NaN: transformers 5 loaded DeBERTa-v3-base in half precision by default.
   cacsa2 0.1.2 loads float32 explicitly and stops on any non-finite loss. Settings are unchanged from manifest v1.1
   (lr 2e-5, batch 16, 3 epochs, 6% warm-up, max length 512, seeds 13, 42, 71, 7595, 1676).

| Cell | Expected |
|---|---|
| install + Drive + CondaQA | 2–3 min |
| Proposition-1 re-check (10 models, CPU) | 3–10 min, one line per model |
| C1, five seeds | about 10–15 min per seed, one line per epoch |
| metrics, manifest, zip | 1 min |
""")

code("""# ============================== CONFIGURATION ==============================
PROJECT_ROOT   = "/content/drive/MyDrive/CACSA_v2"
PACKAGE_ZIP    = f"{PROJECT_ROOT}/cacsa2_project.zip"
OLD_RUN_ID     = "main_20261003T050337Z"
CONDAQA_REPO   = "https://github.com/AbhilashaRavichander/CondaQA.git"
CONDAQA_COMMIT = "bd4857f1f2819937f1505925ad8c809544e8e5ef"
CONDAQA_SHA256 = {
    "condaqa_train.json": "f5f8a9d1b64a6c3b111b00d2e125f1d7f4da8b50acfdbc788d2901e2e43d79ec",
    "condaqa_dev.json":   "b9ec9ab7453fbdc61fbed988119c03f507341ac495f1f5f95b13b27f59e639a2",
    "condaqa_test.json":  "ffd7f4ad85689b2317283430a226cd759c3ccc40070e6a32652e7ae84ae90e8e",
}
SEEDS          = [13, 42, 71, 7595, 1676]
FULLFT_MODEL   = "microsoft/deberta-v3-base"
FULLFT_KW      = dict(lr=2e-5, epochs=3, batch_size=16, max_length=512)
RUN_PROP1, RUN_C1 = True, True
LIMIT_ROWS     = None            # testing only: cap the number of train/dev/test rows
# ===========================================================================
""")

code("""%pip install -q "transformers>=4.45" "spacy>=3.8,<3.9" wordfreq scipy sentencepiece tiktoken protobuf
from google.colab import drive
drive.mount('/content/drive')
import os, sys, json, zipfile, time, subprocess, random
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
from datetime import datetime, timezone
ROOT = Path(PROJECT_ROOT)
with zipfile.ZipFile(PACKAGE_ZIP) as z: z.extractall('/content/cacsa2_unzipped')
PKG_ROOT = str(next(Path('/content/cacsa2_unzipped').rglob('cacsa2/__init__.py')).parent.parent)
sys.path.insert(0, PKG_ROOT)
import torch, cacsa2
from cacsa2 import condaqa, encoders, train as ctrain, metrics, analysis, provenance, prop1
PACKAGE_SHA256 = provenance.sha256_file(PACKAGE_ZIP)
print('cacsa2', cacsa2.__version__, '| zip', PACKAGE_SHA256[:16], '| torch', torch.__version__, '| cuda', torch.cuda.is_available())
assert cacsa2.__version__ >= '0.1.2', 'upload cacsa2_project.zip 0.1.2'
RUN_ID = datetime.now(timezone.utc).strftime('followup_v11_%Y%m%dT%H%M%SZ')
RUN = ROOT / 'runs' / RUN_ID; RUN.mkdir(parents=True, exist_ok=True)
env = provenance.run_environment({'run_id': RUN_ID, 'cacsa2_version': cacsa2.__version__, 'package_zip_sha256': PACKAGE_SHA256,
                                  'purpose': 'follow-up of main_20261003T050337Z under manifest v1.1: Proposition-1 float64 re-check; C1 re-execution with float32 fix',
                                  'old_run_id': OLD_RUN_ID, 'fullft': {'model': FULLFT_MODEL, **FULLFT_KW, 'seeds': SEEDS}})
(RUN / 'run_environment.json').write_text(json.dumps(env, indent=2))
DATA_DIR = Path('/content/CondaQA')
if not DATA_DIR.exists():
    subprocess.run(['git', 'clone', '--quiet', CONDAQA_REPO, str(DATA_DIR)], check=True)
    subprocess.run(['git', '-C', str(DATA_DIR), 'checkout', '--quiet', CONDAQA_COMMIT], check=True)
hashes = condaqa.file_hashes(DATA_DIR / 'data')
assert all(CONDAQA_SHA256[k] == v for k, v in hashes.items()), f'CondaQA files differ: {hashes}'
data = condaqa.load_condaqa(DATA_DIR / 'data')
TRAIN, DEV, TEST = data['train'], data['dev'], data['test']
C1_TRAIN, C1_DEV, C1_TEST = ((TRAIN[:LIMIT_ROWS], DEV[:LIMIT_ROWS], TEST[:LIMIT_ROWS]) if LIMIT_ROWS else (TRAIN, DEV, TEST))
print(RUN)
""")

code("""# ---- (1) Proposition-1 re-check in float64 on the archived E0 states ---------------------------------
if RUN_PROP1:
    OLD = ROOT / 'runs' / OLD_RUN_ID / 'runs' / 'E0-hash'
    found = sorted(OLD.glob('*/seed*/module_state.pt'))
    print(f'archived E0 module states found under {OLD}: {len(found)}')
    assert found, 'old run directory not found on Drive: check OLD_RUN_ID / PROJECT_ROOT'
    enc0 = encoders.HashEncoder(dim=384); store0 = encoders.EncodedStore(enc0)
    cands = [st.states for st in store0.get_or_encode(condaqa.render_candidates())]
    test_by_id = {r.sample_id: r for r in TEST}
    rows_p1 = [x for x in TEST if x.edit_id == 0 and x.eligible and x.cue]
    out = {}
    for agg in ('max', 'lse'):
        for seed in SEEDS:
            sd = OLD / agg / f'seed{seed}'
            if not (sd / 'module_state.pt').exists():
                print('missing', sd); continue
            t0 = time.time()
            m32 = prop1.load_module(str(sd / 'module_state.pt'), 384, agg, dtype=torch.float32)
            m64 = prop1.load_module(str(sd / 'module_state.pt'), 384, agg, dtype=torch.float64)
            # parity: recompute the archived test scores (float32 CPU) for 500 archived records
            recs = [json.loads(l) for l in open(sd / 'test_predictions.jsonl')][:500]
            dmax = 0.0
            for rec in recs:
                r = test_by_id[rec['sample_id']]
                ctx = store0.get_or_encode([condaqa.context_text(r.question, r.passage)])[0].states
                s = prop1.option_scores(m32, ctx, cands, dtype=torch.float32)
                dmax = max(dmax, max(abs(a - b) for a, b in zip(s, rec['scores'])))
            pairs, n_pass = [], 0
            for r in rows_p1:
                edited, n = condaqa.delete_cue(r.passage, r.cue)
                if n == 0:
                    continue
                n_pass += 1
                full = store0.get_or_encode([condaqa.context_text(r.question, r.passage)])[0].states
                dele = store0.get_or_encode([condaqa.context_text(r.question, edited)])[0].states
                pairs.append((prop1.option_scores(m64, full, cands), prop1.option_scores(m64, dele, cands)))
            res = {**prop1.violations(pairs, tol=1e-9), 'passages': n_pass, 'parity_max_abs_diff_vs_archived_scores': dmax,
                   'parity_records_checked': len(recs)}
            out[f'{agg}/seed{seed}'] = res
            print(f"{agg:4s} seed {seed:5d}  parity {dmax:.1e}  float64 violations {res['violations']}/{res['n_scores']}  "
                  f"max increase {res['max_increase_after_deletion']:.2e}  ({time.time()-t0:.0f}s)", flush=True)
    (RUN / 'proposition1_float64_recheck.json').write_text(json.dumps(out, indent=2))
""")

code("""# ---- (2) C1 re-execution: full fine-tuning in float32 with a NaN guard ----------------------------------
if RUN_C1:
    from cacsa2 import finetune
    from huggingface_hub import HfApi
    sha = None if Path(FULLFT_MODEL).exists() else HfApi().model_info(FULLFT_MODEL).sha
    for seed in SEEDS:
        out_dir = RUN / 'runs' / 'C1-fullft' / (FULLFT_MODEL.split('/')[-1] + '-ft') / f'seed{seed}'
        if (out_dir / 'run_summary.json').exists():
            continue
        print(f'start full fine-tuning seed {seed} ...', flush=True)
        s = finetune.finetune_run(FULLFT_MODEL, C1_TRAIN, C1_DEV, C1_TEST, seed=seed, out_dir=out_dir, revision=sha,
                                  state_dir='/content/fullft_states', **FULLFT_KW)
        print(f"seed {seed}: dev {s['dev']['accuracy']:.2f}  test {s['test']['accuracy']:.2f}  selected epoch {s['selected_epoch']}  "
              f"losses {[round(h['train_loss'], 4) for h in s['history']]}  ({s['train_time_s']:.0f}s)", flush=True)
""")

code("""# ---- metrics for C1 against the label-prior baselines --------------------------------------------------
if RUN_C1:
    runs = analysis.load_runs(RUN / 'runs', split='test')
    rows = analysis.per_run_metrics(runs)
    sm = analysis.seed_mean_table(rows)
    (RUN / 'c1_per_run_metrics.json').write_text(json.dumps(rows, indent=1))
    (RUN / 'c1_seed_means.json').write_text(json.dumps(sm, indent=1))
    recs = next(iter(runs.values()))['records']
    prior = {l: 100.0 * sum(r['label'] == l for r in recs) / len(recs) for l in ('YES', 'NO', "DON'T KNOW")}
    (RUN / 'label_prior_baselines.json').write_text(json.dumps(prior, indent=1))
    print('constant-prediction baselines:', {k: round(v, 2) for k, v in prior.items()})
    print(analysis.markdown_table(rows, ['seed', 'selected_epoch', 'acc', 'acc_original', 'acc_paraphrase', 'acc_scope', 'acc_affirmative',
                                         'consistency', 'flip_paraphrase', 'flip_affirmative']))
""")

code("""# ---- file manifest, zip, download --------------------------------------------------------------------------
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
out = Path(__file__).parent / "CACSA_v11_followup_colab.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(out, len(cells), "cells")
