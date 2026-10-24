#!/usr/bin/env python3
"""Number registry of the manuscript (study version 3.0 and the October 2026 revision).

Every number printed in the manuscript is taken from this registry, and every registry entry is read from a file of an
archived run or build directory (or computed from such files by the code below; the `how` field states the
computation). Outputs (next to this script): registry.json (key -> value, source, how, kind, decimals) and
traceability.csv.

Sources, relative to the data root (CACSA_DATA_ROOT; default: <repository>/outputs)
  main_v3/np_v3_main/            experiment run np_v3_main (7 October 2026, A100), results zip unpacked
  main_v3/followup/np_v3_main/   float64 follow-up for E0 and E3 on NegPairs-Perm (optional; registered when present)
  np_data_v3/negpairs_v3/        NegPairs-Fresh and NegPairs-Perm build (5 October 2026, L4)
  np_data/negpairs_v1/           NegPairs v1
  indep_v3/out/, indep_v3/out_f64/   independent recomputation from the raw predictions
  public_v3/public_systems/      released systems (21 and 22 October 2026, L4)
  finetune_v3/finetune_li/       fine-tuned late-interaction checkpoints (22 October 2026, L4)
  paper2/registry.json           the registry of the v2.0 manuscript (CondaQA v1.1 run, pilot and v2.0 confirmatory
                                 run, NegPairs v1 construction); imported under the prefix `v2.`
The code root (CACSA_CODE_ROOT; default: the repository) holds the `cacsa2` package and `human_check/`.

Rounding rule used by the manuscripts: half up, from the full-precision value, to the number of decimals of the entry
(two for percentages and percentage points unless stated otherwise).
"""
import os
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
_REPO = HERE.parent
# data root that holds the archived inputs (see paper/README.md for the expected layout); override with CACSA_DATA_ROOT
S = Path(os.environ.get('CACSA_DATA_ROOT', _REPO / 'outputs' if (_REPO / 'outputs').is_dir() else _REPO))
# code root with the cacsa2 package and human_check/; override with CACSA_CODE_ROOT
CODE = Path(os.environ.get('CACSA_CODE_ROOT', _REPO if (_REPO / 'cacsa2').is_dir() else _REPO / 'cacsa2_project'))
sys.path.insert(0, str(CODE))
MAIN = S / 'main_v3/np_v3_main'
FU = S / 'main_v3/followup/np_v3_main'
BUILD = S / 'np_data_v3/negpairs_v3'
V1 = S / 'np_data/negpairs_v1'
INDEP = S / 'indep_v3/out/recompute_report.json'
OLD = S / 'paper2/registry.json'
OUT = HERE

REG = {}


def rel(p):
    """Source path as recorded in the registry: relative to the data root, or 'cacsa2_project/...' for code files,
    so that the registry is identical wherever the archive and the repository are unpacked."""
    if isinstance(p, (list, tuple)):
        return '; '.join(rel(x) for x in p)
    p = Path(p)
    try:
        return str(p.relative_to(S))
    except ValueError:
        pass
    try:
        return str(Path('cacsa2_project') / p.relative_to(CODE))
    except ValueError:
        return str(p)


def put(key, value, src, how, kind='pct', dec=2):
    if key in REG:
        raise KeyError(f'duplicate registry key {key}')
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ValueError(f'{key}: non-finite value')
    REG[key] = {'value': value, 'src': src if isinstance(src, str) else rel(src), 'how': how, 'kind': kind, 'dec': dec}


def jload(p):
    with open(p, encoding='utf-8') as fh:
        return json.load(fh)


def jlines(p):
    with open(p, encoding='utf-8') as fh:
        return [json.loads(l) for l in fh if l.strip()]


ENC = {'E0-hash': 'E0', 'E1-minilm': 'E1', 'E2-deberta': 'E2', 'E3-glove': 'E3', 'E4-mpnet': 'E4', 'E5-bge': 'E5',
       'E6-colbertv2': 'E6', 'C1-mcft': 'C1'}
AGG = {'max': 'A1', 'mean': 'A2', 'lse': 'A3', 'signed_max': 'A4', 'signed_max_gate': 'A5',
       'signed_max_sentence': 'A6', 'signed_max_oracle': 'D1', 'head_only': 'C0', 'option_only': 'C2',
       'deberta-v3-base-mc-ft': 'C1', 'm0': 'M0', 'm1': 'M1', 'm0_native': 'M0n'}
SEEDS = [13, 42, 71, 7595, 1676, 101, 202, 303, 404, 505]      # manifest order
ENCS7 = ['E0', 'E3', 'E1', 'E2', 'E4', 'E5', 'E6']
CONTEXTUAL = ['E1', 'E2', 'E4', 'E5', 'E6']
STATIC = ['E0', 'E3']
TRAINED = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'D1', 'M1', 'C0', 'C2']
T95_9 = float(stats.t.ppf(0.975, 9))


def metric_kind(m):
    last = m.split('.')[-1]
    if last in ('n', 'n_N', 'n_S', 'n_scenarios', 'n_seeds'):
        return 'int', 0
    if last.startswith('N_error_margin') or last.startswith('N_error_frac'):
        return 'float', 3
    if 'SS' in last or 'gap' in last:
        return 'pp', 2
    return 'pct', 2


def t_interval(vals, conf=0.95):
    n = len(vals)
    m = statistics.fmean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    q = float(stats.t.ppf(0.5 + conf / 2, n - 1))
    h = q * sd / math.sqrt(n)
    return m, sd, m - h, m + h


# =============================================================================================== seed means (3 sets)
SETS = {'v1': 'v1test', 'fresh': 'fresh', 'perm': 'perm'}
SM = {}
for short, name in SETS.items():
    p = MAIN / f'seed_means_{name}.json'
    for r in jload(p):
        e, a = ENC[r['encoder']], AGG[r['aggregator']]
        SM[(short, e, a)] = r
        put(f'sm.{short}.{e}.{a}.n_seeds', r['n_seeds'], p, f'[{r["encoder"]},{r["aggregator"]}].n_seeds', 'int', 0)
        for k, v in r.items():
            if k in ('encoder', 'aggregator', 'n_seeds') or v is None:
                continue
            base = k[:-3] if k.endswith('_sd') else k
            kind, dec = metric_kind(base)
            if k.endswith('_sd'):
                put(f'sm.{short}.{e}.{a}.{base}.sd', v, p, f'[{r["encoder"]},{r["aggregator"]}].{k}', 'pp' if kind in ('pct', 'pp') else kind, dec)
            else:
                put(f'sm.{short}.{e}.{a}.{k}', v, p, f'[{r["encoder"]},{r["aggregator"]}].{k}', kind, dec)

# =============================================================================================== per-run values (selected metrics), seed intervals, extremes
PR_METRICS = {
    'v1': ['all.SS', 'seen.SS', 'unseen.SS', 'all.acc', 'all.acc_OP', 'seen.flip_O_N', 'seen.flip_O_S', 'unseen.flip_O_N',
           'unseen.flip_O_S', 'S_same_scen.flip_O_S', 'samecue_not.flip_O_N', 'samecue_not.flip_O_S', 'all.flip_O_P'],
    'fresh': ['all.SS', 'seen.SS', 'unseen.SS', 'all.acc', 'all.acc_OP', 'seen.flip_O_N', 'seen.flip_O_S', 'unseen.flip_O_N',
              'unseen.flip_O_S', 'S_same_scen.flip_O_S', 'samecue_not.flip_O_N', 'samecue_not.flip_O_S', 'all.flip_O_P',
              'prior_incongruent.acc_OP', 'family.affixal.flip_O_N', 'family.lexical.flip_O_N'],
    'perm': ['all.SS_bal', 'seen.SS_bal', 'unseen.SS_bal', 'seen.first.SS', 'seen.second.SS', 'seen.pooled.SS',
             'seen.pooled.joint_NS_acc', 'all.pooled.SS', 'all.pooled.joint_NS_acc', 'seen.first.flip_O_N', 'seen.first.flip_O_S',
             'seen.second.flip_O_N', 'seen.second.flip_O_S', 'seen.pooled.flip_O_N', 'seen.pooled.flip_O_S',
             'all.first.acc_O', 'all.second.acc_O', 'all.first.flip_O_S', 'all.second.flip_O_S', 'all.first.SS', 'all.second.SS',
             'family.syntactic.SS_bal', 'family.quantifier.SS_bal', 'family.affixal.SS_bal', 'family.lexical.SS_bal'],
}
PR = defaultdict(dict)
for short, name in SETS.items():
    p = MAIN / f'per_run_{name}.json'
    rows = jload(p)
    by_cond = defaultdict(list)
    for r in rows:
        e, a = ENC[r['encoder']], AGG[r['aggregator']]
        PR[(short, e, a)][r['seed']] = r
        by_cond[(e, a)].append(r)
        for m in PR_METRICS[short]:
            v = r.get(m)
            if v is None:
                continue
            kind, dec = metric_kind(m)
            put(f'pr.{short}.{e}.{a}.s{r["seed"]}.{m}', v, p, f'[{r["encoder"]},{r["aggregator"]},seed={r["seed"]}].{m}', kind, dec)
    for (e, a), rs in by_cond.items():
        if len(rs) < 2:
            continue
        for m in PR_METRICS[short]:
            vals = [r[m] for r in rs if r.get(m) is not None]
            if len(vals) != len(rs):
                continue
            kind, dec = metric_kind(m)
            mean, sd, lo, hi = t_interval(vals)
            put(f'ci.{short}.{e}.{a}.{m}.low', lo, p, f'seed-level 95% t interval (n={len(vals)}, df={len(vals)-1}) of [{e},{a}].{m}: low', kind, dec)
            put(f'ci.{short}.{e}.{a}.{m}.high', hi, p, f'seed-level 95% t interval (n={len(vals)}, df={len(vals)-1}) of [{e},{a}].{m}: high', kind, dec)
            put(f'ext.{short}.{e}.{a}.{m}.min', min(vals), p, f'min over seeds of [{e},{a}].{m}', kind, dec)
            put(f'ext.{short}.{e}.{a}.{m}.max', max(vals), p, f'max over seeds of [{e},{a}].{m}', kind, dec)
            put(f'ext.{short}.{e}.{a}.{m}.n_pos', sum(1 for v in vals if v > 0), p, f'seeds with [{e},{a}].{m} > 0', 'int', 0)

# ranges over conditions used in the text
def put_range(key, keys, how_text, kind='pp', dec=2):
    vals = [REG[k]['value'] for k in keys]
    put(key + '.min', min(vals), 'registry', how_text + ': min', kind, dec)
    put(key + '.max', max(vals), 'registry', how_text + ': max', kind, dec)


put_range('rng.perm.A1.joint_NS_acc.contextual', [f'sm.perm.{e}.A1.seen.pooled.joint_NS_acc' for e in CONTEXTUAL],
          'sm.perm.{E1,E2,E4,E5,E6}.A1.seen.pooled.joint_NS_acc', 'pct')
put_range('rng.perm.A6.seen_SS_bal.all7', [f'sm.perm.{e}.A6.seen.SS_bal' for e in ENCS7], 'sm.perm.{all encoders}.A6.seen.SS_bal')
put_range('rng.perm.A4.seen_SS_bal.contextual', [f'sm.perm.{e}.A4.seen.SS_bal' for e in CONTEXTUAL], 'sm.perm.{contextual}.A4.seen.SS_bal')
put_range('rng.perm.D1.seen_SS_bal.contextual', [f'sm.perm.{e}.D1.seen.SS_bal' for e in CONTEXTUAL], 'sm.perm.{contextual}.D1.seen.SS_bal')
put_range('rng.fresh.A5mA1.unseen', [], 'placeholder', 'pp') if False else None
put_range('rng.fresh.M0.SS.allsets', [f'sm.{s}.{e}.M0.{g}.SS' for s in ('v1', 'fresh') for e in ENCS7 for g in ('seen', 'unseen')]
          + [f'sm.perm.{e}.M0.seen.SS_bal' for e in ENCS7], 'M0 SS on v1 test and Fresh (seen, unseen) and SS_bal on Perm (seen), all encoders')
put_range('rng.fresh.C0.absSS', [], 'placeholder') if False else None
vals = [abs(REG[f'sm.fresh.{e}.C0.{g}.SS']['value']) for e in ENCS7 for g in ('seen', 'unseen')]
put('ctrl.C0.fresh.max_abs_SS', max(vals), 'registry', 'max |sm.fresh.{e}.C0.{seen,unseen}.SS| over encoders', 'pp', 2)
vals = [abs(REG[f'sm.perm.{e}.C0.seen.SS_bal']['value']) for e in ENCS7]
put('ctrl.C0.perm.max_abs_SS_bal', max(vals), 'registry', 'max |sm.perm.{e}.C0.seen.SS_bal| over encoders', 'pp', 2)
put('ctrl.C2.fresh.max_flip', max(REG[f'sm.fresh.{e}.C2.all.flip_O_{x}']['value'] for e in ENCS7 for x in 'NSP'), 'registry',
    'max sm.fresh.{e}.C2.all.flip_O_{N,S,P}', 'pct', 2)
put('ctrl.C2.perm.max_flip', max(REG[f'sm.perm.{e}.C2.all.pooled.flip_O_{x}']['value'] for e in ENCS7 for x in 'NSP'), 'registry',
    'max sm.perm.{e}.C2.all.pooled.flip_O_{N,S,P}', 'pct', 2)
put_range('rng.fresh.module.acc', [f'sm.fresh.{e}.{a}.all.acc' for e in ENCS7 for a in ('A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'D1', 'M1')],
          'sm.fresh.{e}.{A1..A6,D1,M1}.all.acc', 'pct')
put_range('rng.fresh.C2.acc_OP', [f'sm.fresh.{e}.C2.all.acc_OP' for e in ENCS7], 'sm.fresh.{e}.C2.all.acc_OP', 'pct')
put_range('rng.fresh.A1.acc_OP', [f'sm.fresh.{e}.A1.all.acc_OP' for e in ENCS7], 'sm.fresh.{e}.A1.all.acc_OP', 'pct')
put_range('rng.fresh.A1.incong_acc_OP', [f'sm.fresh.{e}.A1.prior_incongruent.acc_OP' for e in ENCS7], 'sm.fresh.{e}.A1.prior_incongruent.acc_OP', 'pct')

# =============================================================================================== run summaries (parameters, selection, time)
RUNS = MAIN / 'runs'
time_total = 0.0
time_grid = 0.0
n_grid = 0
params = {}
for p in sorted(RUNS.glob('*/*/seed*/run_summary.json')):
    r = jload(p)
    e = ENC[p.parts[-4]]
    a = AGG[p.parts[-3]]
    seed = int(p.parts[-2][4:])
    if not r.get('trained', True):
        if 'parameter_counts' in r:
            params[(e, a)] = r['parameter_counts']['total']
        continue
    time_total += r['train_time_s']
    if a in ('A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'D1', 'M1'):
        time_grid += r['train_time_s']
        n_grid += 1
    params[(e, a)] = r['parameter_counts']['total']
    dev_acc = r['dev']['accuracy'] if isinstance(r.get('dev'), dict) and 'accuracy' in r['dev'] else max(h['dev_acc'] for h in r['history'])
    put(f'train.{e}.{a}.s{seed}.dev_acc', dev_acc, p, 'dev.accuracy (selected checkpoint)', 'pct', 2)
    put(f'train.{e}.{a}.s{seed}.selected_epoch', r['selected_epoch'], p, 'selected_epoch', 'int', 0)
    put(f'train.{e}.{a}.s{seed}.epochs_run', r['epochs_run'], p, 'epochs_run', 'int', 0)
    put(f'train.{e}.{a}.s{seed}.time_s', r['train_time_s'], p, 'train_time_s', 'float', 1)
for (e, a), v in params.items():
    put(f'param.{e}.{a}', v, RUNS / 'run_summary.json files', f'parameter_counts.total of [{e},{a}] (identical across seeds)', 'int', 0)
put('train.time_total_h', time_total / 3600, RUNS, 'sum of train_time_s over all 718 run_summary.json files / 3600', 'float', 2)
put('train.time_grid_h', time_grid / 3600, RUNS, 'sum of train_time_s over the 560 grid runs / 3600', 'float', 2)
put('train.grid_mean_s', time_grid / n_grid, RUNS, 'mean train_time_s over the 560 grid runs', 'float', 0)
put('train.n_runs', len(list(RUNS.glob('*/*/seed*/run_summary.json'))), RUNS, 'count of run_summary.json files (718: 560 grid, 140 controls, 10 C1, 7 M0, 1 M0-native)', 'int', 0)
put('train.n_trained_runs', sum(1 for p in RUNS.glob('*/*/seed*/run_summary.json') if jload(p).get('trained', True)), RUNS, 'run_summary.json files of trained runs', 'int', 0)
put('train.n_grid_runs', n_grid, RUNS, 'runs with a trained condition A1-A6, D1, M1', 'int', 0)
# dev accuracy extremes per condition (used for the E0 bimodality discussion)
for (short, e, a), d in list(PR.items()):
    pass
devacc = defaultdict(list)
for k, v in REG.items():
    m = re.match(r'train\.(E\d|C1)\.(\w+)\.s(\d+)\.dev_acc$', k)
    if m:
        devacc[(m.group(1), m.group(2))].append((int(m.group(3)), v['value']))
for (e, a), vals in devacc.items():
    vs = [v for _, v in vals]
    put(f'train.{e}.{a}.dev_acc.min', min(vs), RUNS, f'min over seeds of dev.accuracy of [{e},{a}]', 'pct', 2)
    put(f'train.{e}.{a}.dev_acc.max', max(vs), RUNS, f'max over seeds of dev.accuracy of [{e},{a}]', 'pct', 2)
    put(f'train.{e}.{a}.dev_acc.mean', statistics.fmean(vs), RUNS, f'mean over seeds of dev.accuracy of [{e},{a}]', 'pct', 2)

# =============================================================================================== gate
G = jload(MAIN / 'gate.json')
put('G.threshold', int(G['threshold']), MAIN / 'gate.json', 'threshold (integer-valued)', 'int', 0)
put('G.k_R2_R3', G['k_R2_R3'], MAIN / 'gate.json', 'k_R2_R3', 'int', 0)
put('G.k_R8', G['k_R8'], MAIN / 'gate.json', 'k_R8', 'int', 0)
put('G.n_passed', len(G['passed']), MAIN / 'gate.json', 'len(passed)', 'int', 0)
put('G.computed_utc', G['computed_utc'][:19].replace('T', ' '), MAIN / 'gate.json', 'computed_utc', 'text', 0)
for enc, g in G['encoders'].items():
    e = ENC[enc]
    put(f'G.{e}.mean', g['A1_seed_mean_dev_SS'], MAIN / 'gate.json', f'encoders.{enc}.A1_seed_mean_dev_SS', 'pp', 2)
    put(f'G.{e}.passed', bool(g['passed']), MAIN / 'gate.json', f'encoders.{enc}.passed', 'bool', 0)
    for sd, v in zip(g['seeds'], g['A1_dev_SS_by_seed']):
        put(f'G.{e}.s{sd}', v, MAIN / 'gate.json', f'encoders.{enc}.A1_dev_SS_by_seed[seed {sd}]', 'pp', 2)
    put(f'G.{e}.min', min(g['A1_dev_SS_by_seed']), MAIN / 'gate.json', f'min encoders.{enc}.A1_dev_SS_by_seed', 'pp', 2)
    put(f'G.{e}.max', max(g['A1_dev_SS_by_seed']), MAIN / 'gate.json', f'max encoders.{enc}.A1_dev_SS_by_seed', 'pp', 2)
    put(f'G.{e}.n_pos', sum(1 for v in g['A1_dev_SS_by_seed'] if v > 0), MAIN / 'gate.json', f'seeds with dev SS > 0 ({enc})', 'int', 0)
    put(f'G.{e}.n_ge_thr', sum(1 for v in g['A1_dev_SS_by_seed'] if v >= G['threshold']), MAIN / 'gate.json', f'seeds with dev SS >= threshold ({enc})', 'int', 0)
    put(f'G.{e}.n_le_zero', sum(1 for v in g['A1_dev_SS_by_seed'] if v <= 0), MAIN / 'gate.json', f'seeds with dev SS <= 0 ({enc})', 'int', 0)
# seed-dependence of the gate: seeds at or above the threshold against seeds below it
for enc, g in G['encoders'].items():
    e = ENC[enc]
    above = [v for v in g['A1_dev_SS_by_seed'] if v >= G['threshold']]
    below = [v for v in g['A1_dev_SS_by_seed'] if v < G['threshold']]
    put(f'G.{e}.above.n', len(above), MAIN / 'gate.json', f'seeds with dev SS >= threshold ({enc})', 'int', 0)
    put(f'G.{e}.below.n', len(below), MAIN / 'gate.json', f'seeds with dev SS < threshold ({enc})', 'int', 0)
    if above:
        put(f'G.{e}.above.min', min(above), MAIN / 'gate.json', f'min dev SS among seeds at or above the threshold ({enc})', 'pp', 2)
        put(f'G.{e}.above.max', max(above), MAIN / 'gate.json', f'max dev SS among seeds at or above the threshold ({enc})', 'pp', 2)
    if below:
        put(f'G.{e}.below.min', min(below), MAIN / 'gate.json', f'min dev SS among seeds below the threshold ({enc})', 'pp', 2)
        put(f'G.{e}.below.max', max(below), MAIN / 'gate.json', f'max dev SS among seeds below the threshold ({enc})', 'pp', 2)

# =============================================================================================== rules
RULES = jload(MAIN / 'rules.json')
for k, v in RULES['thresholds'].items():
    kind = 'sci' if k == 'exact_tolerance' else ('float' if k == 'family_alpha' else 'int')
    put(f'thr.{k}', int(v) if kind == 'int' else v, MAIN / 'rules.json', f'thresholds.{k}' + (' (integer-valued threshold)' if kind == 'int' else ''), kind, 0 if kind == 'int' else 2)
for rule in ('R2', 'R3', 'R8'):
    for enc, r in RULES[rule].items():
        e = ENC[enc]
        src = MAIN / 'rules.json'
        put(f'{rule}.{e}.evaluated', bool(r['evaluated']), src, f'{rule}.{enc}.evaluated', 'bool', 0)
        put(f'{rule}.{e}.met', bool(r['met']), src, f'{rule}.{enc}.met', 'bool', 0)
        ti = r['t_interval']
        put(f'{rule}.{e}.mean', ti['mean'], src, f'{rule}.{enc}.t_interval.mean', 'pp', 2)
        put(f'{rule}.{e}.sd', ti['sd'], src, f'{rule}.{enc}.t_interval.sd', 'pp', 2)
        put(f'{rule}.{e}.low', ti['low'], src, f'{rule}.{enc}.t_interval.low', 'pp', 2)
        put(f'{rule}.{e}.high', ti['high'], src, f'{rule}.{enc}.t_interval.high', 'pp', 2)
        put(f'{rule}.{e}.pos', ti['positive_count'], src, f'{rule}.{enc}.t_interval.positive_count', 'int', 0)
        put(f'{rule}.{e}.n', ti['n'], src, f'{rule}.{enc}.t_interval.n', 'int', 0)
        for sd, v in zip(r['seeds'], r['values']):
            put(f'{rule}.{e}.s{sd}', v, src, f'{rule}.{enc}.values[seed {sd}]', 'pp', 2)
        put(f'{rule}.{e}.vmin', min(r['values']), src, f'min {rule}.{enc}.values', 'pp', 2)
        put(f'{rule}.{e}.vmax', max(r['values']), src, f'max {rule}.{enc}.values', 'pp', 2)
        if r['cluster_interval']:
            ci = r['cluster_interval']
            put(f'{rule}.{e}.cl_low', ci['low'], src, f'{rule}.{enc}.cluster_interval.low', 'pp', 2)
            put(f'{rule}.{e}.cl_high', ci['high'], src, f'{rule}.{enc}.cluster_interval.high', 'pp', 2)
            put(f'{rule}.{e}.cl_point', ci['point'], src, f'{rule}.{enc}.cluster_interval.point', 'pp', 2)
        put(f'{rule}.{e}.conf', r['conf'] * 100, src, f'{rule}.{enc}.conf x 100', 'pct', 2)
        put(f'{rule}.{e}.k', r['k'], src, f'{rule}.{enc}.k', 'int', 0)
        put(f'{rule}.{e}.threshold', int(r['threshold']), src, f'{rule}.{enc}.threshold (integer-valued)', 'int', 0)
        for c, v in r['conditions'].items():
            put(f'{rule}.{e}.cond_{c[0]}', bool(v), src, f'{rule}.{enc}.conditions.{c}', 'bool', 0)
        if rule == 'R2':
            put(f'{rule}.{e}.n_scen', r['n_unseen_fresh_scenarios'], src, f'{rule}.{enc}.n_unseen_fresh_scenarios', 'int', 0)
        elif rule == 'R3':
            put(f'{rule}.{e}.n_scen', r['n_S_same_scenarios'], src, f'{rule}.{enc}.n_S_same_scenarios', 'int', 0)
        else:
            put(f'{rule}.{e}.n_first', r['n_seen_perm_by_order']['first'], src, f'{rule}.{enc}.n_seen_perm_by_order.first', 'int', 0)
            put(f'{rule}.{e}.n_second', r['n_seen_perm_by_order']['second'], src, f'{rule}.{enc}.n_seen_perm_by_order.second', 'int', 0)
    put(f'{rule}.n_met', sum(1 for r in RULES[rule].values() if r['met']), MAIN / 'rules.json', f'count of {rule}.*.met', 'int', 0)
    put(f'{rule}.n_evaluated', sum(1 for r in RULES[rule].values() if r['evaluated']), MAIN / 'rules.json', f'count of {rule}.*.evaluated', 'int', 0)
for enc, r in RULES['R1_descriptive'].items():
    e = ENC[enc]
    src = MAIN / 'rules.json'
    put(f'R1.{e}.mean', r['seed_mean'], src, f'R1_descriptive.{enc}.seed_mean', 'pp', 2)
    put(f'R1.{e}.v2_threshold', int(r['v2_threshold']), src, f'R1_descriptive.{enc}.v2_threshold (integer-valued)', 'int', 0)
    put(f'R1.{e}.cl_low', r['cluster_interval']['low'], src, f'R1_descriptive.{enc}.cluster_interval.low', 'pp', 2)
    put(f'R1.{e}.cl_high', r['cluster_interval']['high'], src, f'R1_descriptive.{enc}.cluster_interval.high', 'pp', 2)
    put(f'R1.{e}.min', min(r['A1_unseen_SS_fresh_by_seed']), src, f'min R1_descriptive.{enc}.A1_unseen_SS_fresh_by_seed', 'pp', 2)
    put(f'R1.{e}.max', max(r['A1_unseen_SS_fresh_by_seed']), src, f'max R1_descriptive.{enc}.A1_unseen_SS_fresh_by_seed', 'pp', 2)
    put(f'R1.{e}.n_boot', r['cluster_interval']['n_boot'], src, f'R1_descriptive.{enc}.cluster_interval.n_boot', 'int', 0)
for rule in ('R5', 'R6', 'R7'):
    for k, v in RULES[rule].items():
        put(f'{rule}.{k}', bool(v) if isinstance(v, bool) else v, MAIN / 'rules.json', f'{rule}.{k}', 'bool' if isinstance(v, bool) else 'int', 0)
ss = RULES['subset_sizes']
put('subset.fresh_unseen', ss['fresh_unseen_scenarios'], MAIN / 'rules.json', 'subset_sizes.fresh_unseen_scenarios', 'int', 0)
put('subset.fresh_S_same', ss['fresh_S_same_scenarios'], MAIN / 'rules.json', 'subset_sizes.fresh_S_same_scenarios', 'int', 0)
put('subset.perm_seen_first', ss['perm_seen_by_order']['E1-minilm']['first'], MAIN / 'rules.json', 'subset_sizes.perm_seen_by_order.E1-minilm.first (identical for every encoder)', 'int', 0)
put('subset.perm_seen_second', ss['perm_seen_by_order']['E1-minilm']['second'], MAIN / 'rules.json', 'subset_sizes.perm_seen_by_order.E1-minilm.second (identical for every encoder)', 'int', 0)

# =============================================================================================== exact rules in detail
R5 = jload(MAIN / 'exact_R5.json')
for enc in ('E0-hash', 'E3-glove'):
    e = ENC[enc]
    for short, name in SETS.items():
        put(f'R5.{e}.{short}.pairs', R5[f'{enc}/{name}/pairs'], MAIN / 'exact_R5.json', f'{enc}/{name}/pairs', 'int', 0)
    viol = 0
    n_checks = 0
    largest = 0.0
    for k, v in R5.items():
        if k.startswith(enc + '/') and isinstance(v, dict) and 'violations' in v:
            viol += v['violations']
            n_checks += 1
            largest = min(largest, v['largest_decrease'])
    put(f'R5.{e}.checks', n_checks, MAIN / 'exact_R5.json', f'count of check entries of {enc}', 'int', 0)
    put(f'R5.{e}.violations', viol, MAIN / 'exact_R5.json', f'sum of violations over check entries of {enc}', 'int', 0)
    put(f'R5.{e}.largest_decrease', largest, MAIN / 'exact_R5.json', f'min of largest_decrease over check entries of {enc}', 'sci', 2)
put('R5.pairs_total_per_encoder', sum(R5[f'E0-hash/{n}/pairs'] for n in SETS.values()), MAIN / 'exact_R5.json', 'sum of E0-hash/*/pairs', 'int', 0)
put('R5.summary_text', json.dumps(R5['summary'])[:200], MAIN / 'exact_R5.json', 'summary (string)', 'text', 0)
n_scores_r5 = sum(v['n_scores'] for k, v in R5.items() if isinstance(v, dict) and 'n_scores' in v)
put('R5.n_scores_total', n_scores_r5, MAIN / 'exact_R5.json', 'sum of n_scores over all check entries', 'int', 0)

R6 = jload(MAIN / 'exact_R6.json')
for enc in ('E0-hash', 'E3-glove'):
    e = ENC[enc]
    put(f'R6.{e}.A6_equal_marked', R6[f'{enc}/signed_max_sentence/equal_marked_multisets'], MAIN / 'exact_R6.json', f'{enc}/signed_max_sentence/equal_marked_multisets', 'int', 0)
    d = R6[f'{enc}/gpu_vs_float64_decision_disagreements']
    put(f'R6.{e}.gpu_vs_f64_disagreements', d['n'], MAIN / 'exact_R6.json', f'{enc}/gpu_vs_float64_decision_disagreements.n (trained unmasked conditions A1, A2, A3, A5, M1, C0 x 10 seeds)', 'int', 0)
    n_checks = 0
    sv = dd = nt = 0
    for k, v in R6.items():
        if k.startswith(enc + '/') and isinstance(v, dict) and 'score_violations' in v:
            n_checks += 1
            sv += v['score_violations']
            dd += v['decision_disagreements']
            nt += v['near_ties']
    put(f'R6.{e}.checks', n_checks, MAIN / 'exact_R6.json', f'count of check entries of {enc}', 'int', 0)
    put(f'R6.{e}.score_violations', sv, MAIN / 'exact_R6.json', f'sum score_violations ({enc})', 'int', 0)
    put(f'R6.{e}.decision_disagreements', dd, MAIN / 'exact_R6.json', f'sum decision_disagreements ({enc})', 'int', 0)
    put(f'R6.{e}.near_ties', nt, MAIN / 'exact_R6.json', f'sum near_ties ({enc})', 'int', 0)
put('R6.pairs', R6['E0-hash/max/seed13']['pairs'], MAIN / 'exact_R6.json', 'E0-hash/max/seed13.pairs (every Perm scenario)', 'int', 0)
put('R6.n_item_predictions_per_encoder', R6['E0-hash/max/seed13']['pairs'] * 2 * 5 * 10, MAIN / 'exact_R6.json',
    '274 scenarios x 2 arms (N, S) x 5 trained unmasked conditions compared with the GPU in the notebook (A1, A2, A3, A5, M1) x 10 seeds', 'int', 0)
put('R6.A6_pairs', R6['E0-hash/signed_max_sentence/seed13']['pairs'], MAIN / 'exact_R6.json', 'E0-hash/signed_max_sentence/seed13.pairs (Perm scenarios with equal marked multisets)', 'int', 0)
R7 = jload(MAIN / 'exact_R7.json')
for enc in ('E0-hash', 'E3-glove'):
    e = ENC[enc]
    n_checks = viol = 0
    pa = []
    for k, v in R7.items():
        if k.startswith(enc + '/') and isinstance(v, dict) and 'violations' in v:
            n_checks += 1
            viol += v['violations']
            pa.append(v['paired_accuracy_margin'])
    put(f'R7.{e}.checks', n_checks, MAIN / 'exact_R7.json', f'count of check entries of {enc}', 'int', 0)
    put(f'R7.{e}.violations', viol, MAIN / 'exact_R7.json', f'sum violations ({enc})', 'int', 0)
    put(f'R7.{e}.paired_acc_nested_max', max(pa), MAIN / 'exact_R7.json', f'max paired_accuracy_margin on the nested subset ({enc})', 'pct', 2)

# pilot exact checks
PE = jload(MAIN / 'pilot_exact_checks.json')
for enc, d in PE.items():
    e = ENC[enc]
    src = MAIN / 'pilot_exact_checks.json'
    put(f'pilot.{e}.dev_items', d['dev_items'], src, f'{enc}.dev_items', 'int', 0)
    put(f'pilot.{e}.dev_insertion_pairs', d['dev_insertion_pairs'], src, f'{enc}.dev_insertion_pairs', 'int', 0)
    for cond, v in d['permutation'].items():
        a = AGG[cond]
        put(f'pilot.{e}.perm.{a}.violations', v['violations'], src, f'{enc}.permutation.{cond}.violations', 'int', 0)
        put(f'pilot.{e}.perm.{a}.largest_abs_diff', v['largest_abs_diff'], src, f'{enc}.permutation.{cond}.largest_abs_diff', 'sci', 2)
        put(f'pilot.{e}.perm.{a}.n_scores', v['n_scores'], src, f'{enc}.permutation.{cond}.n_scores', 'int', 0)
    for cond, v in d['insertion'].items():
        a = AGG[cond]
        put(f'pilot.{e}.ins.{a}.violations', v['violations'], src, f'{enc}.insertion.{cond}.violations', 'int', 0)
        put(f'pilot.{e}.ins.{a}.n_scores', v['n_scores'], src, f'{enc}.insertion.{cond}.n_scores', 'int', 0)
        put(f'pilot.{e}.ins.{a}.largest_decrease', v['largest_decrease'], src, f'{enc}.insertion.{cond}.largest_decrease', 'sci', 2)
    put(f'pilot.{e}.passed', bool(d['passed']), src, f'{enc}.passed', 'bool', 0)
put('pilot.perm.largest_abs_diff_any', max(v['largest_abs_diff'] for d in PE.values() for v in d['permutation'].values()), MAIN / 'pilot_exact_checks.json',
    'max largest_abs_diff over encoders and conditions (permutation check)', 'sci', 2)
PRPT = jload(MAIN / 'pilot_report.json')
put('pilot.c1_check_acc_OP', PRPT['c1_check_dev_acc_OP'], MAIN / 'pilot_report.json', 'c1_check_dev_acc_OP', 'pct', 2)
put('pilot.stop_conditions_met', len(PRPT['stop_conditions_met']), MAIN / 'pilot_report.json', 'len(stop_conditions_met)', 'int', 0)
C1C = jload(MAIN / 'c1_check.json')
put('c1check.dev_acc', C1C['dev_accuracy'], MAIN / 'c1_check.json', 'dev_accuracy', 'pct', 2)
put('c1check.acc_OP', C1C['dev_block']['acc_OP'], MAIN / 'c1_check.json', 'dev_block.acc_OP', 'pct', 2)
put('c1check.SS', C1C['dev_block']['SS'], MAIN / 'c1_check.json', 'dev_block.SS', 'pp', 2)
put('c1check.flip_O_N', C1C['dev_block']['flip_O_N'], MAIN / 'c1_check.json', 'dev_block.flip_O_N', 'pct', 2)
put('c1check.flip_O_S', C1C['dev_block']['flip_O_S'], MAIN / 'c1_check.json', 'dev_block.flip_O_S', 'pct', 2)

# learning-rate grid
LR = jload(MAIN / 'lr_grid.json')
put('lr.test_files_parsed_before', bool(LR['test_files_parsed_before']), MAIN / 'lr_grid.json', 'test_files_parsed_before', 'bool', 0)
for enc, d in LR['by_encoder'].items():
    e = ENC[enc]
    put(f'lr.{e}.chosen', d['chosen_lr'], MAIN / 'lr_grid.json', f'by_encoder.{enc}.chosen_lr', 'sci', 2)
    for g in d['grid']:
        tag = {0.0003: '3e-4', 0.001: '1e-3', 0.003: '3e-3'}[g['lr']]
        put(f'lr.{e}.acc.{tag}', g['dev_acc_mean'], MAIN / 'lr_grid.json', f'by_encoder.{enc}.grid[lr={g["lr"]}].dev_acc_mean', 'pct', 2)
    best = max(d['grid'], key=lambda g: g['dev_acc_mean'])
    put(f'lr.{e}.best_acc', best['dev_acc_mean'], MAIN / 'lr_grid.json', f'max dev_acc_mean over the grid of {enc}', 'pct', 2)
put('lr.grid.1', 3e-4, MAIN / 'lr_grid.json', 'grid value 1', 'sci', 2)
put('lr.grid.2', 1e-3, MAIN / 'lr_grid.json', 'grid value 2', 'sci', 2)
put('lr.grid.3', 3e-3, MAIN / 'lr_grid.json', 'grid value 3', 'sci', 2)
put('lr.n_3e3', sum(1 for d in LR['by_encoder'].values() if d['chosen_lr'] == 0.003), MAIN / 'lr_grid.json', 'encoders with chosen_lr 3e-3', 'int', 0)
put('lr.n_1e3', sum(1 for d in LR['by_encoder'].values() if d['chosen_lr'] == 0.001), MAIN / 'lr_grid.json', 'encoders with chosen_lr 1e-3', 'int', 0)

# =============================================================================================== descriptive blocks
DB = jload(MAIN / 'descriptive_blocks.json')
for enc, d in DB.items():
    e = ENC[enc]
    src = MAIN / 'descriptive_blocks.json'
    cl = d['fresh']['A1_unseen_SS_cluster']
    put(f'bridge.{e}.fresh.unseen_SS.cl_low', cl['low'], src, f'{enc}.fresh.A1_unseen_SS_cluster.low', 'pp', 2)
    put(f'bridge.{e}.fresh.unseen_SS.cl_high', cl['high'], src, f'{enc}.fresh.A1_unseen_SS_cluster.high', 'pp', 2)
    for name, c in d['R4_A5'].items():
        tag = name.replace('A5_minus_A1', 'A5mA1').replace('A5_minus_A4', 'A5mA4')
        put(f'R4.{e}.{tag}.mean', c['mean'], src, f'{enc}.R4_A5.{name}.mean', 'pp', 2)
        put(f'R4.{e}.{tag}.low', c['low'], src, f'{enc}.R4_A5.{name}.low', 'pp', 2)
        put(f'R4.{e}.{tag}.high', c['high'], src, f'{enc}.R4_A5.{name}.high', 'pp', 2)
        put(f'R4.{e}.{tag}.pos', c['positive_count'], src, f'{enc}.R4_A5.{name}.positive_count', 'int', 0)
    for name, c in d['D1'].items():
        tag = name.replace('D1_minus_A1', 'D1mA1').replace('D1_minus_A4', 'D1mA4')
        put(f'D1.{e}.{tag}.mean', c['mean'], src, f'{enc}.D1.{name}.mean', 'pp', 2)
        put(f'D1.{e}.{tag}.low', c['low'], src, f'{enc}.D1.{name}.low', 'pp', 2)
        put(f'D1.{e}.{tag}.high', c['high'], src, f'{enc}.D1.{name}.high', 'pp', 2)
        put(f'D1.{e}.{tag}.pos', c['positive_count'], src, f'{enc}.D1.{name}.positive_count', 'int', 0)
    for name, c in d['MaxSim'].items():
        if isinstance(c, dict):
            tag = name.replace('M1_minus_A1', 'M1mA1')
            put(f'MaxSim.{e}.{tag}.mean', c['mean'], src, f'{enc}.MaxSim.{name}.mean', 'pp', 2)
            put(f'MaxSim.{e}.{tag}.low', c['low'], src, f'{enc}.MaxSim.{name}.low', 'pp', 2)
            put(f'MaxSim.{e}.{tag}.high', c['high'], src, f'{enc}.MaxSim.{name}.high', 'pp', 2)
            put(f'MaxSim.{e}.{tag}.pos', c['positive_count'], src, f'{enc}.MaxSim.{name}.positive_count', 'int', 0)
        else:
            put(f'MaxSim.{e}.{name}', c, src, f'{enc}.MaxSim.{name}', 'pp', 2)
    for cond, c in d['perm_signed'].items():
        a = AGG[cond]
        for k, v in c.items():
            put(f'permsigned.{e}.{a}.{k}', v, src, f'{enc}.perm_signed.{cond}.{k}', 'pct' if 'acc' in k else 'pp', 2)
    for cond, c in d['samecue_not_fresh'].items():
        a = AGG[cond]
        put(f'samecue.fresh.{e}.{a}.N', c['flip_O_N'], src, f'{enc}.samecue_not_fresh.{cond}.flip_O_N', 'pct', 2)
        put(f'samecue.fresh.{e}.{a}.S', c['flip_O_S'], src, f'{enc}.samecue_not_fresh.{cond}.flip_O_S', 'pct', 2)
        put(f'samecue.fresh.{e}.{a}.diff', c['flip_O_N'] - c['flip_O_S'], src, f'{enc}.samecue_not_fresh.{cond}: flip_O_N - flip_O_S', 'pp', 2)
    put(f'samecue.fresh.{e}.n_N', d['samecue_not_fresh']['max']['n_N'], src, f'{enc}.samecue_not_fresh.max.n_N', 'int', 0)
    put(f'samecue.fresh.{e}.n_S', d['samecue_not_fresh']['max']['n_S'], src, f'{enc}.samecue_not_fresh.max.n_S', 'int', 0)
# same cue for C1 from the seed means
put('samecue.fresh.C1.C1.N', REG['sm.fresh.C1.C1.samecue_not.flip_O_N']['value'], MAIN / 'seed_means_fresh.json', '[C1-mcft].samecue_not.flip_O_N', 'pct', 2)
put('samecue.fresh.C1.C1.S', REG['sm.fresh.C1.C1.samecue_not.flip_O_S']['value'], MAIN / 'seed_means_fresh.json', '[C1-mcft].samecue_not.flip_O_S', 'pct', 2)
put_range('rng.R4.A5mA1.unseen', [f'R4.{e}.A5mA1_unseen_SS.mean' for e in ENCS7], 'R4.{e}.A5mA1_unseen_SS.mean over encoders')
put_range('rng.R4.A5mA4.unseen', [f'R4.{e}.A5mA4_unseen_SS.mean' for e in ENCS7], 'R4.{e}.A5mA4_unseen_SS.mean over encoders')
put_range('rng.MaxSim.M1mA1.unseen', [f'MaxSim.{e}.M1mA1_unseen_SS_fresh.mean' for e in ENCS7], 'MaxSim.{e}.M1mA1_unseen_SS_fresh.mean over encoders')
put_range('rng.MaxSim.M1mA1.seen', [f'MaxSim.{e}.M1mA1_seen_SS_fresh.mean' for e in ENCS7], 'MaxSim.{e}.M1mA1_seen_SS_fresh.mean over encoders')

# =============================================================================================== NevIR
NV = jload(MAIN / 'nevir_results.json')
src = MAIN / 'nevir_results.json'
put('nevir.instances', NV['instances'], src, 'instances', 'int', 0)
put('nevir.nested', NV['nested'], src, 'nested', 'int', 0)
put('nevir.nested_pct', 100 * NV['nested'] / NV['instances'], src, '100 * nested / instances', 'pct', 2)
put('nevir.equal_bags', NV['equal_bags'], src, 'equal_bags', 'int', 0)
put('nevir.complement', NV['instances'] - NV['nested'], src, 'instances - nested', 'int', 0)
put('nevir.revision', NV['revision'], src, 'revision', 'text', 0)
put('nevir.revision_short', NV['revision'][:8], src, 'revision[:8]', 'text', 0)
for k, v in NV['published_test_values'].items():
    if isinstance(v, (int, float)):
        if k == 'random':
            put(f'nevir.published.{k}', int(v), 'Weller et al. (2024), Table 2 (as recorded in nevir_results.json)', f'published_test_values.{k} (integer-valued)', 'int', 0)
        else:
            put(f'nevir.published.{k}', v, 'Weller et al. (2024), Table 2 (as recorded in nevir_results.json)', f'published_test_values.{k} (printed with one decimal, as published)', 'float', 1)
static_trained_all, static_trained_nested = [], []
for key, d in NV['scorers'].items():
    parts = key.split('/')
    e = ENC[parts[0]]
    a = AGG[parts[1]]
    tag = f'nevir.{e}.{a}' + (f'.s{parts[2][4:]}' if len(parts) > 2 else '')
    for sub in ('all', 'nested', 'complement'):
        b = d[sub]
        put(f'{tag}.{sub}.paired', b['paired_accuracy'], src, f'scorers.{key}.{sub}.paired_accuracy', 'pct', 2)
        put(f'{tag}.{sub}.per_query', b['per_query_accuracy'], src, f'scorers.{key}.{sub}.per_query_accuracy', 'pct', 2)
        put(f'{tag}.{sub}.n', b['n'], src, f'scorers.{key}.{sub}.n', 'int', 0)
        if 'bootstrap' in b:
            put(f'{tag}.{sub}.low', b['bootstrap']['low'], src, f'scorers.{key}.{sub}.bootstrap.low', 'pct', 2)
            put(f'{tag}.{sub}.high', b['bootstrap']['high'], src, f'scorers.{key}.{sub}.bootstrap.high', 'pct', 2)
    if len(parts) > 2:
        static_trained_all.append(d['all']['paired_accuracy'])
        static_trained_nested.append(d['nested']['paired_accuracy'])
put('nevir.static_trained.all.min', min(static_trained_all), src, 'min paired_accuracy (all) over trained E0/E3 scorers', 'pct', 2)
put('nevir.static_trained.all.max', max(static_trained_all), src, 'max paired_accuracy (all) over trained E0/E3 scorers', 'pct', 2)
put('nevir.static_trained.nested.max', max(static_trained_nested), src, 'max paired_accuracy (nested) over trained E0/E3 scorers', 'pct', 2)
put('nevir.static_trained.n', len(static_trained_all), src, 'count of trained E0/E3 scorers on NevIR', 'int', 0)
for e in STATIC:
    for a in ('A1', 'A3', 'M1'):
        vals = [NV['scorers'][f'{ {v: k for k, v in ENC.items()}[e] }/{ {v: k for k, v in AGG.items()}[a] }/seed{s}']['all']['paired_accuracy'] for s in SEEDS]
        put(f'nevir.{e}.{a}.all.paired.min', min(vals), src, f'min over seeds of scorers.{e}/{a}/seed*.all.paired_accuracy', 'pct', 2)
        put(f'nevir.{e}.{a}.all.paired.max', max(vals), src, f'max over seeds of scorers.{e}/{a}/seed*.all.paired_accuracy', 'pct', 2)
        put(f'nevir.{e}.{a}.all.paired.mean', statistics.fmean(vals), src, f'mean over seeds of scorers.{e}/{a}/seed*.all.paired_accuracy', 'pct', 2)
put_range('rng.nevir.M0.contextual.all', [f'nevir.{e}.M0.all.paired' for e in CONTEXTUAL], 'nevir.{contextual}.M0.all.paired', 'pct')
put_range('rng.nevir.M0.contextual.nested', [f'nevir.{e}.M0.nested.paired' for e in CONTEXTUAL], 'nevir.{contextual}.M0.nested.paired', 'pct')
put_range('rng.nevir.M0.contextual.complement', [f'nevir.{e}.M0.complement.paired' for e in CONTEXTUAL], 'nevir.{contextual}.M0.complement.paired', 'pct')
M0N = jload(MAIN / 'm0_native_scores.json')
put('m0n.query_maxlen', M0N['settings']['query_maxlen'], MAIN / 'm0_native_scores.json', 'settings.query_maxlen', 'int', 0)
put('m0n.doc_maxlen', M0N['settings']['doc_maxlen'], MAIN / 'm0_native_scores.json', 'settings.doc_maxlen', 'int', 0)
put('m0n.dim', M0N['settings']['dim'], MAIN / 'm0_native_scores.json', 'settings.dim', 'int', 0)
put('m0n.nevir_truncated_docs', M0N['nevir_truncated_docs'], MAIN / 'm0_native_scores.json', 'nevir_truncated_docs', 'int', 0)
put('nevir.docs_total', 2 * NV['instances'], src, '2 x instances', 'int', 0)

# =============================================================================================== probing
PB = jload(MAIN / 'probing.json')
src = MAIN / 'probing.json'
for enc, d in PB.items():
    e = ENC[enc]
    put(f'probe.{e}.n_words_seen', d['n_words_seen'], src, f'{enc}.n_words_seen', 'int', 0)
    put(f'probe.{e}.n_words_unseen', d['n_words_unseen'], src, f'{enc}.n_words_unseen', 'int', 0)
    put(f'probe.{e}.cv_mean', d['cv_acc_mean'], src, f'{enc}.cv_acc_mean', 'pct', 2)
    put(f'probe.{e}.cv_sd', d['cv_acc_sd'], src, f'{enc}.cv_acc_sd', 'pp', 2)
    put(f'probe.{e}.transfer', d['transfer']['acc'], src, f'{enc}.transfer.acc', 'pct', 2)
    if 'syntactic_only' in d:
        put(f'probe.{e}.syn_cv', d['syntactic_only']['cv_acc_mean'], src, f'{enc}.syntactic_only.cv_acc_mean', 'pct', 2)
        put(f'probe.{e}.syn_sd', d['syntactic_only']['cv_acc_sd'], src, f'{enc}.syntactic_only.cv_acc_sd', 'pp', 2)
        put(f'probe.{e}.syn_n_words', d['syntactic_only']['n_words_seen'], src, f'{enc}.syntactic_only.n_words_seen', 'int', 0)
    if 'neutral_twins' in d:
        put(f'probe.{e}.twins_cv', d['neutral_twins']['cv_acc_mean'], src, f'{enc}.neutral_twins.cv_acc_mean', 'pct', 2)
        put(f'probe.{e}.twins_sd', d['neutral_twins']['cv_acc_sd'], src, f'{enc}.neutral_twins.cv_acc_sd', 'pp', 2)
    if 'selectivity_syntactic' in d:
        put(f'probe.{e}.selectivity', d['selectivity_syntactic'], src, f'{enc}.selectivity_syntactic', 'pp', 2)
    if 'A5_gate_auc_mean' in d:
        put(f'probe.{e}.auc_mean', d['A5_gate_auc_mean'], src, f'{enc}.A5_gate_auc_mean', 'float', 3)
        aucs = list(d['A5_gate_auc_by_seed'].values())
        put(f'probe.{e}.auc_min', min(aucs), src, f'min {enc}.A5_gate_auc_by_seed', 'float', 3)
        put(f'probe.{e}.auc_max', max(aucs), src, f'max {enc}.A5_gate_auc_by_seed', 'float', 3)
put_range('rng.probe.cv.contextual', [f'probe.{e}.cv_mean' for e in CONTEXTUAL], 'probe.{contextual}.cv_mean', 'pct')
put_range('rng.probe.transfer.contextual', [f'probe.{e}.transfer' for e in CONTEXTUAL], 'probe.{contextual}.transfer', 'pct')
put_range('rng.probe.twins.contextual', [f'probe.{e}.twins_cv' for e in CONTEXTUAL], 'probe.{contextual}.twins_cv', 'pct')
put_range('rng.probe.auc', [f'probe.{e}.auc_mean' for e in ENCS7 if f'probe.{e}.auc_mean' in REG], 'probe.{e}.auc_mean over encoders', 'float', 3)
put_range('rng.probe.selectivity.contextual', [f'probe.{e}.selectivity' for e in CONTEXTUAL], 'probe.{contextual}.selectivity', 'pp')

# =============================================================================================== Qwen audit of v1
QA = jload(MAIN / 'qwen_audit_v1.json')
src = MAIN / 'qwen_audit_v1.json'
put('qwen.model', QA['model'], src, 'model', 'text', 0)
put('qwen.revision', QA['revision'], src, 'revision', 'text', 0)
put('qwen.revision_short', QA['revision'][:8], src, 'revision[:8]', 'text', 0)
for k, v in QA['agreement'].items():
    tag = k.replace('/', '.')
    put(f'qwen.v1.{tag}.agree', v['agree'], src, f'agreement.{k}.agree', 'int', 0)
    put(f'qwen.v1.{tag}.n', v['n'], src, f'agreement.{k}.n', 'int', 0)
    put(f'qwen.v1.{tag}.pct', v['pct'], src, f'agreement.{k}.pct', 'pct', 2)
# A1 SS on the Qwen-passing v1 test scenarios (seed means)
PQ = jload(MAIN / 'per_run_v1test_qwen_passing.json')
by = defaultdict(list)
for r in PQ:
    by[(ENC[r['encoder']], AGG[r['aggregator']])].append(r)
for (e, a), rs in by.items():
    if a not in ('A1', 'A4', 'D1', 'C1', 'A2'):
        continue
    for m in ('seen.SS', 'unseen.SS'):
        vals = [r[m] for r in rs if r.get(m) is not None]
        if vals:
            put(f'qwenpass.{e}.{a}.{m}', statistics.fmean(vals), MAIN / 'per_run_v1test_qwen_passing.json', f'mean over seeds of [{e},{a}].{m} on the Qwen-passing v1 test scenarios', 'pp', 2)
n_pass_test = sum(1 for s in QA['scenarios'].values() if s['pass'] and s['split'] == 'test')
put('qwen.v1.test.pass_n', n_pass_test, src, 'count of scenarios with pass and split == test', 'int', 0)
inc = Counter()
for s in QA['scenarios'].values():
    for arm, v in s['verdicts'].items():
        if v == 'inconsistent':
            inc[arm] += 1
put('qwen.v1.inconsistent.N', inc['N'], src, 'count of N verdicts == inconsistent', 'int', 0)
put('qwen.v1.inconsistent.total', sum(inc.values()), src, 'count of inconsistent verdicts over all arms', 'int', 0)
n_wrong_N = sum(1 for s in QA['scenarios'].values() if s['verdicts']['N'] != 'inconsistent' and s['verdicts']['N'] != s['expected']['N'])
put('qwen.v1.N_read_as_O', n_wrong_N, src, 'count of N verdicts that equal the O answer (consistent, wrong)', 'int', 0)

# =============================================================================================== environment, encoders, data audit, lock
ENV = jload(MAIN / 'run_environment.json')[0]
src = MAIN / 'run_environment.json'
put('env.gpu', ENV['gpu'].split(',')[0], src, 'gpu (name)', 'text', 0)
put('env.gpu_mem_mib', int(ENV['gpu'].split(',')[1].strip().split()[0]), src, 'gpu (memory, MiB)', 'int', 0)
put('env.python', ENV['python'].split()[0], src, 'python (version)', 'text', 0)
put('env.torch', ENV['torch'], src, 'torch', 'text', 0)
put('env.cuda', ENV['cuda_version'], src, 'cuda_version', 'text', 0)
for k, v in ENV['packages'].items():
    put(f'env.{k}', v, src, f'packages.{k}', 'text', 0)
put('env.cacsa2', ENV['cacsa2_version'], src, 'cacsa2_version', 'text', 0)
put('env.run_name', ENV['run_name'], src, 'run_name', 'text', 0)
put('env.start_utc', ENV['timestamp_utc'][:16].replace('T', ' '), src, 'timestamp_utc', 'text', 0)
put('env.package_sha', ENV['package_zip_sha256'], src, 'package_zip_sha256', 'text', 0)
put('env.package_sha_short', ENV['package_zip_sha256'][:8], src, 'package_zip_sha256[:8]', 'text', 0)
LOCK = jload(MAIN / 'manifest_lock.json')
put('lock.sha', LOCK['sha256'], MAIN / 'manifest_lock.json', 'sha256', 'text', 0)
put('lock.sha_short', LOCK['sha256'][:8], MAIN / 'manifest_lock.json', 'sha256[:8]', 'text', 0)
put('lock.bytes', LOCK['bytes'], MAIN / 'manifest_lock.json', 'bytes', 'int', 0)
put('lock.utc', LOCK['locked_utc'][11:16] + ' UTC on 7 October 2026', MAIN / 'manifest_lock.json', 'locked_utc (2026-10-07T09:56:13Z, reformatted)', 'text', 0)
EM = jload(MAIN / 'encoder_meta.json')
src = MAIN / 'encoder_meta.json'
for enc, d in EM.items():
    e = ENC[enc]
    put(f'enc.{e}.dim', d['dim'], src, f'{enc}.dim', 'int', 0)
    if 'revision' in d:
        put(f'enc.{e}.revision', d['revision'], src, f'{enc}.revision', 'text', 0)
        put(f'enc.{e}.revision_short', d['revision'][:8], src, f'{enc}.revision[:8]', 'text', 0)
    if 'model' in d:
        put(f'enc.{e}.model', d['model'], src, f'{enc}.model', 'text', 0)
    put(f'enc.{e}.truncated_all', d['truncated_texts_all'], src, f'{enc}.truncated_texts_all', 'int', 0)
    put(f'enc.{e}.nevir_truncated_docs', d['nevir_truncated_docs'], src, f'{enc}.nevir_truncated_docs', 'int', 0)
    if e == 'E3':
        put('enc.E3.vocabulary', d['vocabulary'], src, f'{enc}.vocabulary', 'int', 0)
        put('enc.E3.oov_train_dev', 100 * d['oov_rate_train_dev'], src, f'100 x {enc}.oov_rate_train_dev', 'pct', 2)
        put('enc.E3.oov_all', 100 * d['oov_rate_all'], src, f'100 x {enc}.oov_rate_all', 'pct', 2)
        put('enc.E3.oov_examples', ', '.join(d['oov_examples'][i] for i in (0, 4, 6, 7)), src, f'{enc}.oov_examples[0], [4], [6], [7] (the first example and three word-like examples of the recorded list)', 'text', 0)
        put('enc.E3.source_zip_sha_short', d['source_zip_sha256'][:8], src, f'{enc}.source_zip_sha256[:8]', 'text', 0)
    if e == 'E6' and 'projection_shape' in d:
        put('enc.E6.proj_shape', 'x'.join(str(x) for x in d['projection_shape']), src, f'{enc}.projection_shape', 'text', 0)
put('enc.truncated_total', sum(d['truncated_texts_all'] for d in EM.values()), src, 'sum of truncated_texts_all', 'int', 0)
HS = jload(MAIN / 'hf_pipeline_sanity.json')
for enc, d in HS.items():
    if enc in ENC:
        e = ENC[enc]
        put(f'hf.{e}.unk_rate', d['unk_rate'], MAIN / 'hf_pipeline_sanity.json', f'{enc}.unk_rate', 'float', 2)
        put(f'hf.{e}.parameters', d['parameters'], MAIN / 'hf_pipeline_sanity.json', f'{enc}.parameters', 'int', 0)
        put(f'hf.{e}.missing_keys', len(d['missing_keys']), MAIN / 'hf_pipeline_sanity.json', f'len({enc}.missing_keys)', 'int', 0)
put('hf.unk_total', sum(d['unk_tokens'] for enc, d in HS.items() if enc in ENC), MAIN / 'hf_pipeline_sanity.json', 'sum unk_tokens over encoders', 'int', 0)
DA = jload(MAIN / 'data_audit.json')
src = MAIN / 'data_audit.json'
put('data.v1.train.items', DA['train']['items'], src, 'train.items', 'int', 0)
put('data.v1.train.scenarios', DA['train']['scenarios'], src, 'train.scenarios', 'int', 0)
put('data.v1.dev.items', DA['dev']['items'], src, 'dev.items', 'int', 0)
put('data.v1.dev.scenarios', DA['dev']['scenarios'], src, 'dev.scenarios', 'int', 0)
put('data.v1.test.items', DA['v1test']['items'], src, 'v1test.items', 'int', 0)
put('data.v1.test.scenarios', DA['v1test']['items'] // 4, src, 'v1test.items / 4', 'int', 0)
put('data.fresh.items', DA['fresh']['items'], src, 'fresh.items', 'int', 0)
put('data.perm.items', DA['perm']['items'], src, 'perm.items', 'int', 0)
put('data.fresh.scenarios', DA['v3_build']['fresh_scenarios'], src, 'v3_build.fresh_scenarios', 'int', 0)
put('data.perm.scenarios', DA['v3_build']['perm_scenarios'], src, 'v3_build.perm_scenarios', 'int', 0)
put('data.v1.scenarios', DA['train']['scenarios'] + DA['dev']['scenarios'] + DA['v1test']['items'] // 4, src, 'train + dev + test scenarios', 'int', 0)
put('data.test_parsed_in_pilot', bool(DA['test_files_parsed_in_pilot']), src, 'test_files_parsed_in_pilot', 'bool', 0)
put('data.all.scenarios', DA['train']['scenarios'] + DA['dev']['scenarios'] + DA['v1test']['items'] // 4 + DA['v3_build']['fresh_scenarios'] + DA['v3_build']['perm_scenarios'],
    src, 'v1 + Fresh + Perm scenarios', 'int', 0)
put('data.all.items', DA['train']['items'] + DA['dev']['items'] + DA['v1test']['items'] + DA['fresh']['items'] + DA['perm']['items'], src, 'all items', 'int', 0)
for name in ('fresh', 'perm', 'v1test'):
    ids = jload(MAIN / f'S_same_ids_{name}.json')
    put(f'data.{name}.S_same', len(ids), MAIN / f'S_same_ids_{name}.json', 'len', 'int', 0)
put('data.fresh.S_same_pct', 100 * REG['data.fresh.S_same']['value'] / 240, MAIN / 'S_same_ids_fresh.json', '100 x len / 240', 'pct', 2)
TB = jload(MAIN / 'perm_token_bag_check.json')
put('data.perm.bag_equal_all_tokenizers', all(d['n_equal'] == d['n_scenarios'] for d in TB.values()), MAIN / 'perm_token_bag_check.json', 'n_equal == n_scenarios for every encoder', 'bool', 0)
put('data.perm.bag_equal_min', min(d['n_equal'] for d in TB.values()), MAIN / 'perm_token_bag_check.json', 'min n_equal over encoders', 'int', 0)

# =============================================================================================== resolver diagnostics (exact fractions)
def diag(name):
    rows = jlines(MAIN / f'arm_diagnostics_{name}.jsonl')
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[('seen' if r['seen_family'] else 'unseen', r['arm'])].append(r)
        groups[(r['family'], r['arm'])].append(r)
        groups[('all', r['arm'])].append(r)
    return groups


for name in ('fresh', 'perm', 'v1test'):
    groups = diag(name)
    src = MAIN / f'arm_diagnostics_{name}.jsonl'
    for (g, arm), rows in groups.items():
        n = len(rows)
        put(f'res.{name}.{g}.{arm}.n', n, src, f'rows with group {g}, arm {arm}', 'int', 0)
        for field in ('any_cue_detected', 'cue_detected', 'decisive_in_scope', 'same_sentence', 'scope_in_decisive_clause', 'scope_in_other_clause', 'decisive_found'):
            vals = [r[field] for r in rows if field in r and r[field] is not None]
            if vals:
                put(f'res.{name}.{g}.{arm}.{field}.num', sum(1 for v in vals if v), src, f'count of {field} true ({g}/{arm})', 'int', 0)
                put(f'res.{name}.{g}.{arm}.{field}.den', len(vals), src, f'count of {field} defined ({g}/{arm})', 'int', 0)
                put(f'res.{name}.{g}.{arm}.{field}.pct', 100 * sum(1 for v in vals if v) / len(vals), src, f'100 x {field} true / defined ({g}/{arm})', 'pct', 2)
SR = jload(MAIN / 'scope_resolver_report.json')
put('res.spacy', SR['spacy'], MAIN / 'scope_resolver_report.json', 'spacy', 'text', 0)

# =============================================================================================== secondary accuracy view (v2.0 draft rules, descriptive)
AS = jload(MAIN / 'rules_accuracy_secondary.json')
src = MAIN / 'rules_accuracy_secondary.json'
for k, v in AS['thresholds'].items():
    put(f'thr2.{k}', v, src, f'thresholds.{k}', 'pct' if 'acc' in k or 'gain' in k or 'loss' in k or 'gap' in k or 'margin' in k else 'int', 0)
for enc, d in AS['G'].items():
    e = ENC[enc]
    put(f'acc2.G.{e}.A1_acc_OP', d['A1_seed_mean_acc_OP'], src, f'G.{enc}.A1_seed_mean_acc_OP', 'pct', 2)
    put(f'acc2.G.{e}.A1_incong', d['A1_seed_mean_acc_OP_prior_incongruent'], src, f'G.{enc}.A1_seed_mean_acc_OP_prior_incongruent', 'pct', 2)
    put(f'acc2.G.{e}.C2_acc_OP', d['C2_option_only_seed_mean_acc_OP'], src, f'G.{enc}.C2_option_only_seed_mean_acc_OP', 'pct', 2)
    put(f'acc2.G.{e}.margin', d['margin_over_C2'], src, f'G.{enc}.margin_over_C2', 'pp', 2)
    put(f'acc2.G.{e}.passed', bool(d['passed']), src, f'G.{enc}.passed', 'bool', 0)
strata = next(iter(AS['G'].values()))['n_scenarios_by_stratum']
for k, v in strata.items():
    put(f'strata.fresh.{k}', v, src, f'G.*.n_scenarios_by_stratum.{k}', 'int', 0)
put('acc2.n_passed', sum(1 for d in AS['G'].values() if d['passed']), src, 'count of G.*.passed', 'int', 0)
put_range('rng.acc2.margin', [f'acc2.G.{e}.margin' for e in ENCS7], 'acc2.G.{e}.margin over encoders', 'pp')
MC = jload(MAIN / 'mcnemar_accuracy_secondary.json')
put('mcnemar.n_tests', len(MC), MAIN / 'mcnemar_accuracy_secondary.json', 'len', 'int', 0)

# =============================================================================================== build of the v3.0 sets
BR = jload(BUILD / 'build_report.json')
src = BUILD / 'build_report.json'
put('build.utc', BR['built_utc'][11:16] + ' UTC on 5 October 2026', src, 'built_utc (2026-10-05T19:36Z, reformatted)', 'text', 0)
put('build.cacsa2', BR['cacsa2_version'], src, 'cacsa2_version', 'text', 0)
put('build.package_sha_short', BR['package_zip_sha256'][:8], src, 'package_zip_sha256[:8]', 'text', 0)
put('build.manifest_sha', BR['manifest']['sha256'], src, 'manifest.sha256', 'text', 0)
put('build.gpu', BR['gpu'].split(',')[0], src, 'gpu (name)', 'text', 0)
for k, v in BR['models'].items():
    if v is not None:
        put(f'build.models.{k}', v, src, f'models.{k}', 'text' if isinstance(v, str) else ('float' if isinstance(v, float) else 'int'), 1 if isinstance(v, float) else 0)
for k, v in BR['prompt_sha256'].items():
    put(f'build.prompt.{k}', v, src, f'prompt_sha256.{k}', 'text', 0)
    put(f'build.prompt.{k}.short', v[:8], src, f'prompt_sha256.{k}[:8]', 'text', 0)
put('build.n_prompt_hashes', len(BR['prompt_sha256']), src, 'len(prompt_sha256)', 'int', 0)
for k, v in BR['seeds'].items():
    put(f'build.seed.{k}', v, src, f'seeds.{k}', 'id', 0)
for k, v in BR['scheduled_per_family'].items():
    put(f'build.scheduled.{k}', v, src, f'scheduled_per_family.{k}', 'int', 0)
for k, v in BR['budget_caps_usd'].items():
    put(f'build.cap_usd.{k}', v, src, f'budget_caps_usd.{k}', 'usd', 2)
for setname in ('perm', 'fresh'):
    d = BR[setname]
    put(f'build.{setname}.candidates', d['candidates'], src, f'{setname}.candidates', 'int', 0)
    for st, v in d['stages'].items():
        put(f'build.{setname}.stage.{st}', v, src, f'{setname}.stages.{st}', 'int', 0)
    for st in ('verifier', 'qwen', 'accepted', 'filter', 'duplicate', 'generator', 'cap'):
        if st not in d['stages']:
            put(f'build.{setname}.stage.{st}', 0, src, f'{setname}.stages.{st} (absent = 0)', 'int', 0)
    put(f'build.{setname}.qwen_checked', d['qwen']['checked'], src, f'{setname}.qwen.checked', 'int', 0)
    put(f'build.{setname}.qwen_passed', d['qwen']['passed'], src, f'{setname}.qwen.passed', 'int', 0)
    put(f'build.{setname}.halted', d['halted'], src, f'{setname}.halted', 'text', 0)
    for stratum, v in d['accepted_by_stratum'].items():
        put(f'build.{setname}.accepted.{stratum.replace("|", ".")}', v, src, f'{setname}.accepted_by_stratum.{stratum}', 'int', 0)
    for stratum, dd in d['stages_by_stratum'].items():
        for st, v in dd.items():
            put(f'build.{setname}.stages.{stratum.replace("|", ".")}.{st}', v, src, f'{setname}.stages_by_stratum.{stratum}.{st}', 'int', 0)
        put(f'build.{setname}.stages.{stratum.replace("|", ".")}.candidates', sum(dd.values()), src, f'sum of {setname}.stages_by_stratum.{stratum}.*', 'int', 0)
        put(f'build.{setname}.stages.{stratum.replace("|", ".")}.other', dd.get('generator', 0) + dd.get('cap', 0), src, f'{setname}.stages_by_stratum.{stratum}.generator + .cap', 'int', 0)
    put(f'build.{setname}.stage.other', d['stages'].get('generator', 0) + d['stages'].get('cap', 0), src, f'{setname}.stages.generator + .cap', 'int', 0)
    # Qwen pass of Haiku survivors per family: accepted + qwen (+ duplicate at acceptance + cap) / (accepted + qwen + duplicate + cap)
    fam_acc = defaultdict(int)
    fam_qwen = defaultdict(int)
    for stratum, dd in d['stages_by_stratum'].items():
        fam = stratum.split('|')[0]
        fam_acc[fam] += dd.get('accepted', 0) + dd.get('cap', 0)
        fam_qwen[fam] += dd.get('qwen', 0)
    for fam in fam_acc:
        put(f'build.{setname}.qwenpass.{fam}.num', fam_acc[fam], src, f'{setname}: accepted + cap in family {fam} (candidates that passed Qwen)', 'int', 0)
        put(f'build.{setname}.qwenpass.{fam}.den', fam_acc[fam] + fam_qwen[fam], src, f'{setname}: accepted + cap + qwen in family {fam} (candidates shown to Qwen)', 'int', 0)
    for arm, lb in d['label_balance'].items():
        put(f'build.{setname}.label.{arm}.0', lb['0'], src, f'{setname}.label_balance.{arm}.0', 'int', 0)
        put(f'build.{setname}.label.{arm}.1', lb['1'], src, f'{setname}.label_balance.{arm}.1', 'int', 0)
    # discard reasons (top)
    dr = d['discard_reasons']
    for i, (reason, n) in enumerate(sorted(dr.items(), key=lambda kv: -kv[1])[:8]):
        put(f'build.{setname}.discard.{i}.n', n, src, f'{setname}.discard_reasons[{reason!r}]', 'int', 0)
        put(f'build.{setname}.discard.{i}.reason', reason, src, f'{setname}.discard_reasons (key)', 'text', 0)
    put(f'build.{setname}.discard.O_unclear', dr.get("arm O read as 'unclear'", 0), src, f"{setname}.discard_reasons[arm O read as 'unclear']", 'int', 0)
    put(f'build.{setname}.discard.N_unclear', dr.get("arm N read as 'unclear'", 0), src, f"{setname}.discard_reasons[arm N read as 'unclear']", 'int', 0)
    put(f'build.{setname}.discard.N_as_A', dr.get("arm N read as 'A'", 0), src, f"{setname}.discard_reasons[arm N read as 'A']", 'int', 0)
    put(f'build.{setname}.discard.N_as_B', dr.get("arm N read as 'B'", 0), src, f"{setname}.discard_reasons[arm N read as 'B']", 'int', 0)
    put(f'build.{setname}.discard.qwen_N_inconsistent', dr.get("qwen arm N read as 'inconsistent'", 0), src, f"{setname}.discard_reasons[qwen arm N read as 'inconsistent']", 'int', 0)
    put(f'build.{setname}.discard.near_dup', dr.get('near-duplicate (before verification)', 0) + dr.get('near-duplicate of a kept scenario', 0), src, f'{setname}: near-duplicate reasons summed', 'int', 0)
put('build.perm.bag_assertion', BR['perm']['bag_assertion']['equal_E0_bags_N_S'], src, 'perm.bag_assertion.equal_E0_bags_N_S', 'int', 0)
put('build.perm.cap_per_stratum', BR['perm']['caps']['syntactic|first'], src, 'perm.caps.syntactic|first', 'int', 0)
put('build.perm.cap_total', sum(BR['perm']['caps'].values()), src, 'sum perm.caps', 'int', 0)
put('build.fresh.cap_per_family', BR['fresh']['caps']['syntactic'], src, 'fresh.caps.syntactic', 'int', 0)
put('build.fresh.cap_total', sum(BR['fresh']['caps'].values()), src, 'sum fresh.caps', 'int', 0)
put('build.perm.seen_scenarios', sum(v for k, v in BR['perm']['accepted_by_stratum'].items() if k.split('|')[0] in ('syntactic', 'quantifier')), src, 'perm accepted in syntactic + quantifier', 'int', 0)
put('build.perm.unseen_scenarios', sum(v for k, v in BR['perm']['accepted_by_stratum'].items() if k.split('|')[0] in ('affixal', 'lexical')), src, 'perm accepted in affixal + lexical', 'int', 0)
put('build.perm.affixal_total', BR['perm']['accepted_by_stratum']['affixal|first'] + BR['perm']['accepted_by_stratum']['affixal|second'], src, 'perm affixal first + second', 'int', 0)
put('build.perm.lexical_total', BR['perm']['accepted_by_stratum']['lexical|first'] + BR['perm']['accepted_by_stratum']['lexical|second'], src, 'perm lexical first + second', 'int', 0)
put('build.fresh.seen_scenarios', BR['fresh']['accepted_by_family']['syntactic'] + BR['fresh']['accepted_by_family']['quantifier'], src, 'fresh syntactic + quantifier', 'int', 0)
put('build.fresh.unseen_scenarios', BR['fresh']['accepted_by_family']['affixal'] + BR['fresh']['accepted_by_family']['lexical'], src, 'fresh affixal + lexical', 'int', 0)
for setname in ('perm', 'fresh'):
    d = BR['opus_audit'][setname]
    put(f'build.audit.{setname}.sample', d['sample'], src, f'opus_audit.{setname}.sample', 'int', 0)
    put(f'build.audit.{setname}.agree_all', d['agree_all_arms'], src, f'opus_audit.{setname}.agree_all_arms', 'int', 0)
cm = BR['cost_metered']
put('build.cost.total', cm['usd'], src, 'cost_metered.usd', 'usd', 2)
put('build.cost.calls', cm['calls'], src, 'cost_metered.calls', 'int', 0)
put('build.cost.tokens_in', cm['in'], src, 'cost_metered.in', 'int', 0)
put('build.cost.tokens_out', cm['out'], src, 'cost_metered.out', 'int', 0)
for model, (usd, calls, tin, tout) in cm['by_model'].items():
    tag = {'claude-sonnet-5': 'sonnet', 'claude-haiku-4-5-20251001': 'haiku', 'claude-opus-5': 'opus'}[model]
    put(f'build.cost.{tag}.usd', usd, src, f'cost_metered.by_model.{model}[0]', 'usd', 2)
    put(f'build.cost.{tag}.calls', calls, src, f'cost_metered.by_model.{model}[1]', 'int', 0)
put('build.human_verification', BR['human_verification'], src, 'human_verification', 'text', 0)
# phase costs from the build state meter
BS = jload(BUILD / 'build_state.json')
meter = BS['meter']
put('build.meter.usd', meter['usd'], BUILD / 'build_state.json', 'meter.usd', 'usd', 2)
# Opus audit details
def audit_details(setname, scen_file, expected_fn):
    scen = {r['meta']['uid']: r for r in jlines(BUILD / scen_file)}
    aud = BS['audit'][setname]
    agree_ans = Counter()
    natural = Counter()
    unclear = Counter()
    n = 0
    for sid, a in aud.items():
        exp = expected_fn(scen[sid])
        n += 1
        for arm in 'OPNS':
            v = a['verdicts'][arm]
            if v['answer'] == exp[arm]:
                agree_ans[arm] += 1
            if v['answer'] == 'unclear':
                unclear[arm] += 1
            if v['natural']:
                natural[arm] += 1
    src2 = BUILD / 'build_state.json'
    put(f'build.audit.{setname}.n', n, src2, f'len(audit.{setname})', 'int', 0)
    put(f'build.audit.{setname}.answers_agree', sum(agree_ans.values()), src2, f'audit.{setname}: arm answers equal to the expected option, summed over O, P, N, S', 'int', 0)
    put(f'build.audit.{setname}.answers_total', 4 * n, src2, f'4 x len(audit.{setname})', 'int', 0)
    for arm in 'OPNS':
        put(f'build.audit.{setname}.agree.{arm}', agree_ans[arm], src2, f'audit.{setname}: {arm} answers equal to the expected option', 'int', 0)
        put(f'build.audit.{setname}.natural.{arm}', natural[arm], src2, f'audit.{setname}: {arm} judged natural', 'int', 0)
        put(f'build.audit.{setname}.unclear.{arm}', unclear[arm], src2, f'audit.{setname}: {arm} answered unclear', 'int', 0)
    fams = Counter(scen[sid]['meta']['family'] for sid in aud)
    for f, c in fams.items():
        put(f'build.audit.{setname}.family.{f}', c, src2, f'audit.{setname}: scenarios of family {f}', 'int', 0)


def perm_expected(r):
    y = r['generated']['answer']
    other = 'B' if y == 'A' else 'A'
    return {'O': y, 'P': y, 'S': y, 'N': other}


def fresh_expected(r):
    y = r['generated']['answer_O']
    other = 'B' if y == 'A' else 'A'
    return {'O': y, 'P': y, 'S': y, 'N': other}


audit_details('perm', 'perm_scenarios.jsonl', perm_expected)
audit_details('fresh', 'fresh_scenarios.jsonl', fresh_expected)

# ---- dataset statistics recomputed from the item files
from cacsa2 import negpairs_build as nb  # noqa: E402

PERM = jlines(BUILD / 'perm.jsonl')
FRESH = jlines(BUILD / 'fresh.jsonl')
PSCEN = {r['meta']['uid']: r for r in jlines(BUILD / 'perm_scenarios.jsonl')}
FSCEN = {r['meta']['uid']: r for r in jlines(BUILD / 'fresh_scenarios.jsonl')}


def arms_of(rows):
    out = defaultdict(dict)
    for r in rows:
        out[r['scenario_id']][r['arm']] = r
    return out


PA = arms_of(PERM)
FA = arms_of(FRESH)
src = BUILD / 'perm.jsonl'
lower = sum(1 for s in PA.values() if s['O']['context'][:1].islower())
nopunct = sum(1 for s in PA.values() if s['O']['context'].rstrip()[-1:] not in '.!?')
put('data.perm.lowercase_start', lower, src, 'O contexts whose first character is lower case', 'int', 0)
put('data.perm.no_final_punct', nopunct, src, 'O contexts that do not end with . ! or ?', 'int', 0)
put('data.perm.P_lowercase_start', sum(1 for s in PA.values() if s['P']['context'][:1].islower()), src, 'P contexts whose first character is lower case', 'int', 0)
wc = [len(nb.words(s['O']['context'])) for s in PA.values()]
wc_tok = [len(s['O']['context'].split()) for s in PA.values()]
put('data.perm.O_words.min', min(wc_tok), src, 'min whitespace words of O', 'int', 0)
put('data.perm.O_words.max', max(wc_tok), src, 'max whitespace words of O', 'int', 0)
put('data.perm.O_words.median', statistics.median(wc_tok), src, 'median whitespace words of O', 'int', 0)
put('data.perm.max_words_any_arm', max(len(s[a]['context'].split()) for s in PA.values() for a in 'OPNS'), src, 'max whitespace words over all arms', 'int', 0)
conn = Counter(s['O']['perm']['connector'] for s in PA.values())
for c, n in conn.items():
    put(f'data.perm.connector.{c.strip().replace(",", "comma_").replace(" ", "")}', n, src, f'count of perm.connector == {c!r}', 'int', 0)
put('data.perm.connector.n_distinct', len(conn), src, 'distinct connectors', 'int', 0)
cues = Counter((s['N']['family'], s['N']['cue']) for s in PA.values())
for (f, c), n in cues.items():
    put(f'data.perm.cue.{f}.{c}', n, src, f'count of N cue {c!r} in family {f}', 'int', 0)
put('data.perm.cue.affixal.n_distinct', len([1 for (f, c) in cues if f == 'affixal']), src, 'distinct affixal negated forms', 'int', 0)
put('data.perm.cue.lexical.n_distinct', len([1 for (f, c) in cues if f == 'lexical']), src, 'distinct lexical cues', 'int', 0)
bases = Counter(s['O']['perm']['site_words'][0] for s in PA.values() if s['O']['family'] == 'affixal')
put('data.perm.affixal.n_base_words', len(bases), src, 'distinct base words W of affixal scenarios', 'int', 0)
sites = Counter(s['O']['perm']['site_words'][0] for s in PA.values() if s['O']['family'] == 'quantifier')
for w, n in sites.items():
    put(f'data.perm.quantifier.site.{w}', n, src, f'quantifier scenarios with site word {w!r}', 'int', 0)
aux = Counter()
for s in PA.values():
    if s['O']['family'] == 'syntactic':
        ctx = s['N']['context']
        m = re.search(r'(\w+) not\b', ctx)
        aux[m.group(1)] += 1
for w, n in aux.items():
    put(f'data.perm.syntactic.aux.{w}', n, src, f'syntactic scenarios whose "not" follows {w!r}', 'int', 0)
dom = Counter(s['O']['domain'] for s in PA.values())
sit = Counter(s['O']['situation'] for s in PA.values())
put('data.perm.domains', len(dom), src, 'distinct domains', 'int', 0)
put('data.perm.situations', len(sit), src, 'distinct situations', 'int', 0)
first = Counter((s['O']['family'], s['O']['perm']['decisive_first']) for s in PA.values())
for (f, df), n in first.items():
    put(f'data.perm.order.{f}.{"first" if df else "second"}', n, src, f'scenarios of family {f} with decisive clause {"first" if df else "second"}', 'int', 0)
put('data.perm.order.first', sum(n for (f, df), n in first.items() if df), src, 'scenarios with decisive clause first', 'int', 0)
put('data.perm.order.second', sum(n for (f, df), n in first.items() if not df), src, 'scenarios with decisive clause second', 'int', 0)
# edit distances
ed_n = [nb.token_edit_distance(nb.words(s['O']['context']), nb.words(s['N']['context'])) for s in PA.values()]
ed_s = [nb.token_edit_distance(nb.words(s['O']['context']), nb.words(s['S']['context'])) for s in PA.values()]
ed_p = [nb.token_edit_distance(nb.words(s['O']['context']), nb.words(s['P']['context'])) for s in PA.values()]
put('data.perm.edit_N.max', max(ed_n), src, 'max word edit distance O to N', 'int', 0)
put('data.perm.edit_S.max', max(ed_s), src, 'max word edit distance O to S', 'int', 0)
put('data.perm.edit_P.min', min(ed_p), src, 'min word edit distance O to P', 'int', 0)
put('data.perm.edit_P.median', statistics.median(ed_p), src, 'median word edit distance O to P', 'int', 0)
# bag equality check (E0 tokenizer)
from cacsa2.encoders import _WORD_RE as E0RE  # noqa: E402
def e0bag(t):
    return Counter(x.lower() for x in E0RE.findall(t))
put('data.perm.bag_equal_recomputed', sum(1 for s in PA.values() if e0bag(s['N']['context']) == e0bag(s['S']['context'])), src, 'scenarios with equal E0 bags of N and S (recomputed)', 'int', 0)
# insertion pairs: scenarios where Bag(O) is a proper sub-bag of Bag(N) or Bag(S) under E0 tokenizer with the question
def subbag(a, b):
    return all(b[k] >= v for k, v in a.items()) and sum(b.values()) > sum(a.values())
ins = sum(1 for s in PA.values() for x in 'NS' if subbag(e0bag(s['O']['context'] + ' ' + s['O']['question']), e0bag(s[x]['context'] + ' ' + s[x]['question'])))
put('data.perm.insertion_pairs_recomputed', ins, src, 'pairs (O, X) with Bag(O) a proper sub-bag of Bag(X), X in {N, S}, E0 tokenizer, context + question', 'int', 0)

src = BUILD / 'fresh.jsonl'
put('data.fresh.capitalized', sum(1 for s in FA.values() if s['O']['context'][:1].isupper()), src, 'O contexts starting with an upper-case letter', 'int', 0)
put('data.fresh.final_punct', sum(1 for s in FA.values() if s['O']['context'].rstrip()[-1:] in '.!?'), src, 'O contexts ending with . ! or ?', 'int', 0)
ed_n = [nb.token_edit_distance(nb.words(s['O']['context']), nb.words(s['N']['context'])) for s in FA.values()]
ed_s = [nb.token_edit_distance(nb.words(s['O']['context']), nb.words(s['S']['context'])) for s in FA.values()]
for k in range(1, 6):
    put(f'data.fresh.edit_N.{k}', sum(1 for x in ed_n if x == k), src, f'scenarios with word edit distance O to N == {k}', 'int', 0)
put('data.fresh.edit_N.median', statistics.median(ed_n), src, 'median word edit distance O to N', 'int', 0)
put('data.fresh.edit_N.max', max(ed_n), src, 'max word edit distance O to N', 'int', 0)
put('data.fresh.edit_S.median', statistics.median(ed_s), src, 'median word edit distance O to S', 'int', 0)
put('data.fresh.edit_S.max', max(ed_s), src, 'max word edit distance O to S', 'int', 0)
cn = Counter((s['N']['family'], s['N']['cue'].lower()) for s in FA.values())
cs = Counter((s['S']['family'], s['S']['cue'].lower()) for s in FA.values())
for (f, c), n in cn.most_common():
    put(f'data.fresh.cueN.{f}.{c.replace(" ", "_")}', n, src, f'N cue {c!r} in family {f}', 'int', 0)
for (f, c), n in cs.most_common():
    put(f'data.fresh.cueS.{f}.{c.replace(" ", "_")}', n, src, f'S cue {c!r} in family {f}', 'int', 0)
for f in ('syntactic', 'quantifier', 'affixal', 'lexical'):
    put(f'data.fresh.cueN.{f}.n_distinct', len([1 for (ff, c) in cn if ff == f]), src, f'distinct N cues in family {f}', 'int', 0)
    put(f'data.fresh.cueS.{f}.n_distinct', len([1 for (ff, c) in cs if ff == f]), src, f'distinct S cues in family {f}', 'int', 0)
put('data.fresh.domains', len(Counter(s['O']['domain'] for s in FA.values())), src, 'distinct domains', 'int', 0)
put('data.fresh.situations', len(Counter(s['O']['situation'] for s in FA.values())), src, 'distinct situations', 'int', 0)
wc = [len(s['O']['context'].split()) for s in FA.values()]
put('data.fresh.O_words.median', statistics.median(wc), src, 'median whitespace words of O', 'int', 0)
put('data.fresh.O_words.max', max(len(s[a]['context'].split()) for s in FA.values() for a in 'OPNS'), src, 'max whitespace words over all arms', 'int', 0)
ins = sum(1 for s in FA.values() for x in 'NS' if subbag(e0bag(s['O']['context'] + ' ' + s['O']['question']), e0bag(s[x]['context'] + ' ' + s[x]['question'])))
put('data.fresh.insertion_pairs_recomputed', ins, src, 'pairs (O, X) with Bag(O) a proper sub-bag of Bag(X), E0 tokenizer, context + question', 'int', 0)
# unseen-family cue overlap with v1 train/dev text
train_text = ' '.join(r['context'].lower() + ' ' + r['question'].lower() for sp in ('train', 'dev') for r in jlines(V1 / f'{sp}.jsonl'))
def in_train(c):
    return re.search(r'(?<![a-z])' + re.escape(c.lower()) + r'(?![a-z])', train_text) is not None
for f in ('affixal', 'lexical'):
    cuesN = {s['N']['cue'].lower() for s in FA.values() if s['N']['family'] == f}
    put(f'data.fresh.unseen.{f}.N_cues_in_train', sum(1 for c in cuesN if in_train(c)), [src, V1 / 'train.jsonl'], f'distinct {f} N cues of Fresh that occur as words in v1 train or dev text', 'int', 0)
    cuesN = {s['N']['cue'].lower() for s in PA.values() if s['N']['family'] == f}
    put(f'data.perm.unseen.{f}.N_cues_in_train', sum(1 for c in cuesN if in_train(c)), [BUILD / 'perm.jsonl', V1 / 'train.jsonl'], f'distinct {f} N cues of Perm that occur as words in v1 train or dev text', 'int', 0)

# Qwen letter-logit margins on accepted scenarios
def margins(scen):
    out = []
    for r in scen.values():
        for arm in 'OPNS':
            for order in ('ab', 'ba'):
                lg = sorted(r['qwen'][arm][order], reverse=True)
                out.append(lg[0] - lg[1])
    return out
for setname, scen in (('perm', PSCEN), ('fresh', FSCEN)):
    m = margins(scen)
    srcf = BUILD / f'{setname}_scenarios.jsonl'
    put(f'build.{setname}.qwen_margin.median', statistics.median(m), srcf, 'median of (top logit - second logit) over arms and option orders of accepted scenarios', 'float', 1)
    put(f'build.{setname}.qwen_margin.min', min(m), srcf, 'min of (top logit - second logit)', 'float', 3)
    put(f'build.{setname}.qwen_margin.n_below_0_5', sum(1 for x in m if x < 0.5), srcf, 'presentations with margin below 0.5', 'int', 0)
    put(f'build.{setname}.qwen_margin.n', len(m), srcf, 'presentations (scenarios x 4 arms x 2 orders)', 'int', 0)
    put(f'build.{setname}.qwen_margin.n_ties', sum(1 for x in m if x == 0), srcf, 'exact ties', 'int', 0)

# =============================================================================================== independent recomputation
IR = jload(INDEP)
put('indep.runs', IR['runs'], INDEP, 'runs', 'int', 0)
put('indep.compared', IR['compared'], INDEP, 'compared', 'int', 0)
put('indep.mismatches', IR['n_mismatches'], INDEP, 'n_mismatches', 'int', 0)
put('indep.max_abs_diff', IR['max_abs_diff'], INDEP, 'max_abs_diff', 'float', 1)
put('indep.gate_reproduced', all(v['passed_recomputed'] == v['passed_notebook'] for v in IR['gate'].values()), INDEP, 'gate decisions equal for every encoder', 'bool', 0)
put('indep.pred_files', 2935, 'np_v3_main_predictions.zip (file count, np_v3_main_audit.md section 7)', 'files in the predictions zip', 'int', 0)
put('indep.pred_zip_mb', 30.7, 'np_v3_main_predictions.zip (size, np_v3_main_audit.md section 7)', 'size in MB', 'float', 1)

# =============================================================================================== float64 follow-up (optional)
if (FU / 'seed_means_perm_f64_static.json').exists():
    p = FU / 'seed_means_perm_f64_static.json'
    for r in jload(p):
        e, a = ENC[r['encoder']], AGG[r['aggregator']]
        for k, v in r.items():
            if k in ('encoder', 'aggregator', 'n_seeds') or v is None or k.endswith('_sd'):
                continue
            kind, dec = metric_kind(k)
            put(f'f64.{e}.{a}.{k}', v, p, f'[{r["encoder"]},{r["aggregator"]}].{k} (float64 scores)', kind, dec)
    p = FU / 'per_run_perm_f64_static.json'
    by_cond = defaultdict(list)
    for r in jload(p):
        e, a = ENC[r['encoder']], AGG[r['aggregator']]
        by_cond[(e, a)].append(r)
        for m in PR_METRICS['perm']:
            if r.get(m) is not None:
                kind, dec = metric_kind(m)
                put(f'prf64.{e}.{a}.s{r["seed"]}.{m}', r[m], p, f'[{r["encoder"]},{r["aggregator"]},seed={r["seed"]}].{m} (float64 scores)', kind, dec)
    for (e, a), rs in by_cond.items():
        if len(rs) < 2:
            continue
        for m in PR_METRICS['perm']:
            vals = [r[m] for r in rs if r.get(m) is not None]
            if len(vals) != len(rs):
                continue
            kind, dec = metric_kind(m)
            mean, sd, lo, hi = t_interval(vals)
            put(f'cif64.{e}.{a}.{m}.low', lo, p, f'seed-level 95% t interval of [{e},{a}].{m} (float64): low', kind, dec)
            put(f'cif64.{e}.{a}.{m}.high', hi, p, f'seed-level 95% t interval of [{e},{a}].{m} (float64): high', kind, dec)
            put(f'extf64.{e}.{a}.{m}.min', min(vals), p, f'min over seeds of [{e},{a}].{m} (float64)', kind, dec)
            put(f'extf64.{e}.{a}.{m}.max', max(vals), p, f'max over seeds of [{e},{a}].{m} (float64)', kind, dec)
            put(f'extf64.{e}.{a}.{m}.n_pos', sum(1 for v in vals if v > 0), p, f'seeds with [{e},{a}].{m} > 0 (float64)', 'int', 0)
    rep = jload(FU / 'perm_f64_vs_gpu.json')
    put('f64.runs_rescored', rep['runs_rescored'], FU / 'perm_f64_vs_gpu.json', 'runs_rescored', 'int', 0)
    put('f64.item_disagreements', rep['item_decision_disagreements_gpu_vs_f64']['total'], FU / 'perm_f64_vs_gpu.json', 'item_decision_disagreements_gpu_vs_f64.total', 'int', 0)
    put('f64.n_seed_mean_differences', rep['n_differences'], FU / 'perm_f64_vs_gpu.json', 'n_differences', 'int', 0)
    put('f64.items_per_run', rep['item_decision_disagreements_gpu_vs_f64']['items_per_run'], FU / 'perm_f64_vs_gpu.json', 'items_per_run', 'int', 0)
    put('f64.max_abs_seed_mean_diff', max((abs(d['float64'] - d['gpu']) for d in rep['seed_mean_differences']), default=0.0), FU / 'perm_f64_vs_gpu.json', 'max |float64 - gpu| over seed_mean_differences', 'pp', 2)
    put('f64.present', True, FU, 'follow-up files present', 'bool', 0)
else:
    put('f64.present', False, FU, 'follow-up files absent', 'bool', 0)

# =============================================================================================== derived counts
put('n.exact_checks', RULES['R5']['checks'] + RULES['R6']['checks'] + RULES['R7']['checks'], MAIN / 'rules.json', 'R5.checks + R6.checks + R7.checks', 'int', 0)
put('n.f64_item_predictions', 2 * 1096 * 101, FU / 'perm_f64_vs_gpu.json', '2 encoders x 1,096 Perm items x 101 runs per encoder (10 conditions x 10 seeds + M0)', 'int', 0)
F64REP = jload(S / 'indep_v3/out_f64/recompute_f64_report.json')
put('n.f64_compared', F64REP['compared'], S / 'indep_v3/out_f64/recompute_f64_report.json', 'compared', 'int', 0)
put('n.f64_mismatches', F64REP['n_mismatches'], S / 'indep_v3/out_f64/recompute_f64_report.json', 'n_mismatches', 'int', 0)
put('n.f64_seed_means_compared', F64REP['seed_means_compared'], S / 'indep_v3/out_f64/recompute_f64_report.json', 'seed_means_compared', 'int', 0)
put('n.f64_propA_checks', F64REP['propA']['E0-hash']['scenario_checks'], S / 'indep_v3/out_f64/recompute_f64_report.json', 'propA.E0-hash.scenario_checks (identical for E3)', 'int', 0)
put('n.pilot_lr_runs', len(list((MAIN / 'pilot' / 'lr_grid').glob('*/lr*/seed*/run_summary.json'))), MAIN / 'pilot/lr_grid',
    'count of run_summary.json files under pilot/lr_grid/{encoder}/lr*/seed*', 'int', 0)
put('n.qwen_train_disagree_pct', 100 - REG['qwen.v1.split.train.pct']['value'], MAIN / 'qwen_audit_v1.json', '100 - agreement.split/train.pct', 'pct', 2)
# paraphrase flip ranges of A1 (static Perm values from the float64 follow-up, as everywhere in the article)
put_range('rng.fresh.A1.flip_O_P', [f'sm.fresh.{e}.A1.all.flip_O_P' for e in ENCS7], 'sm.fresh.{e}.A1.all.flip_O_P over encoders', 'pct')
put_range('rng.perm.A1.flip_O_P', [f'f64.{e}.A1.all.pooled.flip_O_P' if e in STATIC else f'sm.perm.{e}.A1.all.pooled.flip_O_P' for e in ENCS7],
          'A1.all.pooled.flip_O_P on Perm over encoders (f64.* for E0/E3, sm.perm.* otherwise)', 'pct')
# NevIR: complement minus nested paired accuracy of M0 (positive = nested harder)
for e in CONTEXTUAL:
    put(f'nevir.{e}.M0.gap', REG[f'nevir.{e}.M0.complement.paired']['value'] - REG[f'nevir.{e}.M0.nested.paired']['value'], MAIN / 'nevir_results.json',
        f'scorers.{e}/M0.complement.paired_accuracy - scorers.{e}/M0.nested.paired_accuracy', 'pp', 2)
put('param.E6.M0n', 0, 'cacsa2/colbert.py', 'ColBERTNative has no trainable parameters', 'int', 0)
for e in ENCS7:
    if f'param.{e}.M0' not in REG:
        put(f'param.{e}.M0', 0, 'cacsa2/model.py', 'ZeroShotMaxSim has no parameters', 'int', 0)
put('code.tau_eps', OLDREG_TAU := jload(OLD)['code.tau_eps']['value'], 'cacsa2/model.py (v2.0 registry code.tau_eps)', 'temperature floor of A3', 'sci', 2)
put('cacsa2.release', re.search(r'__version__ = "([^"]+)"', (CODE / 'cacsa2/__init__.py').read_text()).group(1), CODE / 'cacsa2/__init__.py', '__version__', 'text', 0)
put('fu.cacsa2', jload(FU / 'perm_f64_vs_gpu.json')['cacsa2_version'] if (FU / 'perm_f64_vs_gpu.json').exists() else 'n/a', FU / 'perm_f64_vs_gpu.json', 'cacsa2_version', 'text', 0)
put('thr.c1_check_min', 75, 'CACSA_manifest_v3_0.md, section 11 (S3)', 'C1 check stop threshold', 'pct', 0)

# =============================================================================================== public systems as released (revision step 1, 21-22 October 2026)
PUB = S / 'public_v3/public_systems'
PUBF = PUB / 'metrics_full.json'
PUB_ROWS = {r['name']: r for r in jload(PUBF)}
PUB_SUM = jload(PUB / 'summary.json')
PUB_KEY = {'colbertv2': 'colbertv2', 'answerai-colbert-small-v1': 'answerai', 'gte-moderncolbert-v1': 'gtemc', 'bge-m3/colbert': 'bgem3c', 'bge-m3/dense': 'bgem3d',
           'bge-m3/sparse': 'bgem3s', 'bge-base-en-v1.5': 'bgebase', 'e5-base-v2': 'e5', 'all-mpnet-base-v2': 'mpnet', 'gte-modernbert-base': 'gtemb',
           'splade-cocondenser-ensembledistil': 'splade', 'ms-marco-minilm-l6-v2': 'minilm', 'bge-reranker-base': 'bgerr', 'bge-reranker-v2-m3': 'bgerr2',
           'mxbai-rerank-base-v1': 'mxbai', 'monot5-base': 'monot5b', 'monot5-3b': 'monot5l', 'olmo-2-7b-instruct': 'olmo', 'phi-4-mini-instruct': 'phi', 'qwen3-8b': 'qwen3'}
assert set(PUB_KEY) == set(PUB_ROWS), set(PUB_ROWS) ^ set(PUB_KEY)
PUB_LI = ['colbertv2', 'answerai', 'gtemc', 'bgem3c']                                     # pure MaxSim late-interaction systems
PUB_LLM = ['olmo', 'phi', 'qwen3']
PUB_RETR = [k for k in PUB_KEY.values() if k not in PUB_LLM]                              # 17 retrieval scorers (incl. rerankers)
for name, r in PUB_ROWS.items():
    k = PUB_KEY[name]
    put(f'pub.{k}.label', r['label'], PUBF, f'[{name}].label', 'text', 0)
    put(f'pub.{k}.kind', r['kind'], PUBF, f'[{name}].kind', 'text', 0)
    put(f'pub.{k}.revision', str(r['resolved_revision'])[:8], PUBF, f'[{name}].resolved_revision[:8]', 'text', 0)
    put(f'pub.{k}.seconds', r['seconds'], PUBF, f'[{name}].seconds', 'float', 0)
    f = r['perm']['flat']
    for m in ('acc_O', 'acc_P', 'acc_N', 'acc_S', 'flip_O_N', 'flip_O_S', 'flip_O_P', 'joint_NS_acc'):
        put(f'pub.{k}.perm.{m}', f[f'all.pooled.{m}'], PUBF, f'[{name}].perm.flat.all.pooled.{m}', 'pct', 2)
    put(f'pub.{k}.perm.SS', f['all.pooled.SS'], PUBF, f'[{name}].perm.flat.all.pooled.SS', 'pp', 2)
    put(f'pub.{k}.perm.SS_first', f['all.first.SS'], PUBF, f'[{name}].perm.flat.all.first.SS', 'pp', 2)
    put(f'pub.{k}.perm.SS_second', f['all.second.SS'], PUBF, f'[{name}].perm.flat.all.second.SS', 'pp', 2)
    put(f'pub.{k}.perm.SS_bal', f['all.SS_bal'], PUBF, f'[{name}].perm.flat.all.SS_bal', 'pp', 2)
    ci = r['perm']['SS_bal_all_ci']
    put(f'pub.{k}.perm.SS_bal.low', ci['low'], PUBF, f'[{name}].perm.SS_bal_all_ci.low (scenario-cluster bootstrap by clause order, 10,000)', 'pp', 2)
    put(f'pub.{k}.perm.SS_bal.high', ci['high'], PUBF, f'[{name}].perm.SS_bal_all_ci.high', 'pp', 2)
    put(f'pub.{k}.perm.ties', r['perm']['ties'], PUBF, f'[{name}].perm.ties', 'int', 0)
    put(f'pub.{k}.perm.asym', f['all.first.SS'] - f['all.second.SS'], PUBF, f'[{name}].perm first.SS - second.SS', 'pp', 2)
    rep = r['fresh']['report']
    put(f'pub.{k}.fresh.SS', rep['all']['SS'], PUBF, f'[{name}].fresh.report.all.SS', 'pp', 2)
    put(f'pub.{k}.fresh.SS.low', r['fresh']['SS_all_ci']['low'], PUBF, f'[{name}].fresh.SS_all_ci.low', 'pp', 2)
    put(f'pub.{k}.fresh.SS.high', r['fresh']['SS_all_ci']['high'], PUBF, f'[{name}].fresh.SS_all_ci.high', 'pp', 2)
    put(f'pub.{k}.fresh.SS_seen', rep['seen']['SS'], PUBF, f'[{name}].fresh.report.seen.SS', 'pp', 2)
    put(f'pub.{k}.fresh.SS_unseen', rep['unseen']['SS'], PUBF, f'[{name}].fresh.report.unseen.SS', 'pp', 2)
    for m in ('acc_O', 'acc_N', 'acc_S', 'flip_O_N', 'flip_O_S', 'flip_O_P'):
        put(f'pub.{k}.fresh.{m}', rep['all'][m], PUBF, f'[{name}].fresh.report.all.{m}', 'pct', 2)
    put(f'pub.{k}.fresh.ties', r['fresh']['ties'], PUBF, f'[{name}].fresh.ties', 'int', 0)
    nv = r['nevir']
    for sub in ('all', 'nested', 'complement'):
        put(f'pub.{k}.nevir.{sub}', nv[sub]['paired_accuracy'], PUBF, f'[{name}].nevir.{sub}.paired_accuracy', 'pct', 2)
        put(f'pub.{k}.nevir.{sub}.low', nv[sub]['bootstrap']['low'], PUBF, f'[{name}].nevir.{sub}.bootstrap.low (10,000)', 'pct', 2)
        put(f'pub.{k}.nevir.{sub}.high', nv[sub]['bootstrap']['high'], PUBF, f'[{name}].nevir.{sub}.bootstrap.high', 'pct', 2)
        put(f'pub.{k}.nevir.{sub}.perq', nv[sub]['per_query_accuracy'], PUBF, f'[{name}].nevir.{sub}.per_query_accuracy', 'pct', 2)
    put(f'pub.{k}.nevir.equal', nv['equal_bags']['paired_accuracy'], PUBF, f'[{name}].nevir.equal_bags.paired_accuracy (36 instances)', 'pct', 2)
    put(f'pub.{k}.nevir.gap', nv['complement']['paired_accuracy'] - nv['nested']['paired_accuracy'], PUBF, f'[{name}].nevir complement - nested paired accuracy', 'pp', 2)
    put(f'pub.{k}.nevir.ties', nv['instances_with_tie'], PUBF, f'[{name}].nevir.instances_with_tie', 'int', 0)
    if r.get('truncated_docs') is not None:
        put(f'pub.{k}.truncated_docs', r['truncated_docs'], PUBF, f'[{name}].truncated_docs (document encodings truncated, NegPairs and NevIR)', 'int', 0)
    st = r.get('settings') or {}
    for sk in ('query_maxlen', 'doc_maxlen', 'dim', 'query_length', 'document_length', 'max_seq_length', 'max_length', 'max_doc_tokens'):
        if st.get(sk) is not None:
            put(f'pub.{k}.set.{sk}', st[sk], PUBF, f'[{name}].settings.{sk}', 'int', 0)
put('pub.n_rows', len(PUB_ROWS), PUBF, 'systems scored (bge-m3 heads counted separately)', 'int', 0)
put('pub.n_checkpoints', len({r['spec']['repo'] for r in PUB_ROWS.values()}), PUBF, 'distinct checkpoints', 'int', 0)
put('pub.n_retrieval', len(PUB_RETR), PUBF, 'rows that are not LLM judges', 'int', 0)
put('pub.n_llm', len(PUB_LLM), PUBF, 'pointwise LLM judges', 'int', 0)
put('pub.n_li', len(PUB_LI), PUBF, 'pure late-interaction (MaxSim) systems', 'int', 0)
put('pub.n_ci_zero_retrieval', sum(1 for k in PUB_RETR if REG[f'pub.{k}.perm.SS_bal.low']['value'] <= 0 <= REG[f'pub.{k}.perm.SS_bal.high']['value']), PUBF,
    'retrieval rows whose SS_bal interval contains 0', 'int', 0)
put('pub.n_ci_zero_li', sum(1 for k in PUB_LI if REG[f'pub.{k}.perm.SS_bal.low']['value'] <= 0 <= REG[f'pub.{k}.perm.SS_bal.high']['value']), PUBF,
    'late-interaction rows whose SS_bal interval contains 0', 'int', 0)
put('pub.n_skipped', 1, PUB / 'status.json', 'llama-3.1-8b-instruct skipped (gated checkpoint, no token)', 'int', 0)
for grp, keys in (('retrieval', PUB_RETR), ('llm', PUB_LLM), ('li', PUB_LI)):
    put_range(f'pub.rng.{grp}.SS_bal', [f'pub.{k}.perm.SS_bal' for k in keys], f'pub.{grp}.perm.SS_bal over systems')
    put_range(f'pub.rng.{grp}.acc_N', [f'pub.{k}.perm.acc_N' for k in keys], f'pub.{grp}.perm.acc_N over systems', 'pct')
    put_range(f'pub.rng.{grp}.acc_O', [f'pub.{k}.perm.acc_O' for k in keys], f'pub.{grp}.perm.acc_O over systems', 'pct')
    put_range(f'pub.rng.{grp}.nested', [f'pub.{k}.nevir.nested' for k in keys], f'pub.{grp}.nevir.nested over systems', 'pct')
    put_range(f'pub.rng.{grp}.equal', [f'pub.{k}.nevir.equal' for k in keys], f'pub.{grp}.nevir.equal over systems', 'pct')
    put_range(f'pub.rng.{grp}.all', [f'pub.{k}.nevir.all' for k in keys], f'pub.{grp}.nevir.all over systems', 'pct')
    put_range(f'pub.rng.{grp}.fresh_SS', [f'pub.{k}.fresh.SS' for k in keys], f'pub.{grp}.fresh.SS over systems')
best_retr = max(PUB_RETR, key=lambda k: REG[f'pub.{k}.perm.SS_bal']['value'])
put('pub.best_retrieval.key', best_retr, PUBF, 'retrieval row with the largest SS_bal', 'text', 0)
put('pub.best_retrieval.label', REG[f'pub.{best_retr}.label']['value'], PUBF, 'its label', 'text', 0)
# precision comparison (bfloat16 logits of the first run vs float32 decisive logits)
for name, v in PUB_SUM['precision_check'].items():
    k = PUB_KEY[name]
    for which, tag in (('previous', 'prev'), ('rescored', 'new')):
        b = v[which]
        put(f'pub.prec.{k}.{tag}.nevir_all', b['nevir']['paired_all'], PUB / 'summary.json', f'precision_check.{name}.{which}.nevir.paired_all (2,000 replicates)', 'pct', 2)
        put(f'pub.prec.{k}.{tag}.nevir_nested', b['nevir']['paired_nested'], PUB / 'summary.json', f'precision_check.{name}.{which}.nevir.paired_nested', 'pct', 2)
        put(f'pub.prec.{k}.{tag}.nevir_ties', b['nevir']['instances_with_tie'], PUB / 'summary.json', f'precision_check.{name}.{which}.nevir.instances_with_tie', 'int', 0)
        put(f'pub.prec.{k}.{tag}.perm_SS_bal', b['perm']['SS'], PUB / 'summary.json', f'precision_check.{name}.{which}.perm.SS', 'pp', 2)
        put(f'pub.prec.{k}.{tag}.perm_ties', b['perm']['ties'], PUB / 'summary.json', f'precision_check.{name}.{which}.perm.ties', 'int', 0)
        put(f'pub.prec.{k}.{tag}.fresh_SS', b['fresh']['SS'], PUB / 'summary.json', f'precision_check.{name}.{which}.fresh.SS', 'pp', 2)
        put(f'pub.prec.{k}.{tag}.fresh_ties', b['fresh']['ties'], PUB / 'summary.json', f'precision_check.{name}.{which}.fresh.ties', 'int', 0)
    put(f'pub.prec.{k}.nevir_all.delta', v['rescored']['nevir']['paired_all'] - v['previous']['nevir']['paired_all'], PUB / 'summary.json', f'precision_check.{name}: rescored - previous nevir paired_all', 'pp', 2)
put('pub.prec.n_systems', len(PUB_SUM['precision_check']), PUB / 'summary.json', 'systems rescored', 'int', 0)
put('pub.prec.ties_prev_total', sum(v['previous']['nevir']['instances_with_tie'] for v in PUB_SUM['precision_check'].values()), PUB / 'summary.json', 'sum of previous NevIR ties over rescored systems', 'int', 0)
put('pub.prec.ties_new_total', sum(v['rescored']['nevir']['instances_with_tie'] for v in PUB_SUM['precision_check'].values()), PUB / 'summary.json', 'sum of rescored NevIR ties', 'int', 0)
A = jload(PUB / 'anchor_colbertv2.json')
put('pub.anchor.perm.agree', A['perm']['negpairs_decision_agreement_pct'], PUB / 'anchor_colbertv2.json', 'perm.negpairs_decision_agreement_pct (float32 vs archived bfloat16 M0-native)', 'pct', 2)
put('pub.anchor.fresh.agree', A['fresh']['negpairs_decision_agreement_pct'], PUB / 'anchor_colbertv2.json', 'fresh.negpairs_decision_agreement_pct', 'pct', 2)
put('pub.anchor.perm.maxdiff', A['perm']['negpairs_max_abs_diff'], PUB / 'anchor_colbertv2.json', 'perm.negpairs_max_abs_diff', 'float', 3)
put('pub.anchor.nevir.agree', A['perm']['nevir_comparison_agreement_pct'], PUB / 'anchor_colbertv2.json', 'nevir_comparison_agreement_pct', 'pct', 2)
put('pub.anchor.nevir.maxdiff', A['perm']['nevir_max_abs_diff'], PUB / 'anchor_colbertv2.json', 'nevir_max_abs_diff', 'float', 3)
CH = jload(PUB / 'scores/bge-m3.json')['checks']
put('pub.check.bgem3.pairs', CH['pairs'], PUB / 'scores/bge-m3.json', 'checks.pairs', 'int', 0)
for ck, kk in (('dense_vs_sentence_transformers_max_abs_diff', 'dense_st'), ('dense_vs_flagembedding_max_abs_diff', 'dense_fe'),
               ('sparse_vs_flagembedding_max_abs_diff', 'sparse_fe'), ('colbert_vs_flagembedding_max_abs_diff', 'colbert_fe')):
    put(f'pub.check.bgem3.{kk}', CH[ck], PUB / 'scores/bge-m3.json', f'checks.{ck}', 'sci', 1)
put('pub.check.bgem3.maxdiff', max(CH[c] for c in ('dense_vs_flagembedding_max_abs_diff', 'sparse_vs_flagembedding_max_abs_diff', 'colbert_vs_flagembedding_max_abs_diff')),
    PUB / 'scores/bge-m3.json', 'max over the three FlagEmbedding checks', 'sci', 1)
PENV = jload(PUB / 'run_environment.json')
put('pub.env.gpu', PENV[-1]['gpu'].split(',')[0], PUB / 'run_environment.json', '[-1].gpu (name)', 'text', 0)
put('pub.env.sessions', len(PENV), PUB / 'run_environment.json', 'sessions', 'int', 0)
put('pub.env.transformers', PENV[-1]['packages']['transformers'], PUB / 'run_environment.json', '[-1].packages.transformers', 'text', 0)
put('pub.env.sentence_transformers', PENV[-1]['packages_extra']['sentence-transformers'], PUB / 'run_environment.json', '[-1].packages_extra.sentence-transformers', 'text', 0)
put('pub.env.torch', PENV[-1]['torch'].split('+')[0], PUB / 'run_environment.json', '[-1].torch (without the CUDA suffix)', 'text', 0)
put('pub.env.cacsa2', PENV[-1]['cacsa2_version'], PUB / 'run_environment.json', '[-1].cacsa2_version', 'text', 0)
put('pub.env.minutes_total', sum(r['seconds'] for r in PUB_ROWS.values() if not r['name'].startswith('bge-m3/')) / 60 + PUB_ROWS['bge-m3/dense']['seconds'] / 60, PUBF, 'sum of seconds over checkpoints / 60', 'float', 0)
put('pub.nevir.n_pairs', 4 * 1383, 'cacsa2/nevir.py four_combos', '4 x 1,383 query-document pairs', 'int', 0)
put('pub.negpairs.n_pairs', 2 * (1096 + 960), 'cacsa2/public_systems.py', '2 options x (1,096 + 960) items', 'int', 0)
put('pub.llm.max_doc_tokens', 700, 'cacsa2/public_systems.py LLMPointwise', 'document truncation for the judges', 'int', 0)
put('pub.date', '21 and 22 October 2026', PUB / 'run_environment.json', 'session dates', 'text', 0)

# nested-subset interval against the complement value, per released system
_cov = 0
for name, r in PUB_ROWS.items():
    nv = r['nevir']
    if nv['nested']['bootstrap']['low'] <= nv['complement']['paired_accuracy'] <= nv['nested']['bootstrap']['high']:
        _cov += 1
put('pub.n_nested_ci_covers_complement', _cov, PUBF, 'systems whose nested-subset bootstrap interval contains their complement paired accuracy', 'int', 0)
# human check design (sheets built 22 October 2026; results added when the workbooks are returned)
HCK = CODE / 'human_check/sheets/human_check_key.csv'
_hc = list(csv.DictReader(open(HCK, encoding='utf-8')))
put('hc.perm.scenarios', len({r['scenario_id'] for r in _hc if r['set'] == 'perm'}), HCK, 'distinct Perm scenarios in the workbooks', 'int', 0)
put('hc.fresh.scenarios', len({r['scenario_id'] for r in _hc if r['set'] == 'fresh'}), HCK, 'distinct Fresh scenarios in the workbooks', 'int', 0)
put('hc.fresh.per_family', len({r['scenario_id'] for r in _hc if r['set'] == 'fresh' and r['family'] == 'affixal'}), HCK, 'Fresh scenarios per family (affixal count; equal by design)', 'int', 0)
put('hc.rows', len(_hc), HCK, 'rows per workbook', 'int', 0)
put('hc.annotators', 2, 'human_check/make_sheets.py', 'workbooks issued', 'int', 0)

# =============================================================================================== fine-tuned late-interaction checkpoints (revision step 3, 22 October 2026)
FT = S / 'finetune_v3/finetune_li'
FTS = jload(FT / 'summary.json')
FT_KEY = {'colbertv2': 'colbertv2', 'answerai-colbert-small-v1': 'answerai'}
FT_COND = {'baseline': 'base', 'finetuned/negpairs': 'np', 'finetuned/nevir': 'nv'}
FT_Q = {'perm.SS_bal': ('perm.SS_bal', 'pp'), 'perm.SS_bal_seen': ('perm.SS_bal_seen', 'pp'), 'perm.SS_bal_unseen': ('perm.SS_bal_unseen', 'pp'),
        'perm.flip_O_N': ('perm.flip_O_N', 'pct'), 'perm.flip_O_S': ('perm.flip_O_S', 'pct'), 'perm.joint_NS_acc': ('perm.joint_NS_acc', 'pct'),
        'fresh.SS': ('fresh.SS', 'pp'), 'fresh.SS_seen': ('fresh.SS_seen', 'pp'), 'fresh.SS_unseen': ('fresh.SS_unseen', 'pp'),
        'nevir.all': ('nevir.all', 'pct'), 'nevir.nested': ('nevir.nested', 'pct'), 'nevir.complement': ('nevir.complement', 'pct'),
        'nevir.equal': ('nevir.equal', 'pct'), 'nevir.ties': ('nevir.ties', 'int')}
for gk, g in FTS['groups'].items():
    name, cond = gk.split('|')
    m, c = FT_KEY[name], FT_COND[cond]
    put(f'ft.{m}.{c}.n_runs', g['n_runs'], FT / 'summary.json', f'groups[{gk}].n_runs', 'int', 0)
    for q, (qq, kind) in FT_Q.items():
        v = g[q]
        dec = 0 if kind == 'int' else 2
        put(f'ft.{m}.{c}.{qq}', v['mean'], FT / 'summary.json', f'groups[{gk}].{q}.mean (over seeds)', kind, dec)
        if g['n_runs'] > 1:
            put(f'ft.{m}.{c}.{qq}.min', v['min'], FT / 'summary.json', f'groups[{gk}].{q}.min', kind, dec)
            put(f'ft.{m}.{c}.{qq}.max', v['max'], FT / 'summary.json', f'groups[{gk}].{q}.max', kind, dec)
            for i, x in enumerate(v['values'], 1):
                put(f'ft.{m}.{c}.{qq}.s{i}', x, FT / 'summary.json', f'groups[{gk}].{q}.values[{i - 1}] (seed {g["seeds"][i - 1]})', kind, dec)
    sf = g['scifact.ndcg10']
    put(f'ft.{m}.{c}.scifact', sf['mean'], FT / 'summary.json', f'groups[{gk}].scifact.ndcg10.mean', 'pct', 2)
    if g['n_runs'] > 1:
        put(f'ft.{m}.{c}.scifact.min', min(sf['values']), FT / 'summary.json', f'groups[{gk}].scifact.ndcg10 min', 'pct', 2)
        put(f'ft.{m}.{c}.scifact.max', max(sf['values']), FT / 'summary.json', f'groups[{gk}].scifact.ndcg10 max', 'pct', 2)
        put(f'ft.{m}.{c}.dev_before', g['dev_before'], FT / 'summary.json', f'groups[{gk}].dev_before (selection metric of the released checkpoint)', 'pct', 2)
        for i, x in enumerate(g['selected_dev'], 1):
            put(f'ft.{m}.{c}.dev_after.s{i}', x, FT / 'summary.json', f'groups[{gk}].selected_dev[{i - 1}]', 'pct', 2)
        for i, x in enumerate(sf['values'], 1):
            put(f'ft.{m}.{c}.scifact.s{i}', x, FT / 'summary.json', f'groups[{gk}].scifact.ndcg10.values[{i - 1}]', 'pct', 2)
        put(f'ft.{m}.{c}.dev_after.mean', sum(g['selected_dev']) / len(g['selected_dev']), FT / 'summary.json', f'groups[{gk}].selected_dev mean', 'pct', 2)
        put(f'ft.{m}.{c}.dev_after.min', min(g['selected_dev']), FT / 'summary.json', f'groups[{gk}].selected_dev min', 'pct', 2)
        put(f'ft.{m}.{c}.dev_after.max', max(g['selected_dev']), FT / 'summary.json', f'groups[{gk}].selected_dev max', 'pct', 2)
        put(f'ft.{m}.{c}.epochs', ', '.join(str(e) for e in g['selected_epoch']), FT / 'summary.json', f'groups[{gk}].selected_epoch', 'text', 0)
        put(f'ft.{m}.{c}.epoch.min', min(g['selected_epoch']), FT / 'summary.json', f'groups[{gk}].selected_epoch min', 'int', 0)
        put(f'ft.{m}.{c}.epoch.max', max(g['selected_epoch']), FT / 'summary.json', f'groups[{gk}].selected_epoch max', 'int', 0)
        put(f'ft.{m}.{c}.train_minutes', sum(g['train_time_s']) / len(g['train_time_s']) / 60, FT / 'summary.json', f'groups[{gk}].train_time_s mean / 60', 'float', 1)
for m in FT_KEY.values():
    for c in ('np', 'nv'):
        for q, kind in (('perm.SS_bal', 'pp'), ('nevir.all', 'pp'), ('nevir.nested', 'pp'), ('scifact', 'pp'), ('fresh.SS', 'pp'), ('perm.flip_O_N', 'pp'), ('perm.flip_O_S', 'pp')):
            put(f'ft.{m}.{c}.{q}.delta', REG[f'ft.{m}.{c}.{q}']['value'] - REG[f'ft.{m}.base.{q}']['value'], FT / 'summary.json', f'ft.{m}.{c}.{q} - ft.{m}.base.{q}', kind, 2)
for m in FT_KEY.values():
    for c in ('np', 'nv'):
        put(f'ft.{m}.{c}.scifact.drop', REG[f'ft.{m}.base.scifact']['value'] - REG[f'ft.{m}.{c}.scifact']['value'], FT / 'summary.json', f'ft.{m}.base.scifact - ft.{m}.{c}.scifact (decrease)', 'pp', 2)
for e in ENCS7:
    put(f'fresh.A5mA1.unseen.{e}', REG[f'sm.fresh.{e}.A5.unseen.SS']['value'] - REG[f'sm.fresh.{e}.A1.unseen.SS']['value'], MAIN / 'seed_means_fresh.json', f'[{e}].A5.unseen.SS - A1.unseen.SS (seed means)', 'pp', 2)
put_range('rng.fresh.A5mA1.unseen.all7', [f'fresh.A5mA1.unseen.{e}' for e in ENCS7], 'A5 minus A1 unseen-family SS on Fresh over all encoders')
put_range('ft.rng.np.scifact.drop', [f'ft.{m}.np.scifact.drop' for m in FT_KEY.values()], 'SciFact decrease after NegPairs fine-tuning over models')
put_range('ft.rng.nv.scifact.drop', [f'ft.{m}.nv.scifact.drop' for m in FT_KEY.values()], 'SciFact decrease after NevIR fine-tuning over models')
put_range('ft.rng.np.SS_bal', [f'ft.{m}.np.perm.SS_bal' for m in FT_KEY.values()], 'ft.{model}.np.perm.SS_bal over the two models')
put_range('ft.rng.np.SS_bal.seeds', [f'ft.{m}.np.perm.SS_bal.s{i}' for m in FT_KEY.values() for i in (1, 2, 3)], 'per-seed ft.{model}.np.perm.SS_bal over models and seeds')
put_range('ft.rng.nv.SS_bal', [f'ft.{m}.nv.perm.SS_bal' for m in FT_KEY.values()], 'ft.{model}.nv.perm.SS_bal over the two models')
put_range('ft.rng.np.scifact.delta', [f'ft.{m}.np.scifact.delta' for m in FT_KEY.values()], 'SciFact change after NegPairs fine-tuning over models')
put_range('ft.rng.nv.scifact.delta', [f'ft.{m}.nv.scifact.delta' for m in FT_KEY.values()], 'SciFact change after NevIR fine-tuning over models')
FTD = FTS['data']
put('ft.v1.train_items', FTD['v1']['train_items'], FT / 'data_audit.json', 'v1.train_items', 'int', 0)
put('ft.v1.train_scenarios', FTD['v1']['train_scenarios'], FT / 'data_audit.json', 'v1.train_scenarios', 'int', 0)
put('ft.v1.dev_items', FTD['v1']['dev_items'], FT / 'data_audit.json', 'v1.dev_items', 'int', 0)
put('ft.nevir.train', FTD['nevir']['train'], FT / 'data_audit.json', 'nevir.train (instances)', 'int', 0)
put('ft.nevir.validation', FTD['nevir']['validation'], FT / 'data_audit.json', 'nevir.validation (instances)', 'int', 0)
put('ft.scifact.n_corpus', FTD['scifact']['n_corpus'], FT / 'data_audit.json', 'scifact.n_corpus', 'int', 0)
put('ft.scifact.n_queries', FTD['scifact']['n_test_queries'], FT / 'data_audit.json', 'scifact.n_test_queries', 'int', 0)
put('ft.scifact.n_qrels', FTD['scifact']['n_qrels'], FT / 'data_audit.json', 'scifact.n_qrels', 'int', 0)
put('ft.scifact.sha_short', FTD['scifact']['zip_sha256'][:8], FT / 'data_audit.json', 'scifact.zip_sha256[:8]', 'text', 0)
FTE = jload(FT / 'run_environment.json')[-1]
put('ft.env.gpu', FTE['gpu'].split(',')[0], FT / 'run_environment.json', 'gpu (name)', 'text', 0)
put('ft.cfg.epochs', FTE['config']['epochs'], FT / 'run_environment.json', 'config.epochs', 'int', 0)
put('ft.cfg.patience', FTE['config']['patience'], FT / 'run_environment.json', 'config.patience', 'int', 0)
put('ft.cfg.warmup', FTE['config']['warmup'], FT / 'run_environment.json', 'config.warmup', 'float', 2)
put('ft.cfg.seeds', ', '.join(str(x) for x in FTE['config']['seeds']), FT / 'run_environment.json', 'config.seeds', 'text', 0)
put('ft.cfg.n_seeds', len(FTE['config']['seeds']), FT / 'run_environment.json', 'config.seeds (count)', 'int', 0)
for mm in FTE['config']['models']:
    put(f"ft.{FT_KEY[mm['name']]}.lr", mm['lr'], FT / 'run_environment.json', f"config.models[{mm['name']}].lr", 'sci', 0)
    put(f"ft.{FT_KEY[mm['name']]}.batch", mm['batch'], FT / 'run_environment.json', f"config.models[{mm['name']}].batch", 'int', 0)
put('ft.n_runs', sum(g['n_runs'] for k, g in FTS['groups'].items() if not k.endswith('baseline')), FT / 'summary.json', 'fine-tuning runs', 'int', 0)
put('ft.cfg.lr_colbertv2', 1e-5, FT / 'run_environment.json', 'config.models[colbertv2].lr', 'sci', 0)
put('ft.cfg.lr_answerai', 2e-5, FT / 'run_environment.json', 'config.models[answerai].lr', 'sci', 0)
put('ft.cfg.grad_clip', 1.0, 'cacsa2/finetune_li.py ColBERTTrainer', 'gradient clipping', 'float', 1)
put('ft.cfg.weight_decay', 0.01, 'cacsa2/finetune_li.py ColBERTTrainer', 'weight decay', 'float', 2)
put('ft.cfg.n_boot', FTS['n_boot'], FT / 'summary.json', 'n_boot per run', 'int', 0)

# =============================================================================================== constants of the manifest and the code
put('n.seeds', 10, 'CACSA_manifest_v3_0.md, section 6', 'seeds', 'int', 0)
for i, s in enumerate(SEEDS, 1):
    put(f'seed.{i}', s, 'CACSA_manifest_v3_0.md, section 6', f'seed {i}', 'id', 0)
put('n.encoders', 7, 'CACSA_manifest_v3_0.md, section 5.1', 'encoders', 'int', 0)
put('n.contextual', 5, 'CACSA_manifest_v3_0.md, section 5.1', 'contextual encoders', 'int', 0)
put('n.static', 2, 'CACSA_manifest_v3_0.md, section 5.1', 'static encoders', 'int', 0)
put('n.trained_conditions', 8, 'CACSA_manifest_v3_0.md, section 12', 'A1-A6, D1, M1', 'int', 0)
put('stat.t_crit_df9', T95_9, 'scipy.stats.t.ppf(0.975, 9)', 't quantile', 'float', 3)
put('stat.conf_R2R3', 100 * (1 - 0.05 / 6), 'CACSA_manifest_v3_0.md, section 8.1', '100 x (1 - 0.05/6)', 'pct', 2)
put('stat.conf_R8', 100 * (1 - 0.05 / 5), 'CACSA_manifest_v3_0.md, section 8.1', '100 x (1 - 0.05/5)', 'pct', 2)
put('cfg.proj', 128, 'cacsa2/model.py', 'proj', 'int', 0)
put('cfg.hidden', 256, 'cacsa2/model.py', 'hidden', 'int', 0)
put('cfg.hidden4', 64, 'cacsa2/model.py', 'hidden // 4', 'int', 0)
put('cfg.dropout', 0.1, 'cacsa2/model.py', 'dropout', 'float', 1)
put('cfg.weight_decay', 0.01, 'CACSA_manifest_v3_0.md, section 6', 'weight decay', 'float', 2)
put('cfg.batch', 16, 'CACSA_manifest_v3_0.md, section 6', 'batch', 'int', 0)
put('cfg.epochs', 30, 'CACSA_manifest_v3_0.md, section 6', 'max epochs', 'int', 0)
put('cfg.patience', 5, 'CACSA_manifest_v3_0.md, section 6', 'patience', 'int', 0)
put('cfg.grad_clip', 1.0, 'CACSA_manifest_v3_0.md, section 6', 'gradient clipping', 'float', 1)
put('cfg.max_length_negpairs', 128, 'CACSA_manifest_v3_0.md, section 5.1', 'max tokens NegPairs', 'int', 0)
put('cfg.max_length_nevir', 512, 'CACSA_manifest_v3_0.md, section 5.1', 'max tokens NevIR', 'int', 0)
put('cfg.c1_lr', 2e-5, 'CACSA_manifest_v3_0.md, section 6', 'C1 learning rate', 'sci', 2)
put('cfg.c1_epochs', 5, 'runs/C1-mcft/*/run_summary.json config.epochs', 'C1 epochs', 'int', 0)
put('cfg.c1_max_length', 128, 'runs/C1-mcft/*/run_summary.json config.max_length', 'C1 max length', 'int', 0)
put('cfg.boot_n', 10000, 'CACSA_manifest_v3_0.md, section 9', 'bootstrap replicates', 'int', 0)
put('cfg.boot_seed', 20261003, 'CACSA_manifest_v3_0.md, section 9', 'bootstrap seed', 'id', 0)
put('cfg.probe_seed', 20261031, 'CACSA_manifest_v3_0.md, section 7.5', 'probing fold seed', 'id', 0)
put('cfg.probe_max_iter', 1000, 'CACSA_manifest_v3_0.md, section 7.5', 'lbfgs iterations', 'int', 0)
put('cfg.probe_folds', 5, 'CACSA_manifest_v3_0.md, section 7.5', 'outer folds', 'int', 0)
put('cfg.zipf_affix_min', 2.0, 'cacsa2/perm_build.py AFFIX_MIN_ZIPF', 'Zipf threshold for affixal forms', 'float', 1)
put('cfg.zipf_un_min', 3.0, 'cacsa2/perm_build.py UN_BASE_MIN_ZIPF', 'Zipf threshold for un- bases', 'float', 1)
put('cfg.perm_max_words', 30, 'cacsa2/perm_build.py MAX_WORDS', 'max words per arm', 'int', 0)
put('cfg.dup_jaccard', 0.6, 'CACSA_manifest_v3_0.md, section 4.6', 'Jaccard threshold', 'float', 1)
put('cfg.audit_pct', 10, 'CACSA_manifest_v3_0.md, section 4.5', 'audit sample share', 'int', 0)
put('cfg.v1_split_train', 60, 'CACSA_manifest_v2_0.md, section 4', 'train share within seen families', 'int', 0)
put('cfg.v1_split_dev', 15, 'CACSA_manifest_v2_0.md, section 4', 'dev share', 'int', 0)
put('cfg.v1_split_test', 25, 'CACSA_manifest_v2_0.md, section 4', 'test share', 'int', 0)
put('cfg.safety_after', 72, 'CACSA_manifest_v3_0.md, section 4.6', 'operational safeguard', 'int', 0)
put('cfg.perm_cap_stratum', 38, 'CACSA_manifest_v3_0.md, section 4.3', 'cap per family and clause order', 'int', 0)
put('cfg.fresh_cap_family', 60, 'CACSA_manifest_v3_0.md, section 4.2', 'cap per family', 'int', 0)
put('cfg.qwen_margin_small', 0.5, 'np_v3_main_audit.md / negpairs_v3_audit.md section 4 (descriptive threshold for small letter-logit margins)', 'margin threshold', 'float', 1)
put('cfg.nevir_margin', 1e-9, 'CACSA_manifest_v3_0.md, section 4.8', 'margin for static scorers', 'sci', 2)
put('cfg.qwen_params_b', 7.6, 'CACSA_manifest_v3_0.md, section 4.4', 'Qwen parameters (billions)', 'float', 1)
put('v2.lock.sha_short', 'cc172368', 'CACSA_manifest_v3_0.md (status line): SHA-256 of manifest v2.0', 'sha256[:8]', 'text', 0)
put('v3.manifest_bytes', 72140, 'CACSA_manifest_v3_0.md', 'bytes', 'int', 0)
put('v2.sm.C1.C1.n_seeds_v2', 5, 'CACSA_manifest_v2_0.md, section 6', 'seeds of v2.0', 'int', 0)

# =============================================================================================== v2.0 manuscript registry (imported)
OLDREG = jload(OLD)
for k, e in OLDREG.items():
    key = f'v2.{k}'
    if key in REG:
        continue
    REG[key] = dict(e)
    REG[key]['src'] = 'v2.0 registry: ' + e['src']

# =============================================================================================== write
OUT.mkdir(exist_ok=True)
with open(OUT / 'registry.json', 'w', encoding='utf-8') as fh:
    json.dump(REG, fh, indent=0, ensure_ascii=False)
with open(OUT / 'traceability.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['key', 'value', 'kind', 'decimals', 'source_file', 'how'])
    for k, e in REG.items():
        w.writerow([k, e['value'], e['kind'], e['dec'], e['src'], e['how']])
print(f'{len(REG)} registry entries written ({sum(1 for k in REG if not k.startswith("v2."))} new, {sum(1 for k in REG if k.startswith("v2."))} imported from v2.0)')
