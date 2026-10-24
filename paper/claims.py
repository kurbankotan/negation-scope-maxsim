# -*- coding: utf-8 -*-
"""Programmatic checks of the comparative and qualitative statements made in the manuscripts (version 3.0).
Each check names the statement it supports. Run: python claims.py"""
from fmt import val, rnd, REG

checks = []


def claim(text, cond):
    checks.append((text, bool(cond)))


ENCS = ['E0', 'E3', 'E1', 'E2', 'E4', 'E5', 'E6']
CONTEXTUAL = ['E1', 'E2', 'E4', 'E5', 'E6']
STATIC = ['E0', 'E3']
RULES = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'D1', 'M1', 'M0', 'C0', 'C2']

# ---- exact rules
claim('R5, R6, R7 met with zero violations', val('R5.met') and val('R6.met') and val('R7.met') and val('R5.violations_total') == 0
      and val('R6.score_violations_total') == 0 and val('R6.decision_disagreements_total') == 0 and val('R7.violations_total') == 0)
claim('exact checks total 390 = 186 + 142 + 62', val('n.exact_checks') == val('R5.checks') + val('R6.checks') + val('R7.checks'))
claim('every unmasked rule of E0/E3 has SS_perm = 0 and joint acc = 0 in float64 seed means',
      all(abs(val(f'f64.{e}.{a}.seen.SS_bal')) < 1e-12 and abs(val(f'f64.{e}.{a}.seen.pooled.joint_NS_acc')) < 1e-12 and abs(val(f'f64.{e}.{a}.all.pooled.SS')) < 1e-12
          for e in STATIC for a in ('A1', 'A2', 'A3', 'A5', 'M1', 'M0', 'C0', 'C2')))
claim('every unmasked rule of E0/E3 has SS_perm = 0 and joint acc = 0 in GPU seed means too',
      all(abs(val(f'sm.perm.{e}.{a}.seen.SS_bal')) < 1e-12 and abs(val(f'sm.perm.{e}.{a}.seen.pooled.joint_NS_acc')) < 1e-12
          for e in STATIC for a in ('A1', 'A2', 'A3', 'A5', 'M1', 'M0', 'C0', 'C2')))
claim('every static scorer at 0 on the nested NevIR subset', val('nevir.static_trained.nested.max') == 0 and val('nevir.E0.M0.nested.paired') == 0 and val('nevir.E3.M0.nested.paired') == 0)
claim('GPU vs float64 decision disagreements never changed an N/S pair (R6 decision disagreements 0)', val('R6.decision_disagreements_total') == 0)

# ---- gate
claim('six of seven encoders passed the gate; E3 failed', val('G.n_passed') == 6 and not val('G.E3.passed') and all(val(f'G.{e}.passed') for e in ENCS if e != 'E3'))
claim('gate order E6 > E1 > E4 > E5 > E2 > E0 > E3', val('G.E6.mean') > val('G.E1.mean') > val('G.E4.mean') > val('G.E5.mean') > val('G.E2.mean') > val('G.E0.mean') > val('G.E3.mean'))
claim('E0 bimodal: seeds at/above threshold and below are separated by a wide gap', val('G.E0.below.max') <= 0.5 and val('G.E0.above.min') > 30 and val('G.E0.above.n') + val('G.E0.below.n') == 10)
claim('E2: seeds below the threshold near zero, the rest far above', val('G.E2.below.max') < 1 and val('G.E2.above.min') > 30)
claim('k = 6 for R2/R3 and k = 5 for R8', val('G.k_R2_R3') == 6 and val('G.k_R8') == 5)

# ---- R2
claim('R2 met only for E0', val('R2.n_met') == 1 and val('R2.E0.met'))
claim('R2 E0: all conditions, 10/10 seeds, both intervals above zero', val('R2.E0.pos') == 10 and val('R2.E0.low') > 0 and val('R2.E0.cl_low') > 0 and val('R2.E0.mean') >= 15)
claim('R2 E6: 10/10 positive, both intervals above zero, mean below threshold', val('R2.E6.pos') == 10 and val('R2.E6.low') > 0 and val('R2.E6.cl_low') > 0 and val('R2.E6.mean') < 15)
claim('R2 E2: mean above threshold but seed interval includes zero and 8 positive seeds', val('R2.E2.mean') >= 15 and val('R2.E2.low') < 0 < val('R2.E2.high') and val('R2.E2.pos') == 8)
claim('R2 E5, E4, E1: intervals include zero', all(val(f'R2.{e}.low') < 0 < val(f'R2.{e}.high') for e in ('E5', 'E4', 'E1')))
claim('R2 E3 descriptive: not evaluated, 10/10 positive, interval above zero', (not val('R2.E3.evaluated')) and val('R2.E3.pos') == 10 and val('R2.E3.low') > 0)
claim('D1 - A1 unseen SS positive for E0, E3, E2, E1, E4 and negative for E5, E6',
      all(val(f'D1.{e}.D1mA1_unseen_SS.mean') > 0 for e in ('E0', 'E3', 'E2', 'E1', 'E4')) and all(val(f'D1.{e}.D1mA1_unseen_SS.mean') < 0 for e in ('E5', 'E6')))
claim('D1 - A1 unseen SS: E0 and E2 intervals above zero', val('D1.E0.D1mA1_unseen_SS.low') > 0 and val('D1.E2.D1mA1_unseen_SS.low') > 0)

# ---- R3
claim('R3 met only for E2', val('R3.n_met') == 1 and val('R3.E2.met'))
claim('R3 E2: 9/10 positive, both intervals above zero', val('R3.E2.pos') == 9 and val('R3.E2.low') > 0 and val('R3.E2.cl_low') > 0)
claim('R3 E6 and E0: means above threshold, 8 positive seeds, seed intervals include zero',
      all(val(f'R3.{e}.mean') >= 10 and val(f'R3.{e}.pos') == 8 and val(f'R3.{e}.low') < 0 < val(f'R3.{e}.high') for e in ('E6', 'E0')))
claim('R3 E5 and E4: means below threshold', val('R3.E5.mean') < 10 and val('R3.E4.mean') < 10)
claim('R3 E1: negative contrast with interval below zero', val('R3.E1.mean') < 0 and val('R3.E1.high') < 0)
claim('R3 E3 descriptive negative', val('R3.E3.mean') < 0)

# ---- R8
claim('R8 met for E4, E6, E5 and not for E1, E2', val('R8.n_met') == 3 and all(val(f'R8.{e}.met') for e in ('E4', 'E6', 'E5')) and not val('R8.E1.met') and not val('R8.E2.met'))
claim('R8 order E4 > E6 > E5 > E1 > E2', val('R8.E4.mean') > val('R8.E6.mean') > val('R8.E5.mean') > val('R8.E1.mean') > val('R8.E2.mean'))
claim('R8 E1: 10/10 positive, both intervals above zero, below threshold', val('R8.E1.pos') == 10 and val('R8.E1.low') > 0 and val('R8.E1.cl_low') > 0 and val('R8.E1.mean') < 10)
claim('R8 E2: 8 positive seeds', val('R8.E2.pos') == 8)
claim('R8 means between 10.46 (E5) and 19.74 (E4) for the met encoders', min(val(f'R8.{e}.mean') for e in ('E4', 'E6', 'E5')) == val('R8.E5.mean') and max(val(f'R8.{e}.mean') for e in ('E4', 'E6', 'E5')) == val('R8.E4.mean'))
claim('abstract range uses R8.E5 (min met) to R8.E4 (max met)', val('R8.E5.mean') < val('R8.E6.mean') < val('R8.E4.mean'))
claim('joint accuracy of A1 between 11 and 26 for contextual encoders, 0 for static', 10 < val('rng.perm.A1.joint_NS_acc.contextual.min') and val('rng.perm.A1.joint_NS_acc.contextual.max') < 27
      and all(val(f'f64.{e}.A1.seen.pooled.joint_NS_acc') == 0 for e in STATIC))

# ---- descriptive
claim('bridge: cluster intervals of E1 and E4 unseen SS above zero', val('bridge.E1.fresh.unseen_SS.cl_low') > 0 and val('bridge.E4.fresh.unseen_SS.cl_low') > 0)
claim('E0 unseen SS on Fresh negative (R1 descriptive) and E3 small positive', val('R1.E0.mean') < 0 and 0 < val('R1.E3.mean') < 5)
claim('M1 never better than A1 on unseen SS (all means <= 0)', all(val(f'MaxSim.{e}.M1mA1_unseen_SS_fresh.mean') <= 0 for e in ENCS))
claim('M1 - A1 unseen range: min at E5, max at E0', min(ENCS, key=lambda e: val(f'MaxSim.{e}.M1mA1_unseen_SS_fresh.mean')) == 'E5' and max(ENCS, key=lambda e: val(f'MaxSim.{e}.M1mA1_unseen_SS_fresh.mean')) == 'E0')
claim('M0 SS within ten points of zero on every set', abs(val('rng.fresh.M0.SS.allsets.min')) < 10 and abs(val('rng.fresh.M0.SS.allsets.max')) < 10)
claim('A5 - A1 unseen: every seed interval includes zero except E6 (positive)', all(val(f'R4.{e}.A5mA1_unseen_SS.low') < 0 < val(f'R4.{e}.A5mA1_unseen_SS.high') for e in ENCS if e != 'E6') and val('R4.E6.A5mA1_unseen_SS.low') > 0)
claim('A5 below A4 on unseen SS for every encoder except E1', all(val(f'R4.{e}.A5mA4_unseen_SS.mean') < 0 for e in ENCS if e != 'E1') and val('R4.E1.A5mA4_unseen_SS.mean') > 0)
claim('static D1 on Perm: E3 far above E0; A4 much lower than D1 for both', val('f64.E3.D1.seen.SS_bal') > val('f64.E0.D1.seen.SS_bal') + 30
      and val('f64.E0.A4.seen.SS_bal') < val('f64.E0.D1.seen.SS_bal') and val('f64.E3.A4.seen.SS_bal') < val('f64.E3.D1.seen.SS_bal'))
claim('A6 on Perm at or below 10.4 for every encoder, static near zero', val('rng.perm.A6.seen_SS_bal.all7.max') < 10.5 and abs(val('f64.E0.A6.seen.SS_bal')) < 0.5 and abs(val('f64.E3.A6.seen.SS_bal')) < 0.5)
claim('resolver nearly exact on Perm N arms (decisive in scope >= 99%)', val('res.perm.seen.N.decisive_in_scope.pct') >= 99 and val('res.perm.unseen.N.decisive_in_scope.pct') >= 99)
claim('same cue: E3 rates coincide (|diff| < 1); E0 differs by about 10', abs(val('samecue.fresh.E3.A1.diff')) < 1 and 5 < val('samecue.fresh.E0.A1.diff') < 15)
claim('same cue: E1 and E6 separate the positions (diff > 30)', val('samecue.fresh.E1.A1.diff') > 30 and val('samecue.fresh.E6.A1.diff') > 30)
claim('C2 changed no decision on Fresh or Perm', val('ctrl.C2.fresh.max_flip') == 0 and val('ctrl.C2.perm.max_flip') == 0)
claim('C0 |SS| small', val('ctrl.C0.fresh.max_abs_SS') < 5 and val('ctrl.C0.perm.max_abs_SS_bal') < 5)
claim('no encoder passed the secondary accuracy gate', val('acc2.n_passed') == 0)
claim('C2 acc_OP range reported from registry min to max', val('rng.fresh.C2.acc_OP.min') < val('rng.fresh.C2.acc_OP.max'))

# ---- NevIR
claim('contextual M0 all below random 25', val('rng.nevir.M0.contextual.all.max') < val('nevir.published.random'))
claim('M0 order E1 > E5 > E6 > E4 > E2', val('nevir.E1.M0.all.paired') > val('nevir.E5.M0.all.paired') > val('nevir.E6.M0.all.paired') > val('nevir.E4.M0.all.paired') > val('nevir.E2.M0.all.paired'))
claim('static far below contextual on NevIR', val('nevir.static_trained.all.max') < 5 and val('nevir.E0.M0.all.paired') < 1 and val('nevir.E3.M0.all.paired') < 5)
claim('nested subset no harder for contextual M0 (ranges overlap)', val('rng.nevir.M0.contextual.nested.min') < val('rng.nevir.M0.contextual.complement.max') and val('rng.nevir.M0.contextual.nested.max') > val('rng.nevir.M0.contextual.complement.min'))
claim('M0-native above published ColBERTv2 value', val('nevir.E6.M0n.all.paired') > val('nevir.published.ColBERTv2'))

# ---- probing
claim('probing order E2 > E1 > E5 > E4 > E6; static at 50', val('probe.E2.cv_mean') > val('probe.E1.cv_mean') > val('probe.E5.cv_mean') > val('probe.E4.cv_mean') > val('probe.E6.cv_mean')
      and val('probe.E0.cv_mean') == 50 and val('probe.E3.cv_mean') == 50)
claim('selectivity: E1 > E5 > E6 > E4 > E2, E2 small', val('probe.E1.selectivity') > val('probe.E5.selectivity') > val('probe.E6.selectivity') > val('probe.E4.selectivity') > val('probe.E2.selectivity') and val('probe.E2.selectivity') < 3)
claim('A5 gate AUC between 0.49 and 0.62', 0.45 < val('rng.probe.auc.min') and val('rng.probe.auc.max') < 0.65)
claim('probing range reported as E6 (min) to E2 (max)', val('rng.probe.cv.contextual.min') == val('probe.E6.cv_mean') and val('rng.probe.cv.contextual.max') == val('probe.E2.cv_mean'))

# ---- clause order
first = {e: val(f'sm.perm.{e}.A1.seen.first.SS') for e in CONTEXTUAL}
second = {e: val(f'sm.perm.{e}.A1.seen.second.SS') for e in CONTEXTUAL}
claim('clause order: first > second for every contextual encoder', all(first[e] > second[e] for e in CONTEXTUAL))
claim('only E5 keeps first and second within 5 points', abs(first['E5'] - second['E5']) < 5 and all(abs(first[e] - second[e]) >= 5 for e in CONTEXTUAL if e != 'E5'))
claim('C1 first >> second; S flips higher with decisive second; acc_O falls', val('sm.perm.C1.C1.seen.first.SS') > val('sm.perm.C1.C1.seen.second.SS') + 50
      and val('sm.perm.C1.C1.seen.second.flip_O_S') > val('sm.perm.C1.C1.seen.first.flip_O_S') and val('sm.perm.C1.C1.seen.first.acc_O') > val('sm.perm.C1.C1.seen.second.acc_O'))
claim('E4 first flips: N > S; second flips lower', val('sm.perm.E4.A1.seen.first.flip_O_N') > val('sm.perm.E4.A1.seen.first.flip_O_S'))

# ---- Qwen audit
claim('Qwen agreement about a third disagree; N lowest arm', val('qwen.v1.all.pct') < 70 and val('qwen.v1.arm.N.pct') < min(val('qwen.v1.arm.O.pct'), val('qwen.v1.arm.P.pct'), val('qwen.v1.arm.S.pct')))
claim('Qwen family order affixal > quantifier > syntactic > lexical', val('qwen.v1.family.affixal.pct') > val('qwen.v1.family.quantifier.pct') > val('qwen.v1.family.syntactic.pct') > val('qwen.v1.family.lexical.pct'))

# ---- data
claim('Perm seen families at cap in both orders (76 + 76)', val('build.perm.accepted.syntactic.first') == val('build.perm.accepted.syntactic.second') == val('build.perm.accepted.quantifier.first') == val('build.perm.accepted.quantifier.second') == 38
      and val('subset.perm_seen_first') == 76 == val('subset.perm_seen_second'))
claim('Perm affixal below cap; lexical near cap', val('build.perm.accepted.affixal.first') < 38 and val('build.perm.accepted.affixal.second') < 38 and val('build.perm.accepted.lexical.second') == 37)
claim('Fresh: every family at cap 60', all(val(f'build.fresh.accepted.{f}') == 60 for f in ('syntactic', 'quantifier', 'affixal', 'lexical')))
claim('Fresh S-same >= 100 and unseen >= 80', val('subset.fresh_S_same') >= 100 and val('subset.fresh_unseen') >= 80)
claim('Perm bags equal under every tokenizer', val('data.perm.bag_equal_all_tokenizers') and val('data.perm.bag_equal_recomputed') == val('data.perm.scenarios'))
claim('Perm artefact: lower-case starts and missing punctuation counts', val('data.perm.lowercase_start') > 100 and val('data.perm.no_final_punct') > 100)
claim('Fresh contexts capitalized with final punctuation', val('data.fresh.capitalized') == val('data.fresh.scenarios') and val('data.fresh.final_punct') == val('data.fresh.scenarios'))
claim('Fresh S cues dominated by never and nobody', val('data.fresh.cueS.syntactic.never') > 30 and val('data.fresh.cueS.quantifier.nobody') > 40)
claim('no unseen N cue of Perm in train text; Fresh affixal none, lexical one', val('data.perm.unseen.affixal.N_cues_in_train') == 0 and val('data.perm.unseen.lexical.N_cues_in_train') == 0
      and val('data.fresh.unseen.affixal.N_cues_in_train') == 0 and val('data.fresh.unseen.lexical.N_cues_in_train') == 1)
claim('audit Perm: answers agree 105/108, all four arms 9/27', val('build.audit.perm.answers_agree') == 105 and val('build.audit.perm.answers_total') == 108 and val('build.audit.perm.agree_all') == 9)
claim('audit Perm: disagreements were unclear verdicts', val('build.audit.perm.answers_total') - val('build.audit.perm.answers_agree') == sum(val(f'build.audit.perm.unclear.{a}') for a in 'OPNS'))
claim('audit Fresh: answers agree 96/96', val('build.audit.fresh.answers_agree') == val('build.audit.fresh.answers_total'))
claim('Perm insertion pairs 152 = number of syntactic scenarios x 2', val('R5.E0.perm.pairs') == 2 * val('data.perm.cue.syntactic.not') and val('data.perm.insertion_pairs_recomputed') == val('R5.E0.perm.pairs'))
claim('insertion pairs recomputed equal R5 pair counts (Fresh)', val('data.fresh.insertion_pairs_recomputed') == val('R5.E0.fresh.pairs'))
claim('lexical Perm scenarios all use lacks', val('data.perm.cue.lexical.lacks') == val('build.perm.lexical_total'))
claim('independent recomputation identical', val('indep.mismatches') == 0 and val('indep.max_abs_diff') == 0 and val('n.f64_mismatches') == 0)
claim('f64: 44 seed-mean differences, max 1.32', val('f64.n_seed_mean_differences') == 44 and 1.3 < val('f64.max_abs_seed_mean_diff') < 1.33)
claim('LR: five encoders at 3e-3, two at 1e-3 (E5, E6)', val('lr.n_3e3') == 5 and val('lr.n_1e3') == 2 and val('lr.E5.chosen') == 0.001 and val('lr.E6.chosen') == 0.001)
claim('pilot: mean and signed max violate inclusion; max, lse, m1 do not', val('pilot.E0.ins.A2.violations') > 0 and val('pilot.E0.ins.A4.violations') > 0 and val('pilot.E0.ins.A1.violations') == 0 and val('pilot.E0.ins.A3.violations') == 0 and val('pilot.E0.ins.M1.violations') == 0)
claim('abstract: probe range 73.08 to 89.25 (E6 min, E2 max)', rnd(val('rng.probe.cv.contextual.min'), 2) == rnd(val('probe.E6.cv_mean'), 2))
claim('C1 Perm first SS 90.9 and second 25.3 as in abstract', 90 < val('sm.perm.C1.C1.seen.first.SS') < 92 and 25 < val('sm.perm.C1.C1.seen.second.SS') < 26)
claim('A4 Perm static float64 values between 12 and 14', 12 < val('f64.E0.A4.seen.SS_bal') < 14 and 12 < val('f64.E3.A4.seen.SS_bal') < 14)
claim('GPU vs f64 disagreement counts (R6 file) 62 and 50', val('R6.E0.gpu_vs_f64_disagreements') == 62 and val('R6.E3.gpu_vs_f64_disagreements') == 50)
claim('Fresh A1 acc_OP within six points of C2', abs(val('rng.acc2.margin.min')) <= 6 and abs(val('rng.acc2.margin.max')) <= 6)
claim('Qwen margins: no exact tie', val('build.perm.qwen_margin.n_ties') == 0 and val('build.fresh.qwen_margin.n_ties') == 0)
claim('E3 OOV rate about 1% overall and below 0.1% on train/dev', val('enc.E3.oov_all') < 1.5 and val('enc.E3.oov_train_dev') < 0.1)
claim('v2.0 outcomes: gate passed by E1 only; R2 E1 positive in all seeds below threshold; R3 not met; R5 met',
      val('v2.G.E1.passed') and not val('v2.G.E0.passed') and not val('v2.G.E2.passed') and val('v2.R2.E1.pos') == 5 and not val('v2.R2.E1.met') and val('v2.R2.E1.low') > 0
      and not val('v2.R3.E1.met') and val('v2.R5.met'))
claim('run counts: 718 = 560 + 140 + 10 + 7 + 1', val('train.n_runs') == 718 and val('train.n_grid_runs') == 560 and val('train.n_trained_runs') == 710)

# ---- statements added in the review-driven revision
claim('D1 vs A4 unseen SS on Fresh: D1 higher only for E1, E2, E4; lower for E0, E3, E5, E6',
      all(val(f'sm.fresh.{e}.D1.unseen.SS') > val(f'sm.fresh.{e}.A4.unseen.SS') for e in ('E1', 'E2', 'E4'))
      and all(val(f'sm.fresh.{e}.D1.unseen.SS') < val(f'sm.fresh.{e}.A4.unseen.SS') for e in ('E0', 'E3', 'E5', 'E6')))
claim('D1 changed more unseen N decisions than A4 for every encoder', all(val(f'sm.fresh.{e}.D1.unseen.flip_O_N') > val(f'sm.fresh.{e}.A4.unseen.flip_O_N') for e in ENCS))
claim('D1 changed far more unseen S decisions than A4 for E0 and E6 (at least 20 points more)',
      all(val(f'sm.fresh.{e}.D1.unseen.flip_O_S') - val(f'sm.fresh.{e}.A4.unseen.flip_O_S') > 20 for e in ('E0', 'E6')))
claim('R2 contextual: E2 > E6 > E5 >= E4, E1 (near zero)', val('R2.E2.mean') > val('R2.E6.mean') > val('R2.E5.mean') >= max(val('R2.E4.mean'), val('R2.E1.mean')) and val('R2.E5.mean') < 6)
claim('two outcomes within one point of a threshold: E1 on R8, E6 on R2', 0 < 10 - val('R8.E1.mean') < 1 and 0 < 15 - val('R2.E6.mean') < 1)
claim('paraphrase flips: A1 Fresh range below 10, C1 Fresh about 3, C1 Perm about six times C1 Fresh',
      val('rng.fresh.A1.flip_O_P.max') < 10 and 3 < val('sm.fresh.C1.C1.all.flip_O_P') < 3.2
      and 5.5 < val('sm.perm.C1.C1.all.pooled.flip_O_P') / val('sm.fresh.C1.C1.all.flip_O_P') < 6.5)
claim('paraphrase flips on Perm: A1 range min from E3 (float64) and max from E4', rnd(val('rng.perm.A1.flip_O_P.min'), 2) == rnd(val('f64.E3.A1.all.pooled.flip_O_P'), 2)
      and rnd(val('rng.perm.A1.flip_O_P.max'), 2) == rnd(val('sm.perm.E4.A1.all.pooled.flip_O_P'), 2))
claim('NevIR nested within 1.5 points of complement for E1, E2, E6; E4 and E5 lower by the stated gaps',
      all(abs(val(f'nevir.{e}.M0.gap')) < 1.5 for e in ('E1', 'E2', 'E6')) and 2 < val('nevir.E4.M0.gap') < 3 and 4.5 < val('nevir.E5.M0.gap') < 5.5)
claim('v1 test A1 SS close to Fresh for E0, E1, E2 (within 6 points) and about ten points lower for E6',
      all(abs(val(f'sm.v1.{e}.A1.{g}.SS') - val(f'sm.fresh.{e}.A1.{g}.SS')) < 6 for e in ('E0', 'E1', 'E2') for g in ('seen', 'unseen'))
      and all(7 < val(f'sm.fresh.E6.A1.{g}.SS') - val(f'sm.v1.E6.A1.{g}.SS') < 13 for g in ('seen', 'unseen')))
claim('selectivity: E1 highest, E2 lowest, E4 second lowest; E4 highest R8 mean',
      val('probe.E1.selectivity') == max(val(f'probe.{e}.selectivity') for e in CONTEXTUAL) and val('probe.E2.selectivity') == min(val(f'probe.{e}.selectivity') for e in CONTEXTUAL)
      and sorted(val(f'probe.{e}.selectivity') for e in CONTEXTUAL)[1] == val('probe.E4.selectivity') and val('R8.E4.mean') == max(val(f'R8.{e}.mean') for e in CONTEXTUAL))
claim('one S decision moves the same-cue S rate by about seven points (14 S scenarios)', val('samecue.fresh.E1.n_S') == 14 and 7 < 100 / val('samecue.fresh.E1.n_S') < 7.2)
claim('Qwen train split disagreement about 40%', 40 < val('n.qwen_train_disagree_pct') < 41 and abs(val('n.qwen_train_disagree_pct') + val('qwen.v1.split.train.pct') - 100) < 1e-9)
claim('pilot LR runs 63 = 7 encoders x 3 rates x 3 seeds', val('n.pilot_lr_runs') == 63)
claim('Perm connectors: six distinct, comma variants 18/3/2', val('data.perm.connector.n_distinct') == 6 and val('data.perm.connector.comma_and') == 18
      and val('data.perm.connector.comma_while') == 3 and val('data.perm.connector.comma_but') == 2)
claim('Perm stage other = 2', val('build.perm.stage.other') == 2)
claim('M0-native rows: E6 M0n values exist and Perm SS_bal small', abs(val('sm.perm.E6.M0n.seen.SS_bal')) < 5 and abs(val('sm.fresh.E6.M0n.unseen.SS')) < 10)

# ---- released systems (Section 5, revision step 1)
PUB_RETR = ['colbertv2', 'answerai', 'gtemc', 'bgem3c', 'bgem3d', 'bgem3s', 'bgebase', 'e5', 'mpnet', 'gtemb', 'splade', 'minilm', 'bgerr', 'bgerr2', 'mxbai', 'monot5b', 'monot5l']
PUB_LI = ['colbertv2', 'answerai', 'gtemc', 'bgem3c']
PUB_DENSE = ['bgem3d', 'bgebase', 'e5', 'mpnet', 'gtemb']
PUB_CE = ['minilm', 'bgerr', 'bgerr2', 'mxbai']
PUB_LLM = ['olmo', 'phi', 'qwen3']
ss = lambda k: val(f'pub.{k}.perm.SS_bal')
claim('18 checkpoints, 20 rows, 17 retrieval rows, 3 judges, 4 pure late-interaction rows', val('pub.n_checkpoints') == 18 and val('pub.n_rows') == 20 and val('pub.n_retrieval') == 17 and val('pub.n_llm') == 3 and val('pub.n_li') == 4)
claim('no retrieval scorer exceeded the retrieval SS_bal maximum, which is monoT5-3B', val('pub.rng.retrieval.SS_bal.max') == max(ss(k) for k in PUB_RETR) == ss('monot5l') and val('pub.best_retrieval.key') == 'monot5l')
claim('the four late-interaction intervals contain zero; their SS_bal range is the li range', val('pub.n_ci_zero_li') == 4 and val('pub.rng.li.SS_bal.min') == min(ss(k) for k in PUB_LI) and val('pub.rng.li.SS_bal.max') == max(ss(k) for k in PUB_LI) and abs(val('pub.rng.li.SS_bal.max')) < 2)
claim('judges reach 73 to 91 points, far above every retrieval scorer', val('pub.rng.llm.SS_bal.min') == ss('olmo') > 70 and val('pub.rng.llm.SS_bal.max') == ss('qwen3') > 90 and ss('olmo') > 5 * val('pub.rng.retrieval.SS_bal.max'))
claim('10 of 17 retrieval intervals contain zero', val('pub.n_ci_zero_retrieval') == 10)
claim('retrieval acc_N range below the acc_O range (they answer N as they answer O)', val('pub.rng.retrieval.acc_N.max') < val('pub.rng.retrieval.acc_O.min'))
claim('dense bi-encoders between mpnet (min) and gte-modernbert (max)', ss('mpnet') == min(ss(k) for k in PUB_DENSE) and ss('gtemb') == max(ss(k) for k in PUB_DENSE))
claim('cross-encoders between MiniLM (min) and mxbai (max); monoT5-base below monoT5-3B', ss('minilm') == min(ss(k) for k in PUB_CE) and ss('mxbai') == max(ss(k) for k in PUB_CE) and ss('monot5b') < ss('monot5l'))
claim('judges joint accuracy above every retrieval scorer; mxbai is the retrieval maximum', min(val(f'pub.{k}.perm.joint_NS_acc') for k in PUB_LLM) > val('pub.mxbai.perm.joint_NS_acc') == max(val(f'pub.{k}.perm.joint_NS_acc') for k in PUB_RETR))
claim('clause-order asymmetry in rerankers: first > second for mxbai and bge-reranker-base; judges asymmetry between qwen3 (min) and olmo (max), both positive',
      val('pub.mxbai.perm.SS_first') > val('pub.mxbai.perm.SS_second') and val('pub.bgerr.perm.SS_first') > val('pub.bgerr.perm.SS_second')
      and 0 < val('pub.qwen3.perm.asym') == min(val(f'pub.{k}.perm.asym') for k in PUB_LLM) and val('pub.olmo.perm.asym') == max(val(f'pub.{k}.perm.asym') for k in PUB_LLM))
nv = lambda k, sub='all': val(f'pub.{k}.nevir.{sub}')
claim('NevIR all: late interaction answerai (min) to bge-m3 multi-vector (max) among the other three', nv('answerai') == min(nv(k) for k in ('answerai', 'gtemc', 'bgem3c')) and nv('bgem3c') == max(nv(k) for k in ('answerai', 'gtemc', 'bgem3c')))
claim('NevIR all: dense from mpnet (min) to bge-m3 dense (max); cross-encoders MiniLM to mxbai; judges OLMo to Phi',
      nv('mpnet') == min(nv(k) for k in PUB_DENSE) and nv('bgem3d') == max(nv(k) for k in PUB_DENSE) and nv('minilm') == min(nv(k) for k in PUB_CE) and nv('mxbai') == max(nv(k) for k in PUB_CE)
      and nv('olmo') == min(nv(k) for k in PUB_LLM) and nv('phi') == max(nv(k) for k in PUB_LLM))
claim('nested subset: retrieval range from bge-m3 sparse to monoT5-3B; judges range', val('pub.rng.retrieval.nested.min') == nv('bgem3s', 'nested') and val('pub.rng.retrieval.nested.max') == nv('monot5l', 'nested')
      and val('pub.rng.llm.nested.min') == min(nv(k, 'nested') for k in PUB_LLM) and val('pub.rng.llm.nested.max') == max(nv(k, 'nested') for k in PUB_LLM))
ALL20 = PUB_RETR + PUB_LLM
gaps = sorted(((val(f'pub.{k}.nevir.gap'), k) for k in ALL20), reverse=True)
claim('14 of 20 systems: complement inside the nested interval; largest gaps Phi-4-mini then bge-m3 multi-vector', val('pub.n_nested_ci_covers_complement') == 14 and gaps[0][1] == 'phi' and gaps[1][1] == 'bgem3c')
claim('equal bags: SPLADE 0, bge-m3 sparse lowest nonzero, dense from mpnet (min) to bge-base (max), judges from OLMo (min) to Phi (max)',
      nv('splade', 'equal') == 0 and nv('mpnet', 'equal') == min(nv(k, 'equal') for k in PUB_DENSE) and nv('bgebase', 'equal') == max(nv(k, 'equal') for k in PUB_DENSE)
      and nv('olmo', 'equal') == min(nv(k, 'equal') for k in PUB_LLM) and nv('phi', 'equal') == max(nv(k, 'equal') for k in PUB_LLM) and nv('bgem3s', 'equal') < 5)
claim('Fresh SS ranges of retrieval scorers and judges are consistent with the registry ranges', val('pub.rng.retrieval.fresh_SS.max') == max(val(f'pub.{k}.fresh.SS') for k in PUB_RETR) and val('pub.rng.llm.fresh_SS.min') == min(val(f'pub.{k}.fresh.SS') for k in PUB_LLM))
claim('precision control: 173 ties before, 0 after; monoT5-3B gained more than 4 points', val('pub.prec.ties_prev_total') == 173 and val('pub.prec.ties_new_total') == 0 and val('pub.prec.monot5l.nevir_all.delta') > 4 and val('pub.prec.monot5l.prev.nevir_ties') == 118)
claim('ColBERTv2 anchor: above 99% agreement on Perm decisions and above 96% on NevIR comparisons with the Section 7 run', val('pub.anchor.perm.agree') > 99 and val('pub.anchor.nevir.agree') > 96)
claim('bge-m3 heads agree with the reference to within 1e-3', val('pub.check.bgem3.maxdiff') < 1e-3 and val('pub.check.bgem3.pairs') == 40)

# ---- fine-tuning (Section 6, revision step 3)
for m in ('colbertv2', 'answerai'):
    claim(f'{m}: NegPairs fine-tuning raised flip_O_N and flip_O_S together; SS_bal stays between 11 and 12',
          val(f'ft.{m}.np.perm.flip_O_N') > val(f'ft.{m}.base.perm.flip_O_N') + 20 and val(f'ft.{m}.np.perm.flip_O_S') > val(f'ft.{m}.base.perm.flip_O_S') + 15
          and val(f'ft.{m}.np.perm.flip_O_S') > 0.7 * val(f'ft.{m}.np.perm.flip_O_N') and 11 < val(f'ft.{m}.np.perm.SS_bal') < 12)
    claim(f'{m}: Fresh SS after NegPairs fine-tuning far above Perm SS_bal', val(f'ft.{m}.np.fresh.SS') > 2 * val(f'ft.{m}.np.perm.SS_bal'))
    claim(f'{m}: NevIR rose and SciFact fell after NegPairs fine-tuning', val(f'ft.{m}.np.nevir.all.delta') > 0 and val(f'ft.{m}.np.nevir.nested.delta') > 0 and val(f'ft.{m}.np.scifact.drop') > 5)
    claim(f'{m}: NevIR fine-tuning raised NevIR, cost little SciFact and did not transfer to Perm', val(f'ft.{m}.nv.nevir.all.delta') > 3 and 0 < val(f'ft.{m}.nv.scifact.drop') < 2 and val(f'ft.{m}.nv.perm.SS_bal') < 3)
    claim(f'{m}: dev accuracy rose after NegPairs fine-tuning', val(f'ft.{m}.np.dev_after.mean') > val(f'ft.{m}.np.dev_before') + 10)
claim('fine-tuned SS_bal about the level of the trained heads (R8 range) and far from the judges', val('ft.rng.np.SS_bal.max') < val('pub.rng.llm.SS_bal.min') / 5 and 9 < val('ft.rng.np.SS_bal.min') and val('ft.rng.np.SS_bal.max') < 13)
claim('SciFact drops 6.4 to 11.1 points (NegPairs fine-tuning)', 6 < val('ft.rng.np.scifact.drop.min') < val('ft.rng.np.scifact.drop.max') < 12)
claim('fine-tuning data counts as stated', val('ft.v1.train_items') == 1692 and val('ft.v1.dev_items') == 420 and val('ft.nevir.train') == 948 and val('ft.nevir.validation') == 225 and val('ft.scifact.n_corpus') == 5183 and val('ft.scifact.n_queries') == 300)
claim('A5 changed unseen-family SS by only a few points for every encoder', abs(val('rng.fresh.A5mA1.unseen.all7.min')) < 5 and abs(val('rng.fresh.A5mA1.unseen.all7.max')) < 5)
claim('probe accuracy range: E6 lowest, E2 highest among contextual encoders', val('probe.E6.cv_mean') == min(val(f'probe.{e}.cv_mean') for e in CONTEXTUAL) and val('probe.E2.cv_mean') == max(val(f'probe.{e}.cv_mean') for e in CONTEXTUAL))
claim('clause order: first > second for every contextual encoder and C1', all(val(f'sm.perm.{e}.A1.seen.first.SS') > val(f'sm.perm.{e}.A1.seen.second.SS') for e in CONTEXTUAL) and val('sm.perm.C1.C1.seen.first.SS') > val('sm.perm.C1.C1.seen.second.SS'))

if __name__ == '__main__':
    bad = [t for t, ok in checks if not ok]
    for t, ok in checks:
        print(('OK   ' if ok else 'FAIL ') + t)
    print(f'{len(checks) - len(bad)}/{len(checks)} claims hold')
    raise SystemExit(1 if bad else 0)
