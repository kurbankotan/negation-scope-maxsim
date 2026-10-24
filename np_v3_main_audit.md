# NegPairs v3.0 extended experiment: audit of the delivered results (7 October 2026)

Files audited: `np_v3_main_results.zip` (run `np_v3_main`, one Colab session started 2026-10-07 09:56 UTC on an NVIDIA A100-SXM4-40GB; package `cacsa2` 0.3.1; Python 3.13.15, PyTorch 2.11.0+cu130, Transformers 5.18.0). Every number below was read from the files in the zip by script. Raw predictions and module states are not in this zip (they remain on Drive, listed with SHA-256 in `FILE_MANIFEST_SHA256.csv`); the independent recomputation of every table waits for `np_v3_main_predictions.zip`.

## 1. Integrity and provenance

- `FILE_MANIFEST_SHA256.csv` lists 6,076 files; the 2,378 present in the zip all match (0 mismatches); the absent ones are the 3,698 prediction files and module states.
- `manifest_lock.json`: SHA-256 `57f834c5…1561c` of the frozen manifest, written at 09:56:13 UTC before any training; identical to the frozen file.
- Data: NegPairs v1 (train 1,692 items, 423 scenarios; dev 420, 105) and the audited v3.0 files (v1 test 2,240 items; Fresh 960; Perm 1,096) verified against their manifests and the embedded hashes. The test files were parsed only after the pilot (`lr_grid.json` records `test_files_parsed_before: false`).
- Encoder revisions as pinned (E1 `e8f8c211…`, E2 `8ccc9b6f…`, E4 `e8c3b32e…`, E5 `a5beb1e3…`, E6 `c1e84128…`, GloVe repository `1db2080b…`, 400,001 vectors; projection 128 x 768 read from the ColBERTv2 checkpoint). No text was truncated at 128 tokens (NegPairs) or 512 tokens (NevIR, contextual encoders); E0 and E3 encoded every NevIR token. E3 out-of-vocabulary rate 0.09% on train and dev, 1.10% over all texts including NevIR (examples: 12gb, breakroom, smartwatch, waitlist). Unknown-token rate 0 for every subword tokenizer.
- NevIR test split: 1,383 instances, dataset revision `62635850…`; nested subset 184 instances (13.3%), 36 with equal bags.
- Qwen2.5-7B-Instruct revision recorded from the cache: `a09a35458c702b33eeacc393d103063234e8bc28` (the revision deduced for the data build).
- 718 runs with `done.json`: 560 grid runs, 140 control runs, 7 M0, 1 M0-native, 10 C1. Grid wall clock 10:19 to 13:01 UTC with four workers (mean 68 s per run; 10.6 run-hours of training in 2.7 hours). Perm token multisets of N and S are equal under all seven tokenizers in all 274 scenarios, so no Perm analysis was restricted.

## 2. Pilot (train and dev only)

- Exact checks with five random initializations (E0 and E3): identical scores under token permutation for max, mean, lse, the learned gate, M1 and C0 (largest difference 1.3e-15); no decrease on dev insertion pairs for max, lse and M1; the mean (426 and 234 violations) and the signed max with resolver masks (811 and 840) violate the inclusion property, as expected. S4 not met.
- C1 check (two epochs on dev): dev `acc_OP` 96.19 (S3 threshold 75).
- Learning rates (A1, seeds 13/42/71, mean dev accuracy): 3e-3 for E1 (70.16), E2 (66.11), E4 (70.95), E0 (64.60), E3 (61.59); 1e-3 for E5 (69.13) and E6 (69.29).

## 3. Gate and statistical rules (`gate.json`, `rules.json`)

Gate G (A1 seed-mean SS on v1 dev, threshold 20): E6 60.00, E1 54.29, E4 48.00, E5 45.05, E2 31.62, E0 24.29 passed; E3 11.71 failed. E0 is bimodal across seeds (six seeds between 37.14 and 47.62, four between −2.86 and 0.00; corrected on 7 October 2026 from an earlier reading of five and five); E2 likewise (three seeds between 0.00 and 0.95, seven between 33.33 and 64.76). k = 6 for R2 and R3, k = 5 for R8; intervals at 99.17% and 99.00%.

| Rule | Encoder | Seed mean | t interval | Positive seeds | Cluster interval | Outcome |
|---|---|---|---|---|---|---|
| R2 (A4 − A1, unseen SS, Fresh; ≥ 15) | E0 | +18.00 | [9.88, 26.12] | 10/10 | [8.58, 28.08] | met |
| | E6 | +14.25 | [6.53, 21.97] | 10/10 | [1.64, 27.00] | not met (a) |
| | E2 | +19.83 | [−3.43, 43.10] | 8/10 | [10.25, 29.25] | not met (b, c) |
| | E5 | +5.75 | [−8.77, 20.27] | 8/10 | [−9.36, 20.92] | not met |
| | E4 | +1.58 | [−7.18, 10.34] | 4/10 | [−13.25, 16.17] | not met |
| | E1 | −3.17 | [−15.31, 8.98] | 3/10 | [−11.75, 5.58] | not met |
| R3 (A6 − A4, flip_O_S on S-same, Fresh; ≥ 10; 231 scenarios) | E2 | +17.40 | [2.67, 32.14] | 9/10 | [12.55, 22.29] | met |
| | E6 | +11.77 | [−5.80, 29.35] | 8/10 | [6.58, 16.62] | not met (b, c) |
| | E0 | +10.09 | [−10.23, 30.40] | 8/10 | [5.02, 15.15] | not met (b, c) |
| | E5 | +9.78 | [−4.69, 24.26] | 7/10 | [5.01, 14.46] | not met |
| | E4 | +3.59 | [−10.56, 17.75] | 5/10 | [−2.21, 9.13] | not met |
| | E1 | −6.97 | [−11.73, −2.21] | 0/10 | [−11.21, −2.93] | not met |
| R8 (A1 SS_perm_bal, seen families, Perm; ≥ 10; 76 + 76 scenarios) | E4 | 19.74 | [13.56, 25.91] | 10/10 | [12.83, 26.78] | met |
| | E6 | 14.08 | [11.28, 16.88] | 10/10 | [7.50, 20.92] | met |
| | E5 | 10.46 | [5.07, 15.85] | 10/10 | [3.82, 17.30] | met |
| | E1 | 9.80 | [4.32, 15.28] | 10/10 | [4.41, 15.20] | not met (a) |
| | E2 | 7.89 | [0.65, 15.13] | 8/10 | [3.75, 12.24] | not met (a, b) |

E3: gate not passed; R2, R3 and R8 not evaluated. R1 (descriptive): E0 A1 unseen-family SS on Fresh −0.83 (seeds between −2.5 and 0.0; cluster interval [−1.67, −0.08]); E3 +1.67.

## 4. Exact rules (`exact_R5.json`, `exact_R6.json`, `exact_R7.json`)

- R5 (Proposition B): 186 checks (E0 and E3; A1, A3, M1 x 10 seeds and M0; v1 test 380 insertion pairs, Fresh 137, Perm 152), 0 violations; met.
- R6 (Proposition A): 142 checks on the 274 permutation pairs (E0 and E3; A1, A2, A3, A5, M1, C0 x 10 seeds and M0, plus A6 on the 272 scenarios whose marked multisets agree), 0 score differences above 1e-9, 0 decision disagreements, 0 near-ties; met. Every unmasked rule of E0 and E3 therefore has SS_perm = 0 and joint (N, S) accuracy 0 on all 274 scenarios in every seed. GPU (bfloat16) item decisions differed from the float64 decisions in 62 (E0) and 50 (E3) of 27,400 item predictions (rounding at near-ties); none changed an N/S pair.
- R7 (Corollary B.1): 62 checks on the 184 nested NevIR instances (E0 and E3; M0 and A1, A3, M1 x 10 seeds), 0 violations, paired accuracy 0 on the nested subset for every static scorer; met.

## 5. Descriptive results

NegPairs-Fresh (seed means, A1 = max): seen / unseen SS: E0 25.7 / −0.8; E3 11.3 / 1.7; E1 60.8 / 30.5; E2 31.2 / 13.5; E4 52.6 / 40.8; E5 47.0 / 32.4; E6 65.7 / 23.5; C1 92.7 / 91.2 (accuracy 96.9). Supplied exact scope (D1 − A1, unseen SS): E0 +8.5 [3.3, 13.7], E3 +19.8, E2 +26.4 [15.0, 37.8], E1 +6.2, E4 +3.6, E5 −5.8, E6 −2.6. The learned gate (A5 − A1, unseen SS) lies between −3.2 and +4.4 for every encoder. M1 (learned MaxSim without the affinity network) against A1 on unseen SS: −0.2 (E0), −1.1 (E1), −11.2 (E4), −14.0 (E2), −24.8 (E6), −32.2 (E5). Zero-shot M0 has SS near zero on every set. Same-cue "not" (45 N, 14 S scenarios): E0 35.6 against 25.7, E3 14.9 against 15.0, E1 73.3 against 33.6, E6 76.2 against 29.3.

NegPairs-Perm (seed means; seen families, 76 scenarios per clause order): A1 SS_perm_bal: E0 0.0, E3 0.0 (exact), E1 9.8, E2 7.9, E4 19.7, E5 10.5, E6 14.1; C1 58.1. Clause-order effect for A1 (SS with the decisive clause first / second): E1 25.3 / −5.7, E2 15.0 / 0.8, E4 33.2 / 6.3, E5 12.5 / 8.4, E6 25.4 / 2.8; C1 90.9 / 25.3 (with the decisive clause second, C1 changes its decision on S in 46.2% of scenarios and its O accuracy falls from 96.8 to 74.2). Joint (N, S) accuracy of A1: 11 to 26 for the contextual encoders, 0 for E0 and E3, 86.1 for C1. Signed max with the dataset scope (D1): E0 31.8, E3 68.9, E1 80.9, E2 92.6, E4 76.3, E5 85.9, E6 86.6; with the resolver scope (A4): E0 12.2, E3 12.9, E1 24.5, E2 33.2, E4 36.2, E5 19.4, E6 27.4; sentence-wide mask (A6): 0.0 to 10.4. The resolver detects the cue in 99.2 to 100% of Perm N and S arms, covers the decisive words in 99.2 to 100% of N arms and leaks into the other clause in 0.0 to 3.9% of N arms; in S arms its scope lies in the other clause in 99.2 to 100% and touches the decisive words in 0.7 to 2.5%.

NevIR (paired accuracy; random 25): M0 with E1 21.55 [19.4, 23.7], E5 21.19, E6 19.96, E4 18.87, E2 15.84, E3 4.99, E0 0.80; M0-native (ColBERTv2's own scoring with the checkpoint settings; 705 of 2,766 documents truncated at 180 tokens) 16.49 [14.5, 18.4] (published ColBERTv2 value 13.0, different evaluation code). Trained static modules on NevIR: between 0.0 and 3.5. Nested subset: every static scorer 0.00; contextual M0 between 14.67 and 21.20 on it.

Probing (clause of the edit; chance 50): cross-validated accuracy E2 89.3, E1 83.6, E5 80.8, E4 74.0, E6 73.1, E0 50.0, E3 50.0; transfer seen to unseen 79.1, 71.7, 67.0, 64.1, 63.1. Neutral twins ("also" at the site): E2 82.1, E1 61.0, E5 60.7, E4 62.4, E6 56.9; selectivity (syntactic negation minus twins) E1 18.2, E5 16.6, E6 15.5, E4 7.6, E2 2.2. A5 gate AUC for the same words: 0.49 to 0.61 (E2 0.613).

Qwen audit of NegPairs v1: 680 of 1,088 scenarios (62.5%) pass all four arms (test split 64.8%); arm agreement O 95.4, P 94.6, S 90.7, N 73.8; families: affixal 75.9, quantifier 62.5, syntactic 57.1, lexical 55.8. The v1 test tables recomputed on the passing scenarios are in `table_v1test_flip_qwen_passing.md`.

v1 test (previously read; descriptive): A1 seen / unseen SS: E0 20.6 / −1.7; E1 56.8 / 24.6; E2 30.5 / 14.0; E6 54.2 / 10.6.

## 6. Deviations and open points

1. Manifest section 10 requires every reported E0 and E3 metric on NegPairs-Perm to come from float64 scores. The exact rule R6 was evaluated in float64, but the Perm tables of E0 and E3 (as of every encoder) were computed from the GPU predictions. For the unmasked rules the GPU values are exactly 0 in every seed, like the float64 result; for A4, A6 and D1 the float64 values are still to be computed. `NegPairs_v3_followup_colab.ipynb` (package 0.3.2) re-scores every E0 and E3 condition on Perm in float64 from the archived module states and lists every difference.
2. The independent recomputation of every table from the raw predictions needs `np_v3_main_predictions.zip`.
3. The secondary accuracy gate of v2.0 (A1 `acc_OP` at least 70 and at least 50 on the prior-incongruent stratum) is passed by no encoder on NegPairs-Fresh, as in v2.0; reported as descriptive.

## 7. Independent recomputation (7 October 2026)

`np_v3_main_predictions.zip` (30.7 MB; 2,935 compact prediction files: 718 runs x three test sets plus 781 dev files) was recomputed by a script that does not import the package (`indep_v3/recompute.py`): for every run and test set the flip rates, scope sensitivity (all, seen, unseen), arm accuracies, S-same `flip_O_S`, same-cue "not" flips, and for NegPairs-Perm the pooled and order-split blocks, `SS_perm_bal` and joint (N, S) accuracy; the dev gate quantity of every encoder; the seed means and seed-level t intervals of R2, R3 and R8. 206,092 values were compared with the notebook's `per_run_*.json`, `gate.json` and `rules.json`: 0 mismatches, largest absolute difference 0.0. The gate decision of every encoder is reproduced. Descriptive (gate not passed, rule not evaluated): for E3 the R2 contrast would have been +23.58 [17.41, 29.75], positive in 10 of 10 seeds.

## 8. Float64 follow-up for E0 and E3 on NegPairs-Perm (7 October 2026)

`np_v3_main_followup.zip` (5.84 MB, SHA-256 `c6b61bd8…c739`; `NegPairs_v3_followup_colab.ipynb`, package 0.3.2, CPU, computed 15:58 UTC) re-scored every E0 and E3 condition on NegPairs-Perm in float64 from the archived module states (202 runs: 10 trained conditions x 10 seeds and M0, per encoder) and wrote `perm_predictions_f64.jsonl` beside each GPU prediction file, `per_run_perm_f64_static.json`, `seed_means_perm_f64_static.json`, `table_perm_f64_static.md`, `perm_f64_vs_gpu.json` and a rewritten `FILE_MANIFEST_SHA256.csv` (6,282 entries; the 6,076 entries of the main run are all unchanged). All 206 files in the zip match the manifest. An independent script that does not import the package (`indep_v3/recompute_f64.py`) recomputed every per-run Perm metric from the float64 predictions (43,834 values compared with `per_run_perm_f64_static.json`: 0 mismatches, largest absolute difference 0.0) and every seed mean (4,774 values: 0 mismatches), recounted the GPU-versus-float64 decision disagreements (1,282 of 2 x 1,096 x 101 item predictions, in 137 of 202 runs; identical per run to `perm_f64_vs_gpu.json`; 754 in unmasked and 528 in masked conditions; the two untrained M0 scorers alone account for 394, because their option scores are often nearly tied) and the seed-mean differences (44 of 8 x 22 metrics; largest 1.32 points, E3 D1 `seen.second.SS` 69.87 in float64 against 68.55 on the GPU). Read directly off the archived float64 scores, every unmasked condition gives identical option scores under N and S in all 274 scenarios (19,454 scenario checks per encoder, largest absolute difference 0.0, 0 decision disagreements), which repeats R6 on the complete set of runs.

Consequence for the paper (manifest section 10): every E0 and E3 number on NegPairs-Perm is taken from the float64 files. Unmasked rules: `SS_perm` and joint (N, S) accuracy exactly 0 in every seed (as on the GPU). Masked rules (seed means on the seen families, float64 against GPU): A4 `SS_perm_bal` 12.83 (12.24) for E0 and 13.03 (12.90) for E3; D1 32.30 (31.78) and 70.07 (68.88); A6 0.20 (0.20) and 0.07 (0.00). Deviation 1 of section 6 is thereby resolved.
