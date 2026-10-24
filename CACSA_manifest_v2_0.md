# CACSA Study Manifest, version 2.0

Status: FROZEN, 4 October 2026. The author accepted the rules and thresholds of section 8 on 4 October 2026, after the pilot (train and dev only) and before any model was trained or evaluated on the test split. This file is shipped inside `cacsa2_project.zip` (0.2.5); the confirmatory notebook copies it into the run folder and records its SHA-256 in `manifest_lock.json` before any training. Later changes are amendments with their own version number, hash and date; this text is never edited in place.

Working title of the paper: fixed only after the confirmatory results.

Author: Kurban Kotan (Manisa Celal Bayar University). Analysis support: Claude (Anthropic). All runs are executed by the author on Google Colab.

History of this version: DRAFT 1 (3 October 2026) specified accuracy-based rules with a learnability gate. The pilot showed that the gate could not be met by any encoder for a reason that does not bear on the hypotheses (section 11.5). The frozen version keeps data, models, training, seeds and controls of draft 1 unchanged and moves the primary estimand to paired decision changes (scope sensitivity, section 7.1); the draft-1 accuracy rules are retained as a secondary, descriptive view (section 8.3).

---

## 0. Summary

Modules of the CACSA family score an answer option by aggregating token-pair constraint scores between a context and the option, and CACSA v2 aggregates with a maximum over context tokens. Two propositions (section 3) show that with context-independent token representations such a module is monotone under token addition (Proposition 1) and sees the context only as a bag of token pairs (Proposition 2): inserting a negation cue can never lower any option score, and the effect of a cue cannot depend on which words it negates. The study tests these statements and their empirical counterpart with frozen contextual encoders on NegPairs, a controlled set of minimal-pair scenarios with four arms: an affirmative context (O), its paraphrase (P), the context with one negation cue on the decisive words (N, answer flips) and the context with one negation cue on an irrelevant detail (S, answer unchanged). Two negation families are seen in training and two occur only in the test split. The primary estimand is scope sensitivity: the share of scenarios whose decision changes from O to N minus the share whose decision changes from O to S. It measures whether, and where, a cue changes the decision, independently of how well a model reads the affirmative facts. CondaQA, the benchmark of manifest v1.1, is reported as a bounded negative result (section 1).

## 1. History

### 1.1 The v1.1 run (CondaQA)

Run `main_20261003T050337Z` (3 October 2026, Colab A100, package `cacsa2` 0.1.1) executed the full v1.1 grid. Recomputed from the archived per-row test predictions (7,151 eligible test rows; label shares NO 48.34%, YES 45.63%, DON'T KNOW 6.03%): the seed-mean test accuracy of each of the 18 frozen-encoder conditions (3 encoders by 5 aggregators, plus control C0 per encoder; 90 runs) lay between 44.94% and 47.05% (single seeds 41.84% to 49.08%), that is, at or below the majority-label rate of 48.34%; in 16 of the 18 conditions more than half of the predictions (pooled over seeds) were YES. Control C1 (full fine-tuning) had a non-finite mean training loss in every epoch of every seed and returned NaN probabilities for every row, so its argmax defaulted to the first class, YES (45.63%); the cause was that transformers 5 had loaded the checkpoint in half precision. The GPU version of the Proposition-1 test reported 5 to 28 score increases per A1 seed (of 5,355 scores; largest increase 6.84e-5, above the 1e-5 tolerance) and none for A3.

### 1.2 Follow-up run

Run `followup_v11_20261003T093954Z` (package 0.1.2; 20 files, all SHA-256 values verified) re-scored the archived E0 modules in float64 on the CPU: zero increases for A1 and A3 in every seed (5,355 scores per seed over 1,785 passages; tolerance 1e-9). The v1.1 increases were therefore rounding effects of TF32 and bfloat16 arithmetic, not violations of Proposition 1. C1 was re-executed with weights loaded in float32 and a non-finite-loss guard: training was finite in all five seeds, and the seed-mean test accuracy was 46.60% (SD 0.67; seeds 45.67% to 47.32%); the selected checkpoint was the first epoch in four of five seeds, and the per-class standard deviation of the predicted probabilities across test rows was at most 0.075 in every seed. This matches published CondaQA results, where fully fine-tuned encoders of comparable size stayed near the majority baseline (majority 47.75%, BERT-large 46.3%, RoBERTa-large 54.08%, DeBERTa-v3-large 57.09%; Ravichander et al., 2022, Table 4; learning rate 1e-5 for 10 epochs, whereas C1 used 2e-5 for 3 epochs). These figures are to be re-verified against the publication before submission.

### 1.3 Diagnosis and decision

Three causes made the v1.1 rules uninterpretable as tests of the hypotheses: (a) the candidates were the bare answer strings yes, no and don't know, which gives a pairwise scorer no lexical anchor in the context (a design error of manifest v1.0); (b) no learnability gate was specified; (c) at this model scale CondaQA does not separate even a fully fine-tuned reference from the majority rate. On 3 October 2026 the author decided to move the confirmatory test to a controlled minimal-pair set in the original CACSA format (two content options per scenario), which manifest v1.1 (section 3.2) had deferred, and to keep CondaQA as a bounded negative result. No CondaQA result is used to evaluate any rule of this manifest.

### 1.4 Construction of NegPairs (two builder runs)

The first builder run (construction v1) examined 1,476 candidates and kept 102 (7%). Its record showed that most losses came from the pipeline: generator responses emptied or cut off by thinking under a 1,200-token limit (154 errors and 83 unparseable outputs), a substring test that rejected cues such as "no" because they occur inside other words ("notebook", "afternoon"), names counted as new when they moved out of sentence-initial position or took a possessive, an N edit limit of three words that excluded most natural quantifier and lexical negations, and a verifier wording that rejected clear items. The run was stopped and construction v2 (package 0.2.4) re-checked every candidate under one protocol (section 4.2). The first run's record is archived unchanged (`build_state_v1_backup.json`).

## 2. Scope and non-claims

In scope: modules of the form `score(o) = mean_j A_i(c_ij)` over frozen token representations; three encoders (one context-independent, two frozen contextual); six aggregation rules and one descriptive upper bound; NegPairs (English, model-generated, model-verified); five seeds; three controls.

Not claimed: anything about large language models or about pragmatic inference in general; that any CACSA variant is competitive with fine-tuned cross-encoders (the pilot already shows it is not, section 11.4); that A4 (signed max with parser scope) is a method free of hand-made resources (it uses a cue lexicon and a dependency parser and is reported as a scope-supplied diagnostic; A5 is the variant without a lexicon); that a decision change is a correct answer (scope sensitivity measures whether and where a cue changes the decision; correctness is reported separately as accuracy); that results transfer beyond the four negation families, beyond English, or to human-written text. NegPairs is not human-verified, which is stated wherever it is used.

## 3. Formal object and propositions

Setting and statements are those of manifest v1.1, section 2, unchanged. Context token representations `h_1..h_n`, option token representations `u_1..u_m`, pairwise constraint `c_ij = g(h_i, u_j)` (CACSA v2: `c_ij = e_ij + lambda * tanh(aff_ij)`), option score `S(o) = (1/m) * sum_j A({c_ij}_i)`.

Proposition 1 (token-addition monotonicity). If `h_i` depends only on the identity of token `i` and `A` satisfies `A(C + {c}) >= A(C)` for every finite multiset `C` and value `c` (true for `max` and log-sum-exp), inserting tokens into the context cannot decrease any option score.

Proposition 2 (bag of token pairs). With context-independent representations and a permutation-invariant `A`, `S` is a function of the multiset `{c_ij}`; the effect of a cue on `S` does not depend on which words it negates, so scope is not representable.

Consequences used here. Under E0 a cue can change a decision only by raising the score of the other option through its own pairwise scores, which depend on the cue word and not on its position; a cue word never seen in training has untrained pairwise scores. The module therefore cannot be scope-sensitive for cue families absent from training (rule R1), and for a cue word that occurs both on decisive words and on irrelevant details its decision changes should not depend on that position (same-cue analysis, section 7.2).

Empirical bridge. With frozen contextual encoders the propositions do not hold exactly; polarity and scope can enter only through the encoder's contextualization of `h_i`. Whether they do, in a form a max aggregator can use, is reported for every encoder (bridge block of section 8.2).

## 4. Data: NegPairs

### 4.1 Design

Each scenario has a short context (requested: one or two sentences of at most 25 words; enforced: at most 30 words in every arm), one question asking what someone should do, and two action options (requested: 2 to 5 words; enforced: 1 to 6). Four arms share the question and the options:

| Arm | Context | Answer |
|---|---|---|
| O | affirmative; no negation of any kind | y |
| P | paraphrase of O; no negation | y (edit-cost control) |
| N | O with one negation cue applied to the decisive words; at most five words changed | 1 - y (polarity probe) |
| S | O with one negation cue of the same family on an irrelevant detail; decisive words unchanged; at most ten words changed | y (scope control) |

Negation families: syntactic (not, n't, never, cannot) and quantifier (no, none, nobody, nothing, neither, nowhere) are seen in training; affixal (un-, in-, im-, il-, ir-, dis-, non-, -less) and lexical (lacks, fails to, absent, without, free of, out of, refuses to, missing) occur in the test split only. Scenarios are spread over 14 everyday domains and 8 situation types; a list of over-used topics is excluded in the generation prompt.

### 4.2 Construction v2 (notebook `NegPairs_build_colab.ipynb`, module `cacsa2.negpairs_build`, package 0.2.4)

Generation: `claude-sonnet-5` with the unchanged generation prompt (SHA-256 `8270c1684edaadff9941da5244a8812e7fcdc62b95ada62ea21bfa38e668af35`); candidates of the first run were generated at effort medium with a 1,200-token limit, the rest at effort low with a 4,000-token limit. Deterministic filters (`structural_filter_v2`): required fields; two distinct options of 1 to 6 words; no core negation marker in the question, the options, O or P; question ends with a question mark; every context at most 30 words; decisive words present in O and unchanged in S; declared cues present in N and S and absent from O as whole words; each cue belongs to the assigned family; word edit distance O to N between 1 and 5, O to S at most 10, O to P at least 2; no name or number in P or N that does not occur in O (case-insensitive, possessive stripped). Near-duplicate filter: exact match of O, or Jaccard similarity at least 0.6 of content words (context and options) with a kept scenario of the same domain. Blind verification: each arm is shown alone to `claude-haiku-4-5-20251001` at temperature 0 with the v2 verifier prompt (SHA-256 `1ce31ce47f63c8c7903b22ba4869b5a99c0262a2f9356fec52faa117ce8c3fdd`), which must return the expected option (better-choice wording; "unclear" only when the context gives no reason to prefer either option) and judge the text natural. Every kept scenario, including those generated in the first run, passed this single protocol; first-run verdicts are archived and not reused. Audit: a seeded 10% sample of kept scenarios was re-verified by `claude-opus-5` with the v2 prompt. Targets: 500 scenarios each for the syntactic and quantifier families and 250 each for the affixal and lexical families (caps, not requirements); schedule seed 20261003.

### 4.3 Splits and format

Splits are drawn by scenario: seen families 60% train, 15% dev, 25% test (within family); unseen families test only. The option order is randomized per scenario (seeded). One row per arm (`item_id = <scenario_id>-<arm>`). Every notebook verifies the dataset files against `FILE_MANIFEST_SHA256.csv` and asserts that no scenario crosses splits and that train and dev contain seen families only.

### 4.4 Realized dataset (from `build_report.json`, built 3 October 2026, package 0.2.4)

All 3,300 scheduled candidates were examined under construction v2; 1,088 scenarios were kept (505 generated in the first run, 583 in the second); discards: filter 824, verifier 1,224, near-duplicate 163, generator 1. The targets were not reached in any family. Items: 4,352 (four per scenario).

| Split | syntactic | quantifier | affixal | lexical | scenarios |
|---|---|---|---|---|---|
| train | 221 | 202 | 0 | 0 | 423 |
| dev | 55 | 50 | 0 | 0 | 105 |
| test | 92 | 84 | 228 | 156 | 560 |

Label shares (O arm; label 0 = first option, 1 = second): train 199 / 224, dev 61 / 44, test 264 / 296 (N reversed, P and S as O). Word edits O to N: 1 to 5 (train: 1 word 124, 2 words 89, 3 words 104, 4 words 54, 5 words 52); O to S: median 6, maximum 10. N cues in train and dev are dominated by no (128), none of (86), is not (86), cannot (42) and does not (39); S cues by nobody (206) and never (129); the test split has 54 distinct affixal and 35 distinct lexical N cues. Opus audit (109 sampled scenarios): 70 agreed on all four arms; in 29 the audit stopped at an arm it judged unnatural after agreeing on the answers up to that arm (N 20, S 4, O 4, P 1); in 10 it gave a different answer on one arm (N "unclear" 5, N read as the O answer 3, O read as the other option 1, O "unclear" 1). Metered API cost: 31.98 USD in total (first run 15.66, second 16.32).

### 4.5 Known limitations and how they are handled

(a) Option prior: the option that is correct in the affirmative arms is often recognizable from the option texts alone (a normal action against a fallback); on dev, the context-blind control C2 reached 70.5% to 76.2% on the affirmative arms (section 11.4). Scope sensitivity is a within-scenario difference of decision changes and does not reward such a prior; accuracy-based results are reported with option-prior strata. (b) Cue identity: N and S cues of the seen families differ in distribution (N mostly no, none of, not; S mostly nobody, never), so in the seen families cue identity alone can separate N from S; the confirmatory tests of the limit and of the remedy therefore use the unseen families, where every cue is new, and the same-cue analysis isolates position for the cue "not". (c) Naturalness: minimal edits produce some odd sentences (Opus audit above). (d) The scope resolver is a frozen heuristic resource whose coverage of the unseen families is measured, not tuned (section 5.5).

## 5. Models

### 5.1 Encoders (frozen; last-layer token states; no gradient)

E0 Hash: context-independent; text is split into word and punctuation tokens (`\w+|[^\w\s]`), and each lower-cased token type maps to a fixed unit vector of dimension 384 drawn from a SHA-256-seeded Gaussian (salt `cacsa2-e0`). It is the regime of Propositions 1 and 2.
E1 MiniLM: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions; revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` in the pilot).
E2 DeBERTa: `microsoft/deberta-v3-base` (768 dimensions; revision `8ccc9b6f36199bec6961081d44eb72fb3f7353f3` in the pilot).
For E1 and E2: special tokens are removed, every token state is L2-normalized (as in the archived CACSA v2 notebook), the maximum length is 128 tokens (no text was truncated in the pilot), inference runs in bfloat16 on the GPU, and the revision is resolved and recorded at run time. Token states are cached in half precision on the runtime's local disk and shared by every module variant and control.

### 5.2 Input formulation

The context and the question are encoded jointly as `<context> <question>`, and each option separately. `S(o) = mean over the option's tokens j of A_i(c_ij)`, with `i` over the jointly encoded tokens; the prediction is the argmax over the two options; training minimizes cross-entropy over the two option scores.

### 5.3 Shared module (CACSA v2 core; unchanged from v1.1)

`e_ij = F(h_i)^T F(u_j) / sqrt(128)` with `F` a two-layer MLP (d to 128 to 128); affinity network over `[h_i ; u_j ; h_i - u_j ; h_i * u_j]` (4d to 256 to 64 to 1, ReLU, dropout 0.1); `lambda = sigmoid(theta)`; `c_ij = e_ij + lambda * tanh(aff_ij)`. Trainable parameters: d = 384, 475,778 for A1 and A2, 475,779 for A3, A4, A6 and D1, 476,164 for A5; d = 768, 918,146, 918,147 and 918,916 respectively.

### 5.4 Aggregators (experimental factor) and the descriptive upper bound

| ID | Rule | Monotone under addition | Polarity source | Extra parameters |
|---|---|---|---|---|
| A1 MAX | `max_i c_ij` (CACSA v2) | yes | none | 0 |
| A2 MEAN | `mean_i c_ij` | no | none | 0 |
| A3 LSE | `tau * log sum_i exp(c_ij / tau)`, `tau = softplus(t)` learned | yes | none | 1 |
| A4 SIGNED-MAX, dependency scope | `max_{i not in N} c_ij - beta * max_{i in N} c_ij`, `N` = tokens inside a scope found by the resolver, `beta = softplus(b)` learned | no | scope resolver (external resource) | 1 |
| A5 SIGNED-MAX, learned gate | as A4 with soft membership `p_i = sigmoid(w . h_i + b)` learned from the frozen token state | no | learned, no lexicon | d + 2 |
| A6 SIGNED-MAX, sentence-wide | as A4 with `N` = every token of any sentence that contains a detected cue | no | resolver cues, scope-blind | 1 |
| D1 SIGNED-MAX, dataset-supplied (descriptive) | as A4 with `N` from the dataset: in N items the decisive words and the cue, in S items the cue only, none in O and P | no | dataset annotation and arm identity | 1 |

If `N` is empty, A4, A6 and D1 reduce to A1. D1 reads the arm identity and the generator's `decisive` field, so it is not a method; it shows what exact scope would allow the signed rule to do.

### 5.5 Scope resolver (frozen) and polarity masks

The resolver of manifest v1.1, section 4.4, is used unchanged (cue lexicon and scope rules identical since `cacsa2` 0.1.1; spaCy `en_core_web_sm` 3.8.0), without the CondaQA training-cue list. Its stoplist excludes seven frequent -less words (wireless, homeless, harmless, endless, countless, careless, needless), and "out of" and "missing" are not in its lexicon, so some unseen-family cues cannot be detected; this is reported, not corrected. Masks are computed on `<context> <question>` and mapped to encoder tokens by character overlap. For every test item the notebook records cue detection, whether the resolved scope covers the decisive words, whether the S cue shares a spaCy sentence with the decisive words, and cues detected in O or P.

### 5.6 Controls

C0 Head-only: `score_k = w . [mean(ctx) ; mean(opt_k) ; mean(ctx) * mean(opt_k) ; |mean(ctx) - mean(opt_k)|] + b` (1,537 parameters at d = 384; 3,073 at d = 768).
C2 Option-only: `score_k = MLP(mean(opt_k))` (d to 256 to 1; 98,817 and 197,121 parameters); the context is never read. Its predictions define the option-prior strata of the secondary accuracy view.
C1 Fine-tuned reference: `microsoft/deberta-v3-base` multiple-choice cross-encoder (pairs of `<context> <question>` and each option), weights loaded in float32 with a non-finite-loss guard, AdamW learning rate 2e-5, weight decay 0.01, batches of 16 items, 5 epochs, 6% linear warm-up, maximum length 128, gradient clipping 1.0, selection on dev accuracy, five seeds; weights hashed on local disk and deleted.

## 6. Training and selection

AdamW, weight decay 0.01, batch 16, at most 30 epochs, early stopping on dev accuracy with patience 5, checkpoint selection on dev accuracy over all dev items (ties broken by lower dev loss), gradient clipping 1.0, bfloat16 autocast for the module forward pass on the GPU with the loss in float32, TF32 matrix products allowed, batches bucketed by context length. Seeds: 13, 42, 71, 7595, 1676. Learning rate 1e-3 for every encoder, aggregator, C0 and C2 (chosen in the pilot by the rule of section 11.2; read by the notebook from the pilot's `lr_grid.json`). No other hyperparameter is tuned. The test split is scored once per run with the selected checkpoint; predictions with probabilities and scores are archived.

## 7. Estimands and subsets (test split)

### 7.1 Primary: decision changes and scope sensitivity

For a scenario, `flip_O_N` = 1 if the predicted option for N differs from the predicted option for O, else 0; `flip_O_S` likewise for S; rates are shares of scenarios in percent. Scope sensitivity `SS = flip_O_N - flip_O_S` (percentage points). SS is computed overall, for the seen families (syntactic, quantifier), for the unseen families (affixal, lexical) and per family. On S-same scenarios (the S cue shares a spaCy sentence with a decisive word; ids in `S_same_ids_test.json`) `flip_O_S` measures spurious decision changes caused by a cue placed near, but not on, the decisive words.

### 7.2 Same-cue analysis (descriptive)

In the seen families, `flip_O_N` and `flip_O_S` restricted to scenarios whose N cue (respectively S cue) is the lemma "not" (not, n't, auxiliary + not; cannot excluded). Proposition 2 predicts equal rates under E0, whatever the cue word's own pairwise scores.

### 7.3 Secondary: accuracy view

As in draft 1: `acc_X` per arm, `acc_OP`, negation gap `acc_OP - acc_N`, scope gap, `acc_X_given_O`, `pair_ON`, consistency, error-confidence profile, S-same accuracy, and every metric per option-prior stratum (C2 majority vote over all C2 runs on the O items; ties congruent).

## 8. Gate and prespecified rules

### 8.1 Primary rules (accepted by the author on 4 October 2026)

Thresholds: `gate_SS_seen` 20; `R1_max_unseen_SS` 5; `R2_min_gain_unseen_SS` 15; `R3_min_gain_flipS` 10; `R3_min_items` 100 (S-same scenarios). Values are percentage points except the count. Rules are evaluated per encoder on seed means; R2 and R3 additionally require the contrast to be positive in all five seeds and its paired seed-level 95% t interval to lie above zero. Scenario-cluster bootstrap intervals are reported alongside and never used to rescue a rule.

G Responsiveness gate (per encoder). A1 seed-mean seen-family SS at least 20. A module that did not learn to change its decision for trained negation in N more than in S cannot inform the rules; for such an encoder R1 to R3 are not evaluated and its results are reported descriptively.

R1 Limit (E0 only; Propositions 1 and 2). If E0 passes G: A1 seed-mean unseen-family SS at most 5.

R2 Supplied scope. For each encoder that passes G: A4 minus A1 unseen-family SS at least 15, positive in all seeds, paired seed interval above zero.

R3 Scope precision. For each encoder that passes G, if S-same has at least 100 scenarios: A6 minus A4 `flip_O_S` on S-same at least 10, positive in all seeds, paired seed interval above zero.

R5 Proposition 1, exact. Zero violations for A1 and A3 under E0 in every seed (section 10).

### 8.2 Descriptive blocks (no rule)

Bridge: A1 seen and unseen SS, `flip_O_N` and `flip_O_S` for every encoder (does a frozen contextual encoder carry scope information that a max aggregator uses, and does it transfer to unseen cue families). R4: A5 against A1 and A4 on SS. D1: dataset-supplied polarity against A1 and A4 on SS. Same-cue analysis (section 7.2). Controls C0, C1, C2.

### 8.3 Secondary rules (draft-1 accuracy view; descriptive)

The accuracy-based gate and rules of draft 1 (gate: A1 `acc_OP` at least 70 and, on the option-prior-incongruent stratum, at least 50; R1 unseen negation gap at least 20 in every seed with stratum balancing; R2 A4 minus A1 `acc_N` at least 15 with `acc_OP` loss at most 2; R3 A4 minus A6 S-same accuracy at least 10) are computed with the same code and reported in `rules_accuracy_secondary.json`. They do not decide any claim.

### 8.4 Outcome map

| Outcome | Reading |
|---|---|
| G not passed | The module did not learn to respond to trained negation; no scope claim for that encoder. |
| R1 met (E0) | As Propositions 1 and 2 predict, a max-aggregated module with context-independent tokens shows no scope sensitivity for cue families absent from training. |
| R1 not met (E0) | A defect in code or data (for example a seen word inside unseen cues) is suspected and investigated before any interpretation. |
| R2 met | Supplying scope to the signed rule produces scope-sensitive decision changes for unseen cue families. |
| R2 not met | The signed rule with this resolver does not; D1 shows whether exact scope would. |
| R3 met | Scope precision matters: a sentence-wide polarity mask changes decisions on irrelevant negations that the dependency scope leaves alone. |
| R3 not met or not evaluable | No evidence in these data that scope precision matters beyond sentence-level polarity. |
| R5 not met | Code defect; the study stops until it is resolved. |

## 9. Uncertainty

Paired seed-level 95% t intervals (five seeds, four degrees of freedom) for every contrast; scenario-cluster percentile bootstrap (10,000 replicates, generator seed 20261003, all arms and seeds of a scenario resampled together); exact two-sided McNemar tests on paired predictions within each seed for the secondary accuracy contrasts. Intervals that include zero are reported as inconclusive, never as equivalence. No family-wise error control is claimed; every rule outcome is reported, met or not.

## 10. Proposition-1 exact test (R5)

Insertion pairs: test scenarios whose S or N context contains every word token of the O context (as a multiset, lower-cased) plus at least one more token. For each pair and each option, the score under the longer context must not be lower than under the O context. Scores are recomputed in float64 on the CPU from the trained E0 modules of A1 and A3 (all seeds), tolerance 1e-9.

## 11. Pilot (train and dev only)

### 11.1 Content

Run `np_pilot_20261004T143238Z` (4 October 2026, Colab A100-SXM4-40GB, package 0.2.4, zip SHA-256 `8cbd3ea1866107657495089209f3a4298ae4eb85047ff44236e7adeab4feacb4`; 38 files, all SHA-256 values verified): dataset audit; resolver diagnostics on dev; encoder caches; Hugging Face pipeline checks with a two-epoch C1 run on dev; Proposition-1 checks; learning-rate grid; A1 and C2 on every encoder. The test split was not read.

### 11.2 What the pilot fixed

Only the learning rate: grid {3e-4, 1e-3, 3e-3}, A1 on E1, seed 13, highest dev accuracy over all dev items (ties to the smaller rate). Dev accuracy 62.4, 70.0 and 69.3; chosen 1e-3.

### 11.3 Stop conditions

S1 dataset integrity, S2 tokenizer or weight-loading failure, S3 C1 check below 75% dev `acc_OP`, S4 any Proposition-1 violation by A1 or A3 under E0. None was met.

### 11.4 Pilot record

Resolver on dev: declared cue detected in 100% of N and S items; resolved scope covers a decisive word in 98.1% of N items (103 with decisive words found) and in 9.5% of S items; the S cue shares a sentence with a decisive word in 94.3% of S items (99 of 105); cues detected in 1.9% of O and 4.8% of P items.
Pipeline: unknown-token rate 0.0 for E1 and E2; no backbone weights missing (E2 reports 11 unexpected pretraining-head keys; the C1 head's classifier and pooler are newly initialized). C1 check (two epochs): dev accuracy 94.3%; `acc_OP` 96.2, `acc_N` 87.6, `flip_O_N` 88.6, `flip_O_S` 5.7.
Proposition 1 on 84 dev insertion pairs (168 option scores): zero violations for five random initializations of A1 and of A3 and for the trained E0 A1 module; A2 (78 to 128) and A4 (168 of 168) violate it, as expected.
A1 and C2 on dev (seed 13, learning rate 1e-3):

| Encoder | A1 `acc_OP` | A1 `acc_N` | A1 `flip_O_N` | A1 `flip_O_S` | A1 SS | A1 train acc | C2 `acc_OP` | A1 `acc_OP`, incongruent stratum |
|---|---|---|---|---|---|---|---|---|
| E0 | 71.4 | 43.8 | 27.6 | 1.0 | 26.7 | 82.3 | 70.5 | 37.1 (31 scenarios) |
| E1 | 73.3 | 62.9 | 70.5 | 14.3 | 56.2 | 96.6 | 72.4 | 32.8 (29 scenarios) |
| E2 | 72.4 | 28.6 | 1.0 | 1.0 | 0.0 | 68.2 | 76.2 | 10.0 (25 scenarios) |

Same-cue analysis on dev (lemma "not" as defined in section 7.2): decision changes in N against S were 1 of 43 against 0 of 24 for E0, 28 of 43 against 5 of 24 for E1, 0 of 43 against 0 of 24 for E2, and 40 of 43 against 0 of 24 for the C1 check (C1 SS 82.9).

### 11.5 Why the rules were revised before freezing

Under draft 1, an encoder had to answer the affirmative arms by reading the context, checked on the scenarios where the option prior is wrong (at least 50%). On dev no encoder came close (37.1, 32.8 and 10.0), while A1's affirmative accuracy matched the context-blind control. The confirmatory run would therefore have evaluated none of the rules. The pilot also showed what the modules do learn: E1 changed its decision for 70.5% of N scenarios and 14.3% of S scenarios, and for the same cue word "not" it separated the two positions (28 of 43 against 5 of 24), whereas E0 did not (1 of 43 against 0 of 24), as Proposition 2 predicts. The propositions are statements about whether and where a cue can change a decision, not about reading the affirmative facts; the frozen version therefore makes scope sensitivity the primary estimand and keeps the accuracy view as secondary. Data, models, training, seeds and controls are unchanged; only the learning rate was taken from the pilot; no test data had been read.

## 12. Compute, provenance, release

Confirmatory grid: 3 encoders by 7 aggregator conditions by 5 seeds = 105 runs, plus C0 and C2 (15 runs each) and C1 (5 runs), on one Colab A100 session (resume-safe; about 40 seconds per module run in the pilot). Every run writes `run_summary.json`, `arm.json`, dev and test predictions with probabilities and scores, and the module state (kept on Drive, not zipped). The run directory carries `run_environment.json`, `CACSA_manifest_v2_0.md` and `manifest_lock.json`, `data_audit.json`, `scope_resolver_test_report.json`, `arm_diagnostics_test.jsonl`, `S_same_ids_test.json`, `encoder_meta.json`, `R5_proposition1_float64.json`, `option_prior_strata_test.json`, `per_run.json`, `seed_means.json`, `table_seed_means_flip.md`, `table_seed_means_accuracy.md`, `rules.json` (primary), `rules_accuracy_secondary.json`, `mcnemar_accuracy_secondary.json` and `FILE_MANIFEST_SHA256.csv`. Code (`cacsa2`, MIT licence), notebooks, NegPairs with its construction record, raw predictions and an independent recomputation of every table are released in a public repository; the dataset licence is decided by the author before release.

## 13. Paper

Structure: Introduction; Related work; the module family and Propositions 1 and 2 (proofs in an appendix); NegPairs (construction in two runs, verification, audit, limitations); Experiments (gate, R1 to R5, bridge, controls, secondary accuracy view); CondaQA as a bounded negative result; Analysis; Limitations; Conclusion; Code and Data Availability; Generative AI Declaration.
Disclosure: NegPairs was generated by `claude-sonnet-5`, verified by `claude-haiku-4-5-20251001` and audited by `claude-opus-5` through the Anthropic API; this is described in the data section and listed in the Generative AI Declaration together with every other use of generative AI tools (analysis-code support and any writing support), as the target journal's policy requires; the authors take responsibility for the content. The revision of section 11.5 is reported in the paper.
Writing rules: the author's fifteen in-paper rules apply. Every number is traced to a file in a run directory; every reference is verified before submission.

## 14. Open items

1. Decide the NegPairs licence.
2. Optional, outside every rule: re-run C1 on CondaQA with the publisher's recipe (learning rate 1e-5, 10 epochs).
3. Optional, before submission: a human check of a random sample of scenarios by the author, reported as such.

## 15. Change log

v2.0 DRAFT 1 (2026-10-03): written after the v1.1 run, the follow-up run and the decision to move the confirmatory test to NegPairs.
v2.0 FROZEN (2026-10-04): dataset counts and construction v2 added (sections 1.4, 4.2, 4.4, 4.5); pilot record added (section 11); primary estimand moved to scope sensitivity and rules rewritten accordingly, with the author's acceptance of the thresholds; draft-1 accuracy rules kept as a secondary view (section 8.3). Code: `cacsa2` 0.2.5.
