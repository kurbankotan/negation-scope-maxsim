# CACSA Study Manifest, version 3.0

Status: FROZEN, 5 October 2026. The author accepted the propositions (section 3), the data design (section 4) and the rules and thresholds (section 8) on 5 October 2026, before any v3.0 data set was generated and before any model was trained under this manifest. The text is DRAFT 1 (written after the confirmatory run of manifest v2.0, frozen 4 October 2026, SHA-256 `cc172368339c385d595110dfc35d60771c6436263284e84a519416d461a0da23`, 32,802 bytes, and revised after an internal critical review) with the implementation details listed in section 15 added while the build code was written and tested offline. This file is shipped inside `cacsa2_project.zip` (0.3.0); the build notebook checks its SHA-256 and records it before its first API call, and the experiment notebook before any training.

Freeze procedure: (1) the author accepts the propositions (section 3), the data design (section 4) and the rules and thresholds (section 8); (2) the status line is changed to FROZEN with the date, and the SHA-256 of the file is computed; (3) recommended: the frozen file is deposited with a public timestamp (Zenodo or OSF) before the build notebook is run, and the identifier is recorded; (4) the build notebook records the manifest's SHA-256 before its first API call, and the experiment notebook before any training. The frozen text is never edited; later changes are amendments with their own version number, hash and date.

Working title of the paper: fixed only after the confirmatory results. Target journal: Computational Linguistics (MIT Press).

Author: Kurban Kotan (Manisa Celal Bayar University). Analysis support: Claude (Anthropic). All runs are executed by the author on Google Colab.

---

## 0. Summary

Manifest v2.0 tested two propositions about modules that score an answer option by aggregating token-pair scores between a context and the option with a maximum over context tokens (CACSA v2), using NegPairs, a set of model-generated minimal-pair scenarios, and three frozen encoders. Two of the three encoders failed the responsiveness gate, the two scope rules that could be evaluated were not met, and the exact monotonicity rule was met (section 1.2).

Version 3.0 widens the study in five directions, each aimed at a weakness that v2.0 left open.

1. Propositions. Three statements (section 3) replace Propositions 1 and 2 of v2.0 and contain them as special cases. Proposition A: with context-independent token states, every aggregation that treats context positions symmetrically (maximum, mean, log-sum-exp, a learned token gate) sees the context only as a bag of tokens, and this remains true when a mask is added that ignores position. Proposition B: if the aggregation is also monotone (maximum, log-sum-exp), adding tokens anywhere can never lower a score. Proposition C: a signed maximum whose mask tracks the exact negation scope can separate arms that Proposition A makes inseparable. With static states and a position-blind mask, the aggregator's input is a function of the token bag; positional information has to enter through the states (contextual encoders) or through a position-aware mask. The statements are elementary; their use is that they turn expectations into exact predictions for specific test items that hold for every parameter value, training set and seed.
2. Data. NegPairs-Perm (new): permutation minimal pairs whose N and S arms contain exactly the same words, the negation edit sitting in the decisive clause in N and in an irrelevant clause in S. Proposition A predicts zero scope sensitivity and zero joint accuracy on these pairs for every unmasked module over context-independent states (and for every module whose mask ignores position), so any separation by a contextual encoder can only come from position. NegPairs-Fresh (new): a test set built with the v2.0 pipeline after this manifest is frozen, because the v1 test split has been read.
3. Models. Seven frozen encoders instead of three: two context-independent (the hash encoder of v2.0 and GloVe) and five contextual (the two of v2.0, a sentence encoder, a retrieval encoder, and ColBERTv2), a learned MaxSim head without the CACSA affinity network (M1) and zero-shot MaxSim baselines (M0, M0-native), so that the results concern the MaxSim family rather than one module.
4. External test. NevIR (Weller et al., 2024), a published negation benchmark with human-written queries: Proposition B implies that every instance whose two documents are nested as bags of tokens is failed by every static monotone MaxSim scorer; the share of such instances and the behaviour of contextual scorers on them are reported.
5. Robustness. Ten seeds instead of five; a learning rate chosen per encoder on dev; the responsiveness gate moved to dev, so that the set of evaluated encoders is fixed before any test metric is computed; a correction for evaluating a rule on several encoders; a second verifier from another model family (Qwen2.5-7B-Instruct) as an acceptance condition for the new data; a probing analysis of whether the frozen states encode on which clause a negation sits.

The thresholds of the v2.0 gate and of R2 and R3 are carried over unchanged. The exact rule R5 is extended, two exact rules are new (R6, R7), and one statistical rule on permutation pairs is new (R8). R1 becomes descriptive (section 8.4).

## 1. History

### 1.1 Manifest v1.1 (CondaQA)

The v1.1 run on CondaQA and its follow-up are described in manifest v2.0, sections 1.1 to 1.3. CondaQA remains a bounded negative result and is not used by any rule of this manifest.

### 1.2 Manifest v2.0 confirmatory run

Run `np_main_20261004T154358Z` (4 October 2026, package `cacsa2` 0.2.5) was executed on an NVIDIA Tesla T4 instead of the planned A100 (hardware was not part of any rule): 140 runs, summed training time 1.83 hours, 677 files listed with SHA-256 values; an independent script recomputed every per-run metric from the raw test predictions. Outcomes on the NegPairs v1 test split (seed means over five seeds; percentage points):

- Gate G (A1 seen-family SS at least 20): passed by E1 (53.07); failed by E0 (11.82) and E2 (8.98). R1 was therefore not evaluated, and R2 and R3 were evaluated for E1 only.
- R1 (descriptive for E0): A1 unseen-family SS −1.77.
- R2 (E1): A4 minus A1 unseen-family SS +11.93, positive in five of five seeds, paired seed interval [3.45, 20.40]; below the threshold of 15, not met.
- R3 (E1, 545 S-same scenarios): A6 minus A4 `flip_O_S` −0.92, paired seed interval [−6.08, 4.25]; not met.
- R5 (Proposition 1, exact): met; zero violations in 10 module runs (A1 and A3 under E0, five seeds), each with 760 option scores on 380 insertion pairs from 362 scenarios.
- Bridge (descriptive): E1 A1 unseen-family SS 13.65 (scenario-cluster interval [10.05, 17.19]).
- Same-cue analysis for "not" (69 N and 42 S scenarios): decision changes in N against S were 26.96 against 26.19 for E0 A1 and 71.30 against 37.62 for E1 A1.
- D1 (descriptive): E1 D1 minus A1 unseen-family SS +25.05 (paired seed interval [16.43, 33.68]).
- C1 (fine-tuned cross-encoder): seen-family SS 89.55, unseen-family SS 86.56, accuracy 94.17.

### 1.3 What v2.0 left open

(a) The propositions were tested only indirectly. In NegPairs v1 the N and S arms of a scenario use different cue words, so a module with context-independent states can separate them through the identity of the cue and edited words (E0 A1 seen-family SS 11.82), and the limit rule R1 depended on a gate that E0 did not pass. A test whose N and S arms contain the same words was missing. (b) Two of three encoders failed the gate, so the scope rules rested on one encoder and five seeds (four degrees of freedom); the learning rate had been chosen on one encoder and one seed; the gate was computed on the test split. (c) The v1 test split has been read; rules decided on it again would not be confirmatory. (d) All data were generated and verified by models of one family, without human verification. (e) The module was presented as CACSA only, although the same aggregation (a maximum over the tokens of one text, averaged or summed over the tokens of the other) underlies late-interaction retrieval (ColBERT; Khattab and Zaharia, 2020) and greedy matching in BERTScore (Zhang et al., 2020). (f) No external benchmark was used.

## 2. Scope and non-claims

In scope: scorers of the form `S(o; x) = Ψ_j A(Z_j)` over frozen token states (section 3.1); seven frozen encoders (two context-independent); eight trained conditions (A1 to A6, D1, M1), the zero-shot baselines M0 and M0-native, and three controls; NegPairs v1 (train and dev for training, selection and the gate; test reported descriptively and used by the exact rule R5), NegPairs-Fresh (confirmatory test), NegPairs-Perm (permutation pairs, test only), the NevIR test split (external); ten seeds.

Not claimed: anything about large language models or about pragmatic inference in general; that any variant is competitive with fine-tuned cross-encoders; that the propositions describe contextual encoders (they do not, section 3.5; the bridge to contextual encoders is empirical); that Proposition C says anything about learnability (it is an existence statement); that A4 is free of hand-made resources (it uses a cue lexicon and a dependency parser); that a decision change is a correct answer; that the modules would behave alike if they were trained on permutation pairs (they are trained on standard minimal pairs only); that results transfer beyond the four negation families, beyond English, or to human-written text, except for the NevIR check, whose queries were written by crowdworkers on CondaQA passages; anything about the retrieval quality of ColBERTv2 beyond NevIR; that NegPairs-Perm sentences are a sample of natural text (they repeat a word across two clauses by design and serve as a diagnostic). NegPairs (all parts) is model-generated and model-verified; the new parts are verified by two model families; no part is human-verified (author's decision of 5 October 2026). This is stated wherever the data are used.

## 3. Formal setting and propositions

### 3.1 Setting

A context is a token sequence `x = (x_1, ..., x_n)` over a vocabulary `V` (in NegPairs the jointly encoded `<context> <question>`; in NevIR a document). An option is a sequence `o = (y_1, ..., y_m)` (in NevIR a query).

Encoder. Token states `h_i(x) ∈ R^d`; option states `u_j(o)` are computed from the option alone (true for every model here except C1). The encoder is static (context-independent) if there is a map `φ: V → R^d` with `h_i(x) = φ(x_i)` for every `x` and `i`. E0 and E3 are static; their tokens are those of the tokenization `\w+|[^\w\s]`, and `φ` depends on the lower-cased token, so bags are taken over lower-cased tokens. E1, E2, E4, E5 and E6 are contextual.

Masks. Each position carries a mask vector `μ_i(x) ∈ {0,1}^k`; `k = 0` for unmasked rules. A4: `μ_i = 1` if token `i` lies inside the scope returned by the resolver; A6: inside a sentence that contains a detected cue; D1: inside the scope supplied by the dataset.

Pair scores and aggregation. `c_ij = g(h_i, u_j)` for a fixed (trained) function `g`. For each option token `j`, `a_j = A(Z_j)` with `Z_j(x, o) = ⟦(h_i(x), μ_i(x)) : i = 1, ..., n⟧`, a multiset; `A` may use `u_j` and the parameters, but it reads the context positions only through this multiset (`A` is symmetric). Every rule of section 5.4 has this form: A1, M0 and M1 `max_i c_ij`; A2 `mean_i c_ij`; A3 `τ log Σ_i exp(c_ij / τ)`; A4, A6 and D1 `max_{i: μ_i = 0} c_ij − β max_{i: μ_i = 1} c_ij` (an empty maximum counts as 0, as implemented); A5 `max_i (1 − p_i) c_ij − β max_i p_i c_ij` with `p_i = σ(wᵀh_i + b_0)`.

Option score and decision. `S(o; x) = Ψ(a_1, ..., a_m)` for a fixed `Ψ` (the mean for every module here; the sum in ColBERT). For two options the prediction is the option with the higher score, ties going to the first option (any fixed rule that depends only on the two scores gives the same statements). Output probabilities are the softmax of the two scores. The statements of this section hold in exact arithmetic; section 10 states how the exact rules treat rounding.

Bags. `Bag(x) = ⟦x_1, ..., x_n⟧` and `Bag_μ(x) = ⟦(x_i, μ_i(x)) : i = 1, ..., n⟧`; `⊆` between multisets compares multiplicities.

### 3.2 Proposition A (bag invariance)

Statement. If the encoder is static, then for every option `o`, `S(o; x)` depends on `x` only through `Bag_μ(x)`. With `k = 0` it depends only on `Bag(x)`.

Proof. For a static encoder, `Z_j(x, o)` is the image of `Bag_μ(x)` under the map `(v, m) ↦ (φ(v), m)`. Two contexts with the same `Bag_μ` therefore give the same multiset `Z_j` for every `j`, hence the same `a_j`, because `A` is a function of `Z_j`, and the same `S(o; x)`, because `Ψ` is a function of the `a_j`. ∎

Corollary A.1 (permutation pairs). Let the encoder be static and the rule unmasked: A1, A2, A3, A5, M0, M1, and also the control C0, whose only context input is the mean of the token states (the argument of the proof applies unchanged). If two contexts satisfy `Bag(x^N) = Bag(x^S)`, every option receives the same score under both, so the predictions and output probabilities coincide. For a set of permutation minimal pairs whose gold answers differ (`1 − y` for N, `y` for S), it follows, for every parameter value, training set and seed: `flip_O_N = flip_O_S` in every scenario, so `SS_perm = 0`; exactly one of N and S is answered correctly in every scenario, so the joint (N, S) accuracy is 0 and `acc_N + acc_S = 100`.

Corollary A.2 (masks that ignore position). If `μ_i(x)` depends only on the token `x_i` (for example a lexicon mask that marks cue words), then `Bag_μ(x)` is a function of `Bag(x)` and Corollary A.1 holds for the masked rule. Marking the cue without its scope cannot separate permutation pairs.

Corollary A.3 (sentence-level masks). A6 marks every token of each sentence that contains a detected cue, with sentences as segmented by the resolver's parser on the whole `<context> <question>` string. If the marked tokens of N and of S form the same multiset (as when the parser segments both arms alike and the single context sentence contains a detected cue in both), then `Bag_μ(x^N) = Bag_μ(x^S)` and Corollary A.1 holds for A6.

Remark A.4. Proposition 2 of v2.0 is Proposition A with `k = 0`. On standard minimal pairs (NegPairs v1 and NegPairs-Fresh) N and S differ in their tokens: the cue words differ, and an N edit may replace a decisive word (for example `available` becomes `unavailable`) while S keeps it. Proposition A therefore allows a static module to separate N from S through token identity, and scope sensitivity on such pairs mixes the identity of cue and edited words with their position. Permutation pairs remove the identity component by construction.

### 3.3 Proposition B (inclusion monotonicity)

Statement. Let the encoder be static, `k = 0`, `A` monotone under multiset inclusion (`Z ⊆ Z'` implies `A(Z) ≤ A(Z')`) and `Ψ` nondecreasing in each argument. If `Bag(x) ⊆ Bag(x')`, then `S(o; x) ≤ S(o; x')` for every option `o`.

Proof. `Bag(x) ⊆ Bag(x')` implies `Z_j(x, o) ⊆ Z_j(x', o)` for every `j`, because both are images of the bags under the same map. Monotonicity of `A` gives `a_j(x) ≤ a_j(x')`, and monotonicity of `Ψ` gives the claim. ∎

Which rules qualify. The maximum (A1, M0, M1) does. Log-sum-exp (A3) with fixed `τ > 0` does, because adding a term increases `Σ exp(c / τ)`. The mean (A2) does not. The signed rules (A4, A5, A6, D1) do not, because a token that enters the subtracted maximum can lower the score. Proposition 1 of v2.0 is the special case in which `x'` is `x` with tokens inserted; Proposition B needs only inclusion of bags, so the inserted tokens may also be reordered.

Corollary B.1 (nested documents; NevIR). Let a relevance score be `s(q, d) = Ψ_j A(Z_j(d, q))`, with the document as context (the maximum runs over document tokens) and the query as option, a static encoder, monotone `A` and nondecreasing `Ψ`. If `Bag(d_a) ⊆ Bag(d_b)`, then `s(q, d_a) ≤ s(q, d_b)` for every query `q`, so no query can rank `d_a` strictly above `d_b`. A NevIR instance (query `q_1` relevant to `d_1`, query `q_2` relevant to `d_2`) counts as correct only if each query ranks its own document strictly first (the benchmark's evaluation code uses a strict comparison). Hence every instance whose documents are nested (`Bag(d_1) ⊆ Bag(d_2)` or `Bag(d_2) ⊆ Bag(d_1)`; equal bags included) is failed by every such scorer, for every parameter value.

Remark B.2 (BERTScore). With static token embeddings, BERTScore recall (the mean over reference tokens of the maximum similarity over candidate tokens) satisfies Proposition B for insertions into the candidate. Precision (the mean over candidate tokens) does not, because inserted tokens enter the average. BERTScore as published uses contextual embeddings (Zhang et al., 2020), to which neither statement applies.

### 3.4 Proposition C (possibility with scope-tracking masks)

Setting. A scenario has arms O, P, N and S, options `o^y` (correct in O, P and S) and `o^{1−y}` (correct in N), and a one-bit mask (`k = 1`). Let `R ⊆ V × V` be a relation ("token `v` supports option token `w`"). The scenario is R-separable if (i) every arm contains a token that supports some token of `o^y`; (ii) no token of any arm supports a token of `o^{1−y}`; (iii) in N every supporting token lies inside the mask, and in O, P and S none does. R-separability requires in particular that `o^y` contains a token type that `o^{1−y}` lacks, and that no token occurring outside the mask in N (the question, the other clause, and in permutation pairs the unedited occurrence of the shared word) supports `o^y`.

Statement. Let `c_ij = 1` if `(x_i, y_j) ∈ R` and `c_ij = 0` otherwise, and `β > 0`. The signed maximum (as in A4 and D1) predicts `o^y` for O, P and S and `o^{1−y}` for N in every R-separable scenario.

Proof. Every pair score of `o^{1−y}` is 0, so `S(o^{1−y}) = 0` in every arm. Let `J` be the nonempty set of tokens of `o^y` that are supported in the arm. In O, P and S, for `j ∈ J` the maximum over unmasked tokens is 1 and the maximum over masked tokens is 0, so `a_j = 1`; for `j ∉ J`, `a_j = 0`; hence `S(o^y) = |J| / m > 0`. In N, for `j ∈ J` the unmasked maximum is 0 and the masked maximum is 1, so `a_j = −β`; for `j ∉ J`, `a_j = 0`; hence `S(o^y) = −β |J| / m < 0`. ∎

Corollary C.1 (permutation pairs are separable in principle). If the encoder maps distinct token types to distinct states (E0 almost surely; E3 for tokens with distinct vectors), the pair scores of Proposition C are a function of `(h_i, u_j)`. An R-separable permutation pair is then separated by the signed maximum even with a static encoder, because its masks differ between N and S, whereas Corollary A.1 rules this out for every unmasked rule and Corollaries A.2 and A.3 for masks that ignore position or stop at the sentence. Proposition C's scorer is itself symmetric over (state, mask) pairs; it separates the arms only because its mask carries positional information.

Remark C.2 (what Proposition C does not say). It is an existence statement for a chosen relation `R`. It does not say that the CACSA pair network, with its finite width, represents a given `R`, that training finds such scores, or that the resolver's masks satisfy (iii). D1 (dataset scope) and A4 (resolver scope) measure this empirically.

### 3.5 Contextual encoders and cross-encoders

No exact statement is made. With a contextual encoder `h_i` depends on the whole sequence, so permuted or extended contexts generally change the states, and the premises of Propositions A and B fail. Whether the states carry scope information in a form that a maximum can use is the empirical bridge question (rule R8; the probing analysis of section 7.5). C1 reads the context and the option jointly and lies outside the setting.

### 3.6 Which statement applies to which condition

| Condition | Static encoders (E0, E3) | Contextual encoders (E1, E2, E4, E5, E6) |
|---|---|---|
| A1, M0, M1 (maximum) | Proposition A (permutation pairs), Proposition B (insertion pairs, nested NevIR documents) | none |
| A3 (log-sum-exp) | Propositions A and B | none |
| A2 (mean), A5 (learned gate), C0 (mean pooling) | Proposition A | none |
| A6 (sentence-wide mask) | Corollary A.3 on permutation pairs with equal marked multisets | none |
| A4 (resolver scope), D1 (dataset scope) | Proposition C (possibility only) | none |
| C2 (option only) | context not read; invariant trivially | same |
| C1 (cross-encoder) | outside the setting | outside the setting |

## 4. Data

### 4.1 NegPairs v1 (unchanged)

Files, construction and counts as in manifest v2.0, sections 4.1 to 4.5 (1,088 scenarios; train 423, dev 105, test 560; seen families syntactic and quantifier in all three splits, unseen families affixal and lexical in the test split only). Train and dev are the only training and selection data of v3.0; dev also carries the gate G. The v1 test split was read in the v2.0 run; in v3.0 every condition is scored on it again and reported descriptively as "v1 test (previously read)". No statistical rule of v3.0 uses it; the exact rule R5 includes its insertion pairs.

### 4.2 NegPairs-Fresh (new confirmatory test set)

Construction v2 of manifest v2.0, section 4.2, unchanged: generator `claude-sonnet-5` with the unchanged generation prompt (SHA-256 `8270c1684edaadff9941da5244a8812e7fcdc62b95ada62ea21bfa38e668af35`) at effort low with a 4,000-token limit; `structural_filter_v2`; blind verification of each arm by `claude-haiku-4-5-20251001` at temperature 0 with the v2 verifier prompt (SHA-256 `1ce31ce47f63c8c7903b22ba4869b5a99c0262a2f9356fec52faa117ce8c3fdd`). Added: (a) a new schedule (seed 20261005; at most 480 scheduled candidates per family; identifiers prefixed `fresh-`); (b) the near-duplicate check runs against all NegPairs v1 scenarios, all accepted NegPairs-Perm scenarios and earlier accepted Fresh scenarios (section 4.6); (c) cross-family verification as an acceptance condition (section 4.4). Cap: 60 accepted scenarios per family (240 in total). All scenarios form the split `fresh` (test only); seen and unseen families as in v1.

### 4.3 NegPairs-Perm (new; permutation minimal pairs)

Design. The context is one sentence made of two coordinated clauses: a decisive clause, whose content decides which option is better, and another clause with a detail that does not matter for the choice. Each clause contains one edit site of the same kind (the same word W, or an insertion point after an auxiliary or copula). N applies one negation edit at the site in the decisive clause; S applies the same edit at the site in the other clause. N and S therefore contain the same words and differ only in where the edit sits. O and P are as in v1 (affirmative context and a paraphrase without negation). Answers: O, P and S the answer `y`; N the other option.

Patterns (one per family; closed lists except for the affixal word):

| Family | Pattern | Edit site in both clauses | Edit |
|---|---|---|---|
| syntactic (seen) | not-insertion | the position directly after an auxiliary or copula: is, are, was, were, can, will, should, has, have | insert "not" |
| quantifier (seen) | determiner | the same determiner: a, an or some | replace by "no" |
| affixal (unseen) | prefix | the same adjective or participle W, not preceded by "a" or "an" | replace by W with a prefix un, in, im, il, ir, dis or non (closed form, no hyphen) |
| lexical (unseen) | lexical pair | the same word: has, have, had, with or present | replace by lacks, lack, lacked, without or absent, respectively |

Clause order. Within each family, the clause order (decisive clause first or second) of the scheduled candidates is assigned by a seeded shuffle of a balanced list (seed 20261022), independently of domain and situation. Acceptance is capped per family and clause order (below), and rule R8 uses an estimand that weights the two orders equally (section 7.2). The first word of the sentence is never an edit site, so capitalization is the same in N and S.

Generation (module `cacsa2.perm_build`, package 0.3.0). Schedule seed 20261020; balanced over domain and situation as in v1 (same 14 domains, 8 situations and banned-topic list); at most 600 scheduled candidates per family; identifiers prefixed `perm-`. Generator `claude-sonnet-5` at effort low with a 4,000-token limit and the prompt of Appendix A, whose SHA-256 values are listed there and recomputed by the build notebook before its first call. The generator returns the question, the two options, the answer for the unmarked context, the context with both edit sites marked, the connector between the clauses, the decisive words, the negated form and the paraphrase P.

Assembly (deterministic). O is the marked context with the marks removed. The decisive site is the first mark if the schedule asked for the decisive clause first and the second mark otherwise. N applies the edit at the decisive site and S at the other site: insertion of "not" for the syntactic pattern, replacement of the marked word by the negated form otherwise. Whitespace is normalized to single spaces after every operation.

Deterministic filters (`perm_filter`; a candidate is discarded at the first failing check):

1. Required fields present; `usable` true; answer "A" or "B".
2. Options: two distinct strings of 1 to 6 words without a core negation marker (`CORE_NEG` of `negpairs_build`), whose bags under the E0 tokenizer differ; the question ends with a question mark and contains no core negation marker.
3. Exactly two marks, each standing alone between spaces (no punctuation or other characters attached), so that the whitespace-delimited chunks of N and S form the same multiset; the first word of the sentence is not marked; the connector occurs exactly once between the two marks, matched as whole words (or as the exact punctuation string). The clause spans are the text before and after the connector.
4. Pattern: syntactic, both marks are insertion marks, each directly after a word of the auxiliary list; quantifier, both marked words are the same word of {a, an, some} and the negated form is "no"; affixal, both marked words are the same lower-case word W, the negated form equals one prefix of {un, in, im, il, ir, dis, non} followed by W exactly, it has a Zipf frequency of at least 2.0 in `wordfreq` 3.1.1 (English), it is not on the package's stoplist of prefixed words that do not negate their base (for example invaluable, infamous, inflammable, discovered, disclosed, dispatched, dismissed, discharged), and neither site is preceded by "a" or "an"; lexical, both marked words are the same word and (word, negated form) is one of (has, lacks), (have, lack), (had, lacked), (with, without), (present, absent).
5. The cue ("not" or the negated form) passes `cue_in_family` for the family.
6. O and P contain no core negation marker and no word that consists of "un" followed by a word with a Zipf frequency of at least 3.0, apart from the exceptions listed in `cue_in_family` (a cheap check for affixal negation; other prefixes are left to the verifiers because of their many false positives).
7. Every arm has at most 30 words.
8. The decisive words (after removing any marks the generator copied into them) occur in O inside the decisive-clause span.
9. The word edit distance (`negpairs_build.words`) from O to N and from O to S is exactly 1.
10. `Bag(N) = Bag(S)` under the E0 tokenizer (lower-cased) and under the case-preserving word tokenizer `[A-Za-z0-9']+|[^\sA-Za-z0-9']`, and N differs from S.
11. The word edit distance from O to P is at least 2, and P introduces no name or number that is absent from O (`new_entities` of v2.0).

Verification and acceptance: blind verification of each arm by `claude-haiku-4-5-20251001` with the v2 verifier prompt (expected: the answer for O, P and S, the other option for N; natural), then cross-family verification (section 4.4), then the near-duplicate check and the caps (section 4.6). Cap: 38 accepted scenarios per family and clause order (76 per family, 304 in total). All scenarios form the split `perm` (test only).

Item format: one row per arm as in v1 (`scenario_id`, `arm`, `item_id`, `family`, `seen_family`, `domain`, `situation`, `split`, `context`, `question`, `options`, `label`, `cue`, `decisive`), plus a field `perm` with the pattern, W, the negated form, the clause order, the character spans of the decisive clause and the other clause in every arm, and the word positions of the edit in N and S.

D1 masks for NegPairs-Perm are built from the stored spans, not by searching the decisive words in the whole context (W occurs in both clauses): in N, the decisive words inside the decisive-clause span plus the cue; in S, the cue only; none in O and P (the v1 convention). The resolver diagnostics of NegPairs-Perm use the same spans (section 5.5).

### 4.4 Cross-family verification (Qwen2.5-7B-Instruct)

Model `Qwen/Qwen2.5-7B-Instruct` (Apache 2.0; 7.6 billion parameters; weights in bfloat16 on a GPU with bfloat16 support and at least 22 GiB of memory as reported by the driver, such as an L4 or an A100; the build notebook stops on other hardware; revision resolved and recorded). Each arm is shown alone with the prompt of Appendix B as the user message, formatted with the model's own chat template (default system message) and its generation prompt. The verdict is read from the next-token logits of the tokens "A", "B" and "U", computed in float32 from the final hidden state and the corresponding rows of the output layer; the highest of the three wins (no sampling, no free text). Each arm is presented twice, with the options in both orders; the arm's verdict is the option chosen in both orders, and "inconsistent" otherwise (including "U" in either order, and exact ties). A scenario passes if the verdict equals the expected option in all four arms. All logits are archived.

Use: for NegPairs-Fresh and NegPairs-Perm, passing is an acceptance condition (after the Haiku verification). Because it removes scenarios on which the two model families disagree, the new sets may be easier for language models than NegPairs v1; the discard counts are reported. For NegPairs v1 (all 1,088 scenarios) the same procedure is an audit run in the experiment notebook: agreement is reported per arm, family and split, and the v1 test results are recomputed on the scenarios that pass (descriptive, section 7.7).

### 4.5 Audit and human verification

A seeded 10% sample (seed: the set's schedule seed plus 7, as in v2.0) of the accepted scenarios of each new set is re-verified by `claude-opus-5` with the v2 verifier prompt on all four arms (descriptive; agreement and naturalness, as in v2.0). No human verification is performed (author's decision, 5 October 2026); the paper states this, together with the two-family verification, wherever the data are used. If the author later decides on a human check, it follows open item 6 (section 14) and is reported descriptively; it never changes data or rules.

### 4.6 Order, near-duplicates, caps and budget

The build runs on one GPU runtime (Qwen is applied to each candidate as soon as it has passed Haiku). Order: NegPairs-Perm first, then NegPairs-Fresh. Candidates are processed in schedule order in batches; within a batch, API calls run concurrently, and acceptance is decided in schedule order, so the result does not depend on the order in which calls return.

Near-duplicates: a candidate is discarded if its context O equals (after lower-casing and word tokenization) that of a v1 scenario or of an accepted new scenario, or if the Jaccard similarity of its content words (context O and options) with any of them in the same domain is at least 0.6 (`NearDuplicateIndex` of v2.0, seeded with all 1,088 v1 scenarios). The check runs before the Haiku verification against everything accepted so far and again at acceptance.

Caps and stopping: a candidate is accepted if it passed every check of its set and its stratum (family for Fresh; family and clause order for Perm) is below the cap. Generation for a stratum stops when its cap is reached or its schedule is exhausted. Operational safeguard: a phase that has accepted no candidate after 72 processed candidates stops for inspection; this protects the budget and never changes which candidate is accepted.

Budget: the metered API cost (token usage reported by the API multiplied by the price table recorded in the build notebook) is checked before every call. The Perm phase stops at a cumulative 12.00 USD and the Fresh phase at a cumulative 22.00 USD; the Opus audit runs within the remainder up to a hard cap of 24.00 USD for the whole build (the audit sample is reduced if needed, and the reduction is recorded). If a set ends below its caps it is used as it is; rules whose subsets fall below the minimum sizes of section 8.1 are then reported as not evaluable.

### 4.7 Splits, files and integrity checks

Option order is randomized per scenario as in v1 (generator `Random(schedule seed + 1)`, scenarios sorted by identifier). Seeds: NegPairs-Fresh schedule 20261005, option order 20261006, audit sample 20261012; NegPairs-Perm schedule 20261020, option order 20261021, clause order 20261022, audit sample 20261027. Each new set ships with `build_report.json` (counts per stage, family and clause order, discard reasons, costs, model identifiers and revisions, prompt hashes, schedule seeds, the manifest's SHA-256), its construction record and `FILE_MANIFEST_SHA256.csv`. Every notebook verifies the files against the manifests. The build notebook asserts that every NegPairs-Perm scenario satisfies `Bag(N) = Bag(S)` under the E0 tokenizer and records the assertion in the build report. After the pilot (section 11), when the test files are parsed, the experiment notebook repeats this assertion, asserts that no scenario identifier occurs in two sets and that v1 train and dev contain seen families only, and records, for every encoder tokenizer, which Perm scenarios have equal token multisets in N and S. Analyses of NegPairs-Perm for an encoder, including R8, use only the scenarios with equal token multisets under that encoder's tokenizer (expected: all; exceptions are listed).

### 4.8 NevIR (external test)

Source: `orionweller/NevIR` on Hugging Face (MIT licence; revision recorded), test split only: 1,383 instances according to the Hugging Face dataset viewer on 5 October 2026 and Table 1 of Weller et al. (2024); the count is recorded at run time. Each instance has two queries and two documents; `q1` is relevant to `doc1` and `q2` to `doc2`. The documents are CondaQA passages and their edited versions (scope and affirmative edits; paraphrase edits were excluded by the benchmark's authors); the queries were written by crowdworkers (Weller et al., 2024).

Paired accuracy as in the benchmark's evaluation code: an instance is correct if `s(q1, doc1) > s(q1, doc2)` and `s(q2, doc2) > s(q2, doc1)`; ties are errors. For the static scorers (E0, E3) every NevIR accuracy is computed from float64 scores with a margin: a document counts as ranked first only if its score is higher by more than 1e-9. Published test values (Weller et al., 2024, Table 2): random 25, ColBERTv1 19.7, ColBERTv2 13.0; the benchmark's code evaluated ColBERT checkpoints with its own configuration, which differs from the checkpoint metadata used for M0-native here, so these values are reported for orientation only and used by no rule.

Nested subset: instances with `Bag(doc1) ⊆ Bag(doc2)` or `Bag(doc2) ⊆ Bag(doc1)` under the E0 tokenizer (lower-cased; no truncation). Its size, and the number of instances with equal bags, are reported.

## 5. Models

### 5.1 Encoders (frozen; no gradient)

| ID | Model | Kind | d | Licence |
|---|---|---|---|---|
| E0 Hash | SHA-256-seeded Gaussian unit vector per lower-cased token type (salt `cacsa2-e0`), as in v2.0 | static | 384 | (code) |
| E3 GloVe | `glove.6B.300d` (Pennington et al., 2014; Wikipedia 2014 and Gigaword 5, 6 billion tokens, 400,000 uncased words), file `glove.6B.zip` of the Hugging Face repository `stanfordnlp/glove` | static | 300 | Public Domain Dedication and License 1.0 |
| E1 MiniLM | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, as in v2.0 | contextual; paraphrase sentence encoder | 384 | Apache 2.0 |
| E2 DeBERTa | `microsoft/deberta-v3-base`, as in v2.0 | contextual; masked language model | 768 | MIT |
| E4 MPNet | `sentence-transformers/all-mpnet-base-v2` | contextual; sentence encoder | 768 | Apache 2.0 |
| E5 BGE | `BAAI/bge-base-en-v1.5` | contextual; retrieval bi-encoder | 768 | MIT |
| E6 ColBERTv2 | `colbert-ir/colbertv2.0` (Santhanam et al., 2022): BERT-base followed by the checkpoint's linear projection | contextual; late-interaction retriever | 128 | MIT |

E3 uses the tokenizer of E0. Each lower-cased token is mapped to its GloVe vector, L2-normalized; a token outside the GloVe vocabulary is mapped to an E0-style hash vector in 300 dimensions with salt `cacsa2-e3-oov`. The out-of-vocabulary rate is recorded per data set. E3 is static.

E1, E2, E4 and E5 as E1 and E2 in v2.0: last-layer token states, special tokens removed, every state L2-normalized, maximum length 128 tokens for NegPairs and 512 tokens for NevIR (truncation recorded), inference in bfloat16 on the GPU, revision resolved and recorded.

E6 for the modules and for M0: context and options are encoded in ColBERTv2's document mode (the marker token `[unused1]` placed after `[CLS]`, as in the checkpoint's document tokenizer; no query augmentation); the last-layer states pass through the checkpoint's projection to 128 dimensions and are L2-normalized; `[CLS]`, `[SEP]` and the marker are removed afterwards; punctuation is kept; maximum length 128 tokens for NegPairs and 512 tokens for NevIR (truncation recorded). These are the token embeddings that ColBERTv2 compares with MaxSim. Native ColBERTv2 scoring is defined in section 5.4.

Token states are cached in half precision on the runtime's local disk and shared by every module variant and control, as in v2.0. For E0 and E3 the exact rules (section 10) and every metric on NegPairs-Perm and NevIR are computed from float64 scores on the CPU with the vectors regenerated in float64; for NevIR, E0 and E3 encode every token of a document (no length limit; the notebook asserts that no truncation occurred).

### 5.2 Input formulation

Unchanged from v2.0: context and question encoded jointly as `<context> <question>`, each option separately; `S(o) = mean over the option's tokens j of A_i(c_ij)` over the jointly encoded tokens; prediction by the higher score; training by cross-entropy over the two option scores. For NevIR the document takes the role of the context and the query the role of the option.

### 5.3 Shared module

The CACSA v2 core of v2.0, section 5.3, unchanged. Parameter counts are recorded at run time for every dimension (384, 768, 300 and 128).

### 5.4 Aggregators and MaxSim baselines

A1 to A6 and D1 exactly as in v2.0, section 5.4. New:

| ID | Rule | Training | Parameters |
|---|---|---|---|
| M1 learned MaxSim | `c_ij = (W h_i)ᵀ(W u_j) / sqrt(128)` with a learned linear map `W` (d to 128, no bias), maximum over context tokens, mean over option tokens; the minimal late-interaction head on a frozen encoder, without the affinity network | as A1 | 128·d |
| M0 zero-shot MaxSim | `c_ij = h_iᵀu_j` (cosine; the states are unit vectors), maximum over context tokens, mean over option tokens | none (scored once, no seeds) | 0 |
| M0-native (E6 only) | ColBERTv2's own scoring with the checkpoint's settings (`artifact.metadata`: query length 32 with `[MASK]` augmentation and no attention to the mask tokens, document length 180, dimension 128, cosine similarity, punctuation masked in documents): the option (NegPairs) or query (NevIR) as query, the context or document as document; score = sum over query embeddings of the maximum over document embeddings | none | 0 |

### 5.5 Scope resolver

Unchanged (cue lexicon and scope rules frozen since `cacsa2` 0.1.1; spaCy `en_core_web_sm` 3.8.0; masks mapped to encoder tokens by character overlap). Diagnostics as in v2.0 for NegPairs-Fresh. For NegPairs-Perm, computed with the stored clause spans: whether the resolved scope covers the decisive words of the decisive clause in N, whether it covers words of the other clause in N (leakage), whether it covers the edited word's clause in S, and whether it covers decisive words in S. The resolver is not tuned on any v3.0 data.

### 5.6 Controls

C0 head-only and C2 option-only per encoder, definitions unchanged (parameter counts recorded at run time). C1 fine-tuned reference: recipe of v2.0, section 5.6, unchanged, ten seeds; scored on v1 test, NegPairs-Fresh and NegPairs-Perm.

## 6. Training and selection

As in v2.0, section 6: AdamW, weight decay 0.01, batch 16, at most 30 epochs, early stopping on dev accuracy with patience 5, checkpoint selection on dev accuracy over all dev items (ties broken by lower dev loss), gradient clipping 1.0, bfloat16 autocast for the module forward pass on the GPU with the loss in float32, TF32 matrix products allowed, batches bucketed by context length. Changes:

- Seeds: 13, 42, 71, 7595, 1676, 101, 202, 303, 404, 505 (the first five are those of v2.0).
- Learning rate per encoder, fixed in the pilot (section 11) from the grid {3e-4, 1e-3, 3e-3} with A1 and seeds 13, 42 and 71; criterion: mean dev accuracy over all dev items; ties go to the smaller rate. The chosen rate applies to every trained condition of that encoder (A1 to A6, D1, M1, C0, C2). C1 keeps its own recipe (learning rate 2e-5).
- Every run archives dev predictions and the raw predictions (probabilities and scores) for v1 test, NegPairs-Fresh and NegPairs-Perm, made with the selected checkpoint; runs do not compute any test metric. Test metrics are computed only after the gate file of section 8.3 has been written.
- Independent runs may execute concurrently in separate worker processes on one GPU; each run is seeded and self-contained, so this does not change any computation.

No other hyperparameter is tuned.

## 7. Estimands

### 7.1 Scope sensitivity on standard minimal pairs

As in v2.0, section 7.1, on NegPairs-Fresh (primary), on v1 dev (gate) and on v1 test (descriptive): `flip_O_N`, `flip_O_S` and `flip_O_P` (shares of scenarios in percent whose predicted option differs from that for O), and `SS = flip_O_N − flip_O_S` in percentage points, overall, for the seen families, for the unseen families and per family; `flip_O_S` on S-same scenarios (the S cue shares a spaCy sentence with a decisive word; identifiers recorded per set).

### 7.2 Permutation estimands (NegPairs-Perm)

`flip_O_N`, `flip_O_S`, `flip_O_P`, `SS_perm = flip_O_N − flip_O_S`; joint (N, S) accuracy, the share of scenarios in which both N and S are answered correctly; `acc_X` per arm. Each overall, for the seen families (syntactic, quantifier), for the unseen families (affixal, lexical), per family, and by clause order (decisive clause first or second).

Order-balanced estimand: `SS_perm_bal = (SS_perm_first + SS_perm_second) / 2`, where `SS_perm_first` and `SS_perm_second` are computed on the scenarios whose decisive clause comes first and second. A rule that reacts only to the clause in which a cue sits (for example "a cue in the first clause changes the decision") gives `SS_perm_first` near +100 and `SS_perm_second` near −100, so its `SS_perm_bal` is near 0 whatever the shares of the two orders; the pooled `SS_perm` of such a rule would instead equal 100(2p − 1) for a share p of decisive-first scenarios. R8 therefore uses `SS_perm_bal`.

### 7.3 Same-cue analysis (descriptive)

As in v2.0, section 7.2, on NegPairs-Fresh and v1 test.

### 7.4 NevIR

Paired accuracy (section 4.8) and per-query accuracy, on the whole test split, on the nested subset and on its complement, for M0 with every encoder, for M0-native, and for the trained modules A1, A3 and M1 of E0 and E3 (every seed; document as context, query as option). Trained modules of the contextual encoders are not applied to NevIR.

### 7.5 Probing (descriptive)

Question: do the frozen token states encode on which clause a negation edit sits? Data: the N and S arms of NegPairs-Perm. Words: every word of the two clauses except the cue (the inserted "not" or the negated form) and, for substitution patterns, the unedited occurrence of W; the connector, punctuation and the question are excluded. Label: 1 if the word lies in the clause that carries the edit (the decisive clause in N, the other clause in S), else 0; every included word therefore occurs once in N and once in S with opposite labels. Features: the encoder's token states as used by the modules, averaged over the subword tokens of a word. Probe: L2-regularized logistic regression (scikit-learn, lbfgs solver, at most 1,000 iterations), inverse regularization strength chosen from {0.01, 0.1, 1, 10} by inner 5-fold cross-validation grouped by scenario. Evaluation: (i) seen families, outer 5-fold cross-validation grouped by scenario (fold assignment seed 20261031); (ii) transfer, trained on all seen-family scenarios and tested on the unseen-family scenarios. Metric: accuracy; chance is 50% by construction. For E0 and E3 the accuracy is exactly 50%, because each word has the same features under both labels; this is reported as a check of the pipeline.

Control (neutral twins). Accuracy above 50% may reflect proximity to an inserted word rather than negation. For the syntactic scenarios, a neutral twin of N and of S is built deterministically by inserting "also" instead of "not" at the same site; the same probe protocol is applied to the twins with the label "word in the clause that carries the inserted word". Selectivity is the probe accuracy on the negation arms minus that on the neutral twins (syntactic family; per encoder).

Gate comparison. For every A5 module, the area under the ROC curve of its gate value `p_i` for the same words and labels is reported (does the learned gate, trained only through the decision loss, recover what a probe finds?).

### 7.6 Secondary accuracy view (descriptive)

As in v2.0, section 7.3, on NegPairs-Fresh: accuracy per arm, `acc_OP`, negation gap, scope gap, conditional accuracies, `pair_ON`, consistency, error-confidence profile and option-prior strata (C2 majority vote over all C2 runs of all encoders on the O items; ties counted as congruent).

### 7.7 Cross-family sensitivity (descriptive)

The tables of section 7.1 recomputed on the v1 test scenarios that pass the Qwen audit (section 4.4).

## 8. Gate and prespecified rules

### 8.1 Thresholds, conditions and minimum sizes (to be accepted by the author)

| Name | Value | Used by |
|---|---|---|
| `gate_SS_seen_dev` | 20 | G |
| `R2_min_gain_unseen_SS` | 15 | R2 |
| `R3_min_gain_flipS` | 10 | R3 |
| `R8_min_SS_perm_bal_seen` | 10 | R8 |
| `min_positive_seeds` | 9 of 10 | R2, R3, R8 |
| `family_alpha` | 0.05, divided by the number k of encoders on which the rule is evaluated | R2, R3, R8 |
| `min_scenarios_R2` | 80 unseen-family NegPairs-Fresh scenarios | R2 |
| `min_scenarios_R3` | 100 S-same NegPairs-Fresh scenarios | R3 |
| `min_scenarios_R8` | 40 seen-family NegPairs-Perm scenarios in each clause order (after the tokenizer restriction of section 4.7) | R8 |
| `exact_tolerance` | 1e-9 | R5, R6, R7 |

Values are percentage points except counts, levels and the tolerance. The thresholds of G, R2 and R3 are those of v2.0. A statistical rule is met for an encoder if all of the following hold: (a) the seed mean of its quantity reaches the threshold; (b) the quantity is positive in at least nine of the ten seeds; (c) the two-sided seed-level t interval at confidence level `1 − 0.05/k` (nine degrees of freedom; for R2 and R3 the interval of the paired difference) lies above zero; (d) the two-sided scenario-cluster bootstrap percentile interval at the same level lies above zero. Condition (d) can block a rule and never rescue it. `k` is the number of encoders on which the rule is evaluated; it is fixed by G before any test metric is computed. A rule whose subset is smaller than its minimum is reported as not evaluable, with its quantities given descriptively.

### 8.2 Exact rules (checks of the implementation against Propositions A and B)

A violation of R5, R6 or R7 indicates a defect in code or data; the study stops until it is resolved.

R5 Proposition B (insertion pairs). For E0 and E3, for A1, A3 and M1 in every seed and for M0: on every insertion pair of v1 test, NegPairs-Fresh and NegPairs-Perm, no option score under the longer context is lower than under the O context by more than 1e-9.

R6 Proposition A (permutation pairs). For E0 and E3, for A1, A2, A3, A5, M1 and C0 in every seed and for M0: on every NegPairs-Perm scenario, every option receives the same score under N and under S within 1e-9; and in every scenario whose option-score margin exceeds 1e-9 in both N and S, the decisions agree. Hence, away from near-ties, `SS_perm = 0` and the joint (N, S) accuracy is 0. The same holds for A6 on the Perm scenarios whose marked multisets agree in N and S (Corollary A.3; their number is reported). Near-ties (margin at most 1e-9 in N or S) are counted and reported.

R7 Corollary B.1 (NevIR). For E0 and E3, for M0 and for A1, A3 and M1 in every seed, with the document as context and the query as option: on every nested test instance, for both queries, the score of the contained document does not exceed that of the containing document by more than 1e-9, and the paired accuracy on the nested subset (margin rule of section 4.8) is 0.

An exact rule whose set of items is empty (no insertion pair, no Perm scenario with equal marked multisets for A6, no nested NevIR instance) is reported as not evaluable.

### 8.3 Statistical rules

G Responsiveness gate (per encoder). A1 seed-mean SS on the dev split of NegPairs v1 (seen families; 105 scenarios) at least 20. G is computed from dev predictions after all runs and written to `gate.json` before any test metric is computed; it fixes the encoders on which R2, R3 and R8 are evaluated and thereby `k`. For an encoder that does not pass, R2, R3 and R8 are not evaluated and its results are reported descriptively. Dev also serves checkpoint selection (by accuracy), so dev SS may be somewhat optimistic; G is a precondition, not a test of a hypothesis.

R2 Supplied scope (each encoder that passes G). A4 minus A1 SS on the unseen-family scenarios of NegPairs-Fresh at least 15; conditions (b) to (d) of section 8.1.

R3 Scope precision (each encoder that passes G). A6 minus A4 `flip_O_S` on the S-same scenarios of NegPairs-Fresh at least 10; conditions (b) to (d).

R8 Positional scope in contextual states (each contextual encoder, E1, E2, E4, E5 and E6, that passes G). A1 seed-mean `SS_perm_bal` on the seen-family scenarios of NegPairs-Perm at least 10; conditions (b) to (d), with the bootstrap resampling scenarios within each clause order. Rationale: N and S contain the same tokens and the cue families were seen in training, so a module over static states is at exactly zero (R6), and any decision difference has to come from the encoder's contextualization of positions; the order-balanced estimand removes rules that react only to the position of the clause.

### 8.4 Descriptive blocks (no rule)

R1 (descriptive since v3.0): E0 A1 unseen-family SS on NegPairs-Fresh, shown against the v2.0 threshold of 5. It is not a rule because, on standard pairs, Proposition A allows a static module to separate N from S through token identity (Remark A.4); the limit is tested exactly by R6. Bridge: A1 SS (seen, unseen) on NegPairs-Fresh, and `SS_perm` and `SS_perm_bal` per family on NegPairs-Perm, for every encoder. R4: A5 against A1 and A4. D1 against A1 and A4. M0 and M1 against A1 (does the affinity network matter?). A4 and D1 on NegPairs-Perm under E0 and E3 (Proposition C in practice: what supplied scope achieves where unmasked rules are exactly at zero). Unseen-family NegPairs-Perm. Joint (N, S) accuracy for the contextual encoders. Same-cue analysis. All conditions on v1 test (ten seeds; previously read). NevIR, including M0-native beside the published ColBERTv2 value. Probing, neutral-twin selectivity and A5 gate AUC. Qwen audit of v1 and the sensitivity tables of section 7.7. Controls C0, C1 and C2. Secondary accuracy view. Resolver diagnostics on NegPairs-Perm.

### 8.5 Outcome map

| Outcome | Reading |
|---|---|
| R5, R6 or R7 not met | A defect in code or data; the study stops until it is resolved. |
| R5, R6 or R7 not evaluable | No item of the required kind exists; reported as such. |
| G not passed (encoder) | The module did not learn to respond to trained negation; no scope claim for that encoder. |
| R2 met | Supplying the resolver's scope to the signed rule produces scope-sensitive decision changes for unseen cue families on new data. |
| R2 not met | Not with this resolver; D1 shows whether exact scope would. |
| R3 met | Scope precision matters: a sentence-wide polarity mask changes decisions on irrelevant negations that the dependency scope leaves alone. |
| R3 not met | No evidence in these data that scope precision matters beyond sentence-level polarity. |
| R8 met (encoder) | The frozen states of that encoder carry information about where a trained negation cue sits, in a form that a max-aggregated module trained on standard minimal pairs uses to separate permutation pairs. |
| R8 not met (encoder) | No evidence that the module uses such information for that encoder; the probe shows whether the information is present in the states. |
| R2, R3 or R8 not evaluable | The subset is below its minimum size; the quantities are reported descriptively and no claim is made. |

## 9. Uncertainty

Seed-level t intervals (ten seeds, nine degrees of freedom; paired across conditions within a seed for contrasts) at the levels of section 8.1 for the rules and at 95% elsewhere. Scenario-cluster percentile bootstrap (10,000 replicates, generator seed 20261003; all arms and seeds of a scenario resampled together; within clause order for `SS_perm_bal`). NevIR: instance-level percentile bootstrap of paired accuracy (10,000 replicates, generator seed 20261003). Exact two-sided McNemar tests on paired predictions within each seed for the secondary accuracy contrasts. Probing: mean and standard deviation over outer folds. Intervals that include zero are reported as inconclusive, never as equivalence. Family-wise error control is applied only within each statistical rule across encoders (section 8.1); no control across rules, blocks or data sets is claimed. Every rule outcome is reported, met or not.

## 10. Exact-test protocols

Common: scores are recomputed in float64 on the CPU from the archived module states, one item at a time (no padding), with the E0 and E3 vectors regenerated in float64 and no autocast or TF32; tolerance 1e-9. Predictions made on the GPU in bfloat16 are compared as well; every disagreement with the float64 decision is listed, and none counts as a violation (as in v2.0). All reported metrics of E0 and E3 on NegPairs-Perm and NevIR come from the float64 scores.

R5. Insertion pairs: within a scenario of v1 test, NegPairs-Fresh or NegPairs-Perm, the pair (O, X) with X in {N, S} such that `Bag(O) ⊊ Bag(X)` for `<context> <question>` under the E0 tokenizer (for example every syntactic Perm scenario, where N and S are O with "not" inserted). For every pair and option the score under X must not be lower than under O by more than 1e-9. Reported per set: pairs, scores, violations, largest decrease.

R6. Every NegPairs-Perm scenario; for every listed module and option, `|S(o; N) − S(o; S)| ≤ 1e-9`; equal decisions where the margins exceed 1e-9 in both arms; near-ties counted. For A6 the resolver's masks are computed on N and S, and the check runs on the scenarios with equal marked multisets.

R7. Every nested NevIR test instance (E0 tokenizer, lower-cased; no truncation). For every listed scorer and both queries, `s(q, d_contained) ≤ s(q, d_containing) + 1e-9`; paired accuracy on the nested subset with the margin rule of section 4.8.

## 11. Pilot (train and dev only) and stop conditions

Phase P1 of the experiment notebook runs before any test file is parsed (test files are only checked against their SHA-256 values and the build reports):

1. Dataset audit of v1 train and dev, of the file manifests of all sets, and of the build-time Perm assertion recorded in the build report.
2. Encoder checks: loading of every encoder (for E6 including the 128-dimensional projection), unknown-token rates on train and dev, missing or unexpected weights, E3 out-of-vocabulary rate.
3. Exact checks on dev with random initializations (five per condition): for E0 and E3, identical scores (within 1e-9, float64) for every dev context and a random permutation of its tokens, for A1, A2, A3, A5, M1 and C0 (Proposition A); no decrease on dev insertion pairs for A1, A3 and M1 (Proposition B); violations by A2 and A4 on insertion pairs are reported (expected, as in v2.0).
4. C1 check: two epochs on dev, as in v2.0.
5. Learning-rate grid per encoder (section 6); the result is written to `lr_grid.json` before any test file is parsed.

Stop conditions: S1 dataset integrity (a hash mismatch, or a build report without the Perm assertion; after parsing, a scenario in two sets or a Perm scenario with unequal bags under the E0 tokenizer); S2 tokenizer or weight-loading failure; S3 C1 check below 75% dev `acc_OP`; S4 any violation of Proposition A or B in step 3. The pilot fixes only the learning rate per encoder.

## 12. Compute, provenance, release

Build notebook `NegPairs_v3_build_colab.ipynb`: one GPU runtime with bfloat16 support and at least 22 GiB (L4 or A100); Anthropic API calls with the key in Colab Secrets; Qwen verification inline; acceptance, audit and files. Experiment notebook `NegPairs_v3_main_colab.ipynb`: pilot (63 A1 runs and the C1 check), main grid of 7 encoders by 8 trained conditions (A1 to A6, D1, M1) by 10 seeds = 560 runs, C0 and C2 (140 runs), C1 (10 runs), M0 and M0-native (scoring only), the gate file, exact rules on the CPU, NevIR scoring, probing, Qwen audit of v1, analysis. Both notebooks are resume-safe and run on Google Colab; an A100 is planned for the experiment; hardware is not part of any rule, and the GPU type is recorded. Package `cacsa2` 0.3.0.

The run directory carries `run_environment.json`, `CACSA_manifest_v3_0.md` with `manifest_lock.json` (written before any training), `data_audit.json`, `lr_grid.json`, `pilot_exact_checks.json`, `gate.json`, `encoder_meta.json`, resolver reports and `arm_diagnostics` for NegPairs-Fresh and NegPairs-Perm, `S_same_ids` per set, `perm_token_bag_check.json`, `per_run.json`, `seed_means.json`, flip and accuracy tables, `rules.json` (primary), `exact_R5.json`, `exact_R6.json`, `exact_R7.json`, `nevir_results.json`, `probing.json`, `qwen_audit_v1.json`, `rules_accuracy_secondary.json`, `mcnemar_accuracy_secondary.json` and `FILE_MANIFEST_SHA256.csv`. Module states are kept on Drive and not zipped.

Release: code (`cacsa2`, MIT licence), notebooks, NegPairs v1, NegPairs-Fresh and NegPairs-Perm with their construction records, raw predictions and an independent recomputation of every table, in a public repository with a Zenodo DOI; the dataset licence is decided by the author before release.

## 13. Paper (Computational Linguistics)

Framing: MaxSim-family scorers (late interaction, greedy token matching, CACSA) over frozen token states; CACSA v2 is one instance. The contribution is presented as exact item-level predictions for this family together with two diagnostics that make them testable (permutation minimal pairs; the nested subset of NevIR), not as new mathematics: Propositions A and B are elementary consequences of symmetric and monotone aggregation (related statements on permutation-invariant set functions: Zaheer et al., 2017), and Proposition C is a possibility statement. Planned structure: Introduction; Background (late interaction and greedy matching; negation and its scope; minimal pairs and contrast sets; prespecified analyses); Propositions (statements and corollaries in the main text, proofs in an appendix); Diagnostics (NegPairs v1, NegPairs-Fresh, NegPairs-Perm; NevIR); Design (prespecified rules, history of the manifests); Results; Analysis (probing with neutral twins, same-cue, clause order, error analysis); Discussion; Limitations; Conclusion; Code and Data Availability; Generative AI Declaration; appendices (proofs, CondaQA as a bounded negative result, construction details, all tables). Abstract of 150 to 250 words; author-year citations; length typically 30 to 40 journal pages. An internal critical review and a cover letter are prepared before submission. The paper reports the v2.0 outcomes and the v3.0 outcomes, states that v3.0 was written after the v2.0 test results were known, and explains why the confirmatory rules use NegPairs-Fresh and why R1 became descriptive.

Disclosure: NegPairs v1, NegPairs-Fresh and NegPairs-Perm were generated by `claude-sonnet-5`, verified by `claude-haiku-4-5-20251001` (and, for the new sets, by `Qwen/Qwen2.5-7B-Instruct`) and audited by `claude-opus-5`; this is described in the data section and listed in the Generative AI Declaration together with every other use of generative AI tools (analysis-code support and any writing support); the author takes responsibility for the content.

Writing rules: the author's fifteen in-paper rules apply. Every number is traced to a file in a run directory; every reference is verified before submission.

## 14. Open items

1. Decide the NegPairs licence (all parts).
2. Create the public repository and the Zenodo DOI; optionally deposit the frozen manifest with a public timestamp before the build (freeze procedure, step 3).
3. Confirm which other generative AI tools, if any, were used (for the declaration).
4. Confirm the ORCID.
5. Confirm the JCR quartile of Computational Linguistics through library access.
6. Optional, outside every rule: a human check, if the author decides on one later. Sample fixed now: 50 NegPairs-Perm and 50 NegPairs-Fresh scenarios drawn with seed 20261008 (stratified by family), all four arms judged for the better option and for naturalness, by the author and, if available, a second annotator; agreement with the expected answers (and between annotators) is reported descriptively.
7. Optional, outside every rule: C1 on CondaQA with the publisher's recipe (learning rate 1e-5, 10 epochs).

## 15. Change log

v3.0 DRAFT 1 (2026-10-05): written after the v2.0 confirmatory run and revised after an internal critical review on the same day, before the author's review.

v3.0 FROZEN (2026-10-05): accepted by the author. Added to DRAFT 1 while the build code (package `cacsa2` 0.3.0) was written and tested offline, without changing any proposition, data design, rule or threshold: distinct seeds (the Perm schedule seed moves from 20261006 to 20261020 and the clause-order seed from 20261007 to 20261022, because 20261006 is the option-order seed of NegPairs-Fresh under the v1 convention and 20261007 would have been the option-order seed of NegPairs-Perm; the option-order, audit and probing-fold seeds are now stated); filter 8 removes marks copied into the decisive words; the operational safety stop of section 4.6; the GPU memory requirement stated as reported by the driver (an L4 reports about 22.5 GiB).

Added: Propositions A to C, which contain Propositions 1 and 2 of v2.0 as special cases (section 3); NegPairs-Fresh and NegPairs-Perm (section 4); cross-family verification with Qwen2.5-7B-Instruct; encoders E3 to E6; M1, M0 and M0-native; NevIR; ten seeds; a learning rate per encoder; exact rules R6 and R7; rule R8 with the order-balanced estimand; minimum subset sizes; the cluster-bootstrap condition (d) and the division of the error level by the number of encoders (section 8.1); probing with neutral twins.

Unchanged: NegPairs v1 and its role in training and selection, the shared module, A1 to A6 and D1, the scope resolver, the controls, the training protocol and the thresholds of G, R2 and R3.

Changed: (a) G is computed on v1 dev instead of the test split, so that the evaluated encoders are fixed before any test metric is computed; (b) R2 and R3 are evaluated on NegPairs-Fresh instead of the v1 test split, which has been read; (c) the seed condition of R2 and R3 changes from "positive in all five seeds" to "positive in at least nine of ten seeds" (with ten seeds a single unstable seed would otherwise decide a rule on its own), and conditions (c) and (d) of section 8.1 are added; (d) R1 becomes descriptive, because Proposition A shows that on standard pairs a static module may separate N from S through token identity, so a bound on E0's unseen-family SS does not follow from the theory, whereas R6 tests the limit exactly; (e) R5 is extended to E3, M0 and M1 and to the new sets.

---

## Appendix A. Generator prompt of NegPairs-Perm (frozen with this manifest)

`PERM_GEN_PROMPT` is the exact text between the fences below (UTF-8, no final newline). Placeholders: `{domain}`, `{situation}` and `{banned}` as in v1; `{first}` is `DECISIVE` or `OTHER`; `{family}` is the family name; `{pattern_desc}` and `{negated_form_desc}` are the strings of `PERM_PATTERNS` below. Doubled braces are literal braces after formatting.

```
You are writing ONE test item for a study of how reading models handle the scope of negation.

Write a short, realistic everyday scenario in the domain "{domain}" about {situation}.
Do NOT use any of these topics: {banned}.

The context is ONE sentence (at most 25 words) made of two clauses joined by a connector such as "and", ", and", "while" or "but":
- the DECISIVE clause states a fact that makes exactly one of two options the better choice;
- the OTHER clause states a detail that does not matter for the choice.
Put the {first} clause first.

Negation pattern for this item ({family}): {pattern_desc}
Mark the two edit sites, one in each clause, exactly as described. The first word of the sentence must not be an edit site. Write a marked word in lower case and in exactly the same form at both sites, with no punctuation attached to it.
Applying the pattern at the site in the DECISIVE clause must make the OTHER option the better choice.
Applying the pattern at the site in the OTHER clause must leave the better choice unchanged.
Both edited sentences must be natural English.

Return these fields:
- question: one short question asking what someone should do. It must not reveal the answer and must contain no negation.
- options: exactly two short action phrases (2 to 5 words each). Both must be sensible actions in general; neither may contain a negation word.
- answer: "A" if the first option is the better choice for the unmarked context, "B" if the second.
- context_marked: the sentence with the two edit sites marked.
- connector: the exact word or words joining the two clauses, as written in context_marked.
- decisive: the exact words of the decisive clause that decide the answer, as written in context_marked (without marks).
- negated_form: {negated_form_desc}
- context_P: a paraphrase of the unmarked context: same meaning and same answer, clearly different wording.

The unmarked context, context_P, the question and the options must contain NO negation of any kind (no not, n't, no, never, none, nothing, nobody, without, lacks, fails, un-/in-/dis-/non- words meaning "not", -less words, and so on).

If you cannot satisfy every constraint naturally, return {{"usable": false, "reason": "<= 8 words"}}.

Output strict JSON only:
{{"usable": true, "question": "...", "options": ["...", "..."], "answer": "A", "context_marked": "...", "connector": "...", "decisive": "...", "negated_form": "...", "context_P": "..."}}
```

`PERM_PATTERNS` (the values are the exact strings; JSON escaping only):

```json
{
  "syntactic": {
    "pattern_desc": "the word \"not\" is inserted directly after an auxiliary or copula verb (is, are, was, were, can, will, should, has, have). Each clause must contain such a verb; mark the insertion point after it with [[^]]. Format example: \"the X is [[^]] ready, and the Y is [[^]] warm\".",
    "negated_form_desc": "the word \"not\"."
  },
  "quantifier": {
    "pattern_desc": "the determiner \"a\", \"an\" or \"some\" is replaced by \"no\". Both clauses must use the SAME determiner; mark it in each clause with double brackets. Format example: \"there is [[a]] X at the door, and there is [[a]] Y in the hall\".",
    "negated_form_desc": "the word \"no\"."
  },
  "affixal": {
    "pattern_desc": "the SAME adjective or participle W occurs in both clauses and is replaced by its negated form with a prefix un-, in-, im-, il-, ir-, dis- or non- written without a hyphen (for example available becomes unavailable, accessible becomes inaccessible, connected becomes disconnected). W must not follow \"a\" or \"an\". Mark W in each clause with double brackets. Format example: \"the X is [[available]] today, and the Y is [[available]] too\".",
    "negated_form_desc": "the prefixed word that replaces W (for example \"unavailable\")."
  },
  "lexical": {
    "pattern_desc": "the SAME word occurs in both clauses and is replaced by its lexical negation: has becomes lacks, have becomes lack, had becomes lacked, with becomes without, present becomes absent. Mark the word in each clause with double brackets. Format example: \"the X [[has]] a key, and the Y [[has]] a lamp\".",
    "negated_form_desc": "the replacing word (lacks, lack, lacked, without or absent)."
  }
}
```

## Appendix B. Cross-family verifier prompt (frozen with this manifest)

`QWEN_VERIFY_PROMPT` is the exact text between the fences below (UTF-8, no final newline). The letters "A", "B" and "U" must each be a single token of the Qwen tokenizer; the build notebook checks this before any call and stops otherwise.

```
Read the context and decide which of the two options is the better thing to do. Use only the context.

CONTEXT: {context}
QUESTION: {question}
A) {option_a}
B) {option_b}

Answer with one letter: A or B for the option that the context gives a reason to prefer, or U if the context gives no reason to prefer one option over the other or supports both equally.
```

## Appendix C. SHA-256 of the frozen prompt strings

Computed on the UTF-8 encoding of each string as defined above (module `cacsa2.perm_prompts`, function `prompt_hashes`); the build notebook recomputes every value before its first call and stops if one differs.

| String | SHA-256 |
|---|---|
| `PERM_GEN_PROMPT` | `12ecbd2475c43497d160e8e2c4b94af2979e8ca7c13c2f8d4793287b531dfc74` |
| `QWEN_VERIFY_PROMPT` | `818f46ad91106d988ccc5349bfb891b98bc8a1400ff6ade153f022b62f60661d` |
| `PERM_PATTERNS.syntactic.pattern_desc` | `4b72ae115883be933e21c1e4f806a36502b0d1a359f5da57390cc9a9d46be3da` |
| `PERM_PATTERNS.syntactic.negated_form_desc` | `c1ec01050e8b58dd2e96f774d507b47a4d393614b4a24ceb7279541390fc4349` |
| `PERM_PATTERNS.quantifier.pattern_desc` | `c872365872da723ca30429af162aab7133e048bf89b4f35192c8af19fd89b9c9` |
| `PERM_PATTERNS.quantifier.negated_form_desc` | `dbf7116b6b84c88d375fdc981cd6b1cf17f7d4a1586025df4b681fa8ecfae46e` |
| `PERM_PATTERNS.affixal.pattern_desc` | `00c69444f7251a0dae28590a80bd47666c33df52b0dcc9e5e5a167abc5fa9f67` |
| `PERM_PATTERNS.affixal.negated_form_desc` | `677be83bc308a36b997ee08b5c470710930c3a335a036775c50136b17e99e282` |
| `PERM_PATTERNS.lexical.pattern_desc` | `8510dc14de0895cf073179b84b02322c1db5a735657447e968f91d0ea44a3f65` |
| `PERM_PATTERNS.lexical.negated_form_desc` | `a8e17098bea3776fdd914a569ff3a809fa02f642de0f1084467e6c34ce689ea6` |
