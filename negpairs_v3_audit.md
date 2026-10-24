# NegPairs v3.0 build: audit of the delivered data (6 October 2026)

Files audited: `negpairs_v3.zip` (built 5 October 2026, 19:36 UTC, on Google Colab with an NVIDIA L4 GPU; package `cacsa2` 0.3.0, zip SHA-256 `d011ccca…146df`). Every number below was computed from the files in the zip by script; nothing was edited.

## 1. Integrity

- All 9 files match `FILE_MANIFEST_SHA256.csv` (sizes and SHA-256).
- The manifest copy in the zip and `manifest_lock.json` carry SHA-256 `57f834c56455cb7cd6b2c997f3599a86b2b7dbf68213dea8986ef2fd1fb1561c`, identical to the frozen `CACSA_manifest_v3_0.md`; the lock was written at 18:25:15 UTC, before the first API call (the build report is from 19:36 UTC).
- Package zip SHA-256 in the build report equals the delivered `cacsa2_project.zip` (0.3.0).
- All 12 prompt hashes matched the frozen manifest (checked by the notebook before its first call); `wordfreq` 3.1.1.
- No scenario identifier occurs in two sets or in NegPairs v1 (0 overlaps). The near-duplicate check (v2.0 index seeded with the 1,088 v1 scenarios, then every accepted new scenario) finds no near-duplicate on recomputation.
- Provenance gap: `qwen_revision` is `null` in the build report, because transformers 5.18 no longer sets `config._commit_hash`. The Hugging Face repository `Qwen/Qwen2.5-7B-Instruct` has a single commit on `main` since 12 January 2025 (`a09a35458c702b33eeacc393d103063234e8bc28`, `lastModified` 2025-01-12T02:10:10Z, checked on 6 October 2026), so the build downloaded that revision. The package now records the snapshot commit from the cache (`qwen_verify.QwenVerifier._cached_revision`), and the experiment notebook records it when it loads Qwen for the v1 audit.

## 2. NegPairs-Perm (`perm.jsonl`, 274 scenarios, 1,096 rows)

Phase stopped at its budget cap (12.07 USD metered) after 1,434 candidates: accepted 274, Haiku verifier 814, Qwen 205, filter 129, near-duplicate 10, generator 1, stratum cap 1.

| Family | Decisive clause first | Decisive clause second | Qwen pass (of Haiku survivors) |
|---|---|---|---|
| syntactic (seen) | 38/38 | 38/38 | 76/132 |
| quantifier (seen) | 38/38 | 38/38 | 76/124 |
| affixal (unseen) | 15/38 | 32/38 | 47/77 |
| lexical (unseen) | 38/38 | 37/38 | 76/147 |

Rule R8 needs at least 40 seen-family scenarios in each clause order: 76 and 76. Every structural check of manifest section 4.3 passes on recomputation: `Bag(N) = Bag(S)` under the E0 tokenizer and the case-preserving tokenizer in all 274 scenarios, N differs from S, word edit distance O to N and O to S exactly 1, O to P at least 2, no core or `un-` negation in O, P, the question or the options, options with different bags, every arm at most 30 words (O: 9 to 21 words, median 15), the cue span of N inside the decisive clause and that of S inside the other clause, the unedited W span correct, decisive words inside the decisive clause of O, labels O = P = S = 1 − N. Label balance O: 133 / 141. All 14 domains and 8 situations occur.

Cues: syntactic "not" (76), quantifier "no" (76; sites "a" 67, "some" 9), lexical "lacks" (75; every lexical scenario uses the pair has → lacks), affixal 17 negated forms over 17 base words (unavailable 13, inaccessible 8, unstable 7, insufficient 3, incapable 3, unconfirmed 2, and 11 forms occurring once: inadequate, unreliable, impassable, disconnected, unmarked, uncertain, incompatible, uncooked, unsuitable, inedible, insecure; corrected on 7 October 2026 from an earlier list of eight forms). Connectors: "while" 210, "and" 21, "but" 20, ", and" 18, ", while" 3, ", but" 2. Auxiliary before "not": is 54, has 11, can 10, are 1.

Frozen resolver (spaCy en_core_web_sm 3.8.0; descriptive): the cue is detected in 272 of 274 N and S arms (the two misses are affixal); the resolved scope covers the decisive words in 271 of 274 N arms (syntactic 75/76, quantifier 76/76, affixal 46/47, lexical 74/75) and, in S arms, in 2/76 syntactic, 3/76 quantifier and 6/75 lexical scenarios; cues are falsely detected in 2.6% of O and 6.6% of P arms. For affixal S arms the v2.0 diagnostic reports 46/47 "decisive in scope" because it finds the decisive base word inside the negated form of the other clause (`decisive_token_spans` marks a cue token that contains a decisive word); this is the known limitation that manifest section 4.3 addresses: D1 masks and the Perm diagnostics of the experiment notebook are built from the stored clause spans, not from this word search. Every Perm scenario is one sentence, so all 274 are S-same.

Format artefact (not corrected, because the verified strings are frozen): 158 of 274 contexts start with a lower-case letter and 107 end without punctuation (O, N and S alike; P is generator-written and affected in 13 cases). The cause is the format examples in the frozen generator prompt (`"the X is [[^]] ready, and the Y is [[^]] warm"`), which the generator imitated; the pattern is strongest in the quantifier family (70 lower-case starts, 59 without final punctuation). N and S are affected identically, so N versus S comparisons are unaffected; comparisons with NegPairs v1 and NegPairs-Fresh, whose contexts are capitalized sentences, are descriptive. To be stated in the data section.

Opus audit (seed 20261027; 27 scenarios: syntactic 9, quantifier 1, affixal 6, lexical 11; all four arms judged, no early stop): answers agree in 105 of 108 arm judgements (O 25/27 with 2 "unclear", P 26/27 with 1 "unclear", N 27/27, S 27/27); judged natural O 26/27, P 21/27, N 20/27, S 16/27; agreement on answer and naturalness in all four arms 9/27. The disagreements are almost all naturalness judgements of the mechanically negated clauses (for example "the bread is inedible but a day old", "the pharmacy lacks a long line today"), which is the expected cost of the permutation design (manifest section 2).

## 3. NegPairs-Fresh (`fresh.jsonl`, 240 scenarios, 960 rows)

Phase reached every cap (60 per family) after 1,348 candidates at a cumulative 21.39 USD before the audit: accepted 240, Haiku verifier 494, filter 408, near-duplicate 98, Qwen 108. Qwen pass of Haiku survivors: syntactic 60/87, quantifier 60/97, affixal 60/73, lexical 60/91. `structural_filter_v2` passes on every scenario on recomputation; labels O = P = S = 1 − N; label balance O: 117 / 123; every context capitalized with final punctuation. Word edits O to N: 1 word 94, 2 words 38, 3 words 48, 4 words 32, 5 words 28; O to S median 6, maximum 10. N cues: syntactic "is not" 18, "cannot" 14, "does not" 11; quantifier "no" 31, "none of" 23; affixal 25 distinct forms (unavailable 13, insufficient 9, unsafe 5); lexical 19 distinct forms (lacks 22, "is out of" 10, "is free of" 6). S cues are dominated by "never" (41 of 60 syntactic) and "nobody" (49 of 60 quantifier), as in v1 (manifest v2.0, section 4.5 b). Resolver: declared cue detected in 100% of seen-family and 80% of unseen-family N arms, and in 100% and 70% of S arms (lexicon coverage of unseen cues, reported, not corrected); scope covers a decisive word in 100% of seen-family and 78.8% of unseen-family N arms, and in 9.2% and 1.7% of S arms. S-same scenarios: 231 (R3 needs at least 100). Unseen-family scenarios: 120 (R2 needs at least 80). Seen-family: 120.

Opus audit (seed 20261012; 24 scenarios: affixal 10, quantifier 7, syntactic 5, lexical 2): answers agree in 96 of 96 arm judgements; judged natural O 24/24, P 24/24, N 21/24, S 21/24; agreement on all four arms 18/24.

## 4. Cross-family verification

Qwen2.5-7B-Instruct, bfloat16 weights, float32 letter logits, both option orders. Accepted scenarios: all 274 + 240 pass on recomputation from the archived logits (Haiku verdicts also present and consistent). Letter-logit margins (top minus second): median about 19.9 in both sets; smallest 0.001 (Perm) and 0.051 (Fresh); margins below 0.5 in 5 of 2,192 Perm and 4 of 1,920 Fresh presentations; no exact tie.

## 5. Cost

Metered 21.39 USD in total (9,269 calls): Sonnet 5 18.36 USD (2,788 calls), Haiku 4.5 2.59 USD (6,277 calls), Opus 5 0.45 USD (204 calls). Perm phase 12.07 USD, Fresh phase 8.87 USD, audit 0.45 USD. Hard cap 24.00 USD not reached.

## 6. Consequences for the experiment notebook

1. D1 masks and resolver diagnostics on NegPairs-Perm must use the `perm.spans` field (clause, cue, unedited W), as manifest section 4.3 requires; the v2.0 `oracle_mask` would mark the unedited W in the other clause.
2. Record the Qwen snapshot revision when the model is loaded for the v1 audit and compare it with `a09a354…`.
3. Report the lower-case and punctuation artefact of NegPairs-Perm in the data section; keep the strings as verified.
4. Lexical Perm scenarios carry a single cue ("lacks"); the unseen-family Perm results are descriptive in any case.
