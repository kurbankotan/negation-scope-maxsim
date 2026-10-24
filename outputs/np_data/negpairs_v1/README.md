# NegPairs (folder negpairs_v1), construction v2

Controlled negation minimal pairs in the CACSA format, built 2026-10-03 with cacsa2 0.2.4.
Each scenario: context, question, two options; arms O (affirmative), P (paraphrase), N (one negation cue on the decisive
words; answer flips), S (one negation cue on an irrelevant detail; answer unchanged). Families syntactic and quantifier
are seen (train/dev/test, 60/15/25 by scenario); affixal and lexical are unseen (test only).

Generation: claude-sonnet-5 with an unchanged prompt (first run at effort medium and a 1,200-token limit; this run at effort
low and 4000 tokens). Every kept scenario passed the v2 filters and was verified arm by arm, blind,
by claude-haiku-4-5-20251001 (temperature 0) with the v2 verifier prompt, including the scenarios generated in the first run (their
first-run verdicts are archived, not reused). A seeded 10% sample was re-verified by claude-opus-5
(70/109 checked scenarios agreed on all arms). Not human-verified.
Kept 1088 scenarios, 4352 items; metered cost $31.98 in total. Prompts are in
cacsa2/negpairs_build.py (SHA-256 in build_report.json). Files: train/dev/test.jsonl (one row per arm), scenarios.jsonl,
build_state.json (every candidate of both runs and the reason it was discarded), build_state_v1_backup.json (the first
run's record as it was), opus_audit_v2.json, build_report.json, FILE_MANIFEST_SHA256.csv.
