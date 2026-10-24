"""cacsa2: analysis code for the CACSA polarity study.

Modules
-------
condaqa     CondaQA loading, eligibility, contrast groups, cue partitions, candidate rendering.
encoders    Frozen token encoders: a context-independent hash encoder and frozen Hugging Face encoders.
scope       Negation-cue lexicon and a deterministic dependency-based scope resolver.
model       The shared pair scorer and the five aggregation rules (A1-A5).
train       Training / selection / prediction archiving for one (encoder, aggregator, seed).
metrics     Estimands M1-M7.
stats       Paired seed intervals, passage-cluster bootstrap, exact McNemar.
provenance  SHA-256 manifests and run-environment records.
prop1       Exact Proposition-1 test in float64.
negpairs_build  NegPairs dataset construction (generation prompts, filters, blind verification, splits).
negpairs    NegPairs loading, CACSA items (scope / sentence-wide / dataset-supplied masks), arm and paired metrics,
            cue/scope/sentence diagnostics, insertion pairs, multiple-choice fine-tuning reference.
np_analysis NegPairs analysis: per-run metrics, seed means, paired seed intervals, scenario-cluster bootstrap, McNemar,
            learnability gate and rules of manifest v2.0.
controls    Controls on frozen states: C0 head-only, C2 option-only (context-blind).
perm_prompts  Frozen prompts of the v3.0 build (Perm generator, Qwen verifier) and their SHA-256 values.
perm_build  NegPairs-Perm schedule, deterministic assembly and filters; schedules and item files of the v3.0 sets.
qwen_verify Cross-family verification with Qwen2.5-7B-Instruct (both option orders, float32 letter logits).
build_v3    Build loop of the v3.0 sets: metered calls, batches, acceptance in schedule order, resume, audit.
np_analysis_v3  Permutation estimands, dev gate, rules R2/R3/R8 with four conditions, descriptive blocks (manifest v3.0).
exact       Exact rules R5, R6, R7 in float64 on the CPU; pilot checks with random initializations.
nevir       NevIR external test: loading, nested subset, paired accuracy with margins, bootstrap.
probing     Probing of the clause of the edit in frozen states; neutral twins; A5 gate AUC.
packs, worker  Encoder packs and concurrent training workers on one GPU.
colbert     ColBERTv2 native scoring (M0-native).
finetune_li  Fine-tuning of released ColBERT checkpoints on NegPairs v1 or NevIR train with their own scoring; SciFact check.
public_systems  Public retrieval systems (late interaction, dense, sparse, cross-encoders, monoT5, pointwise LLM judges, an
            API pairwise judge) scored on NegPairs and NevIR through their own interfaces, with the article's metrics.
"""

__version__ = "0.3.5"
VERSION_TUPLE = tuple(int(x) for x in __version__.split("."))
