# NegPairs: run outputs and provenance records

Outputs behind every number of the article *Negation Scope in MaxSim-Family Scorers: Exact Limits, Permutation Minimal
Pairs, and the Behavior of Released Systems* (Kurban Kotan, Manisa Celal Bayar University).

`paper/registry.py` reads every file at the paths below from this folder, the default data root (to use a copy
elsewhere, set `CACSA_DATA_ROOT` to it). The data and run folders (`np_data`, `np_data_v3`, `main_v3`, `public_v3`,
`finetune_v3`) carry a `FILE_MANIFEST_SHA256.csv` with the SHA-256 value of every file of the run folder; every listed
file that is in this release matches its value. Two kinds of listed files are not released as such: the trained module
states of the main run (`module_state.pt`), which no reported number needs, and the full prediction files of the main
run, which are released in compact form in `main_v3/pred/predictions.zip` (sample identifier, decision and option
probabilities rounded to four decimals, as written by the run notebook) and therefore do not match their listed values.
The follow-up manifest also lists the files of the main run folder. `np_data/negpairs_v1/` and
`np_data_v3/negpairs_v3/` are byte-identical copies of `data/negpairs_v1/` and `data/negpairs_v3/`, kept at the paths of
the run layout (Git stores identical files once).

| Path | Content | Run |
|---|---|---|
| `np_data/negpairs_v1/` | NegPairs v1 with its build record | construction v2, CPU, API |
| `np_data_v3/negpairs_v3/` | NegPairs-Fresh and NegPairs-Perm with the build record and the frozen manifest v3.0 | 5 October 2026, Colab NVIDIA L4 |
| `main_v3/np_v3_main/` | prespecified study under manifest v3.0: pilot, gate, 718 run summaries (`runs/`), seed means, rules, exact checks, NevIR, probes, Qwen audit, logs | 7 October 2026, Colab NVIDIA A100-SXM4-40GB |
| `main_v3/pred/predictions.zip` | raw predictions of every run (item-level decisions and option probabilities) | same run |
| `main_v3/followup/np_v3_main/` | float64 NegPairs-Perm predictions and seed means of the static encoders | CPU |
| `indep_v3/out/`, `indep_v3/out_f64/` | independent recomputation from the raw predictions (scripts in the repository, `indep/`) | CPU |
| `public_v3/public_systems/` | 18 released checkpoints on NegPairs-Perm, NegPairs-Fresh and NevIR: scores per system (`scores/`; the bfloat16 first pass of five systems in `scores/old/`), metrics, intervals, precision check, anchor against the main run, the package snapshot used | 21 and 22 October 2026, Colab NVIDIA L4 |
| `finetune_v3/finetune_li/` | ColBERTv2 and answerai-colbert-small-v1 fine-tuned on NegPairs v1 train or NevIR train (3 seeds): training and selection histories, scores, metrics, SciFact nDCG@10 per query, the package snapshot used | 22 October 2026, Colab NVIDIA L4 |
| `paper2/registry.json` | number registry of the earlier (v2.0) manuscript, from which the article imports the NegPairs v1 construction figures (`v2.` keys) | |

Third-party data and models are not included: NevIR, CondaQA, GloVe, BEIR SciFact and the model checkpoints are
downloaded by the notebooks at pinned revisions; NevIR results are stored as instance identifiers and scores only.

License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
