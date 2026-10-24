# Manuscript pipeline

Every number in the article is read from the run outputs by `registry.py`, inserted into the text template
`paper_en.md` by key (`«key»`), and checked after the build. No number is typed by hand.

## Inputs

* The run outputs in `outputs/` at the repository root, the default data root (to use a copy elsewhere, set
  `CACSA_DATA_ROOT` to it). Layout: `main_v3/`, `np_data/`, `np_data_v3/`, `indep_v3/`, `public_v3/`, `finetune_v3/`,
  `paper2/` (see `outputs/README.md`).
* The code root (default: the repository; override with `CACSA_CODE_ROOT`) for the `cacsa2` package and
  `human_check/sheets/human_check_key.csv`.
* For the PDF only: the TACL style files `tacl2021v1.sty` and `acl_natbib.bst` (TACL Author Guidelines,
  https://transacl.org) in `paper/tacl_style/`. They are not redistributed here.
* Python packages: `requirements-paper.txt`; system tools: pandoc and TeX Live (pdflatex, bibtex).

## Steps

```
python paper/registry.py      # registry.json and traceability.csv (key, value, source file, computation)
python paper/build.py en      # paper/build/paper_en.docx and paper_en.md (author-year preview; --anon: anonymous)
python paper/verify.py en     # every digit of the docx is explained; values re-read from the source files
python paper/claims.py        # programmatic checks of the comparative statements in the text
python paper/build_tacl.py    # paper/build/tacl/paper_en.tex and .pdf (anonymous submission format)
python paper/build_tacl.py --final   # the same with the author block and the repository link
```

`build.py` draws, with `figures.py`, only the figures that the template places (`<<FIG:name|caption>>`); the
article has none.

`verify.py` reports any digit in the text, tables and captions that is neither a registry value used by the build nor
a structural label, and re-reads used values from their JSON sources; the article build had 0 problems.
`claims.py` checks 142 comparative statements (orderings, ranges, thresholds); all held.
A block `<<PENDING:tag>>` in the template marks text that waits for results; a build with `PAPER_FINAL=1` refuses it.

## Files

| File | Role |
|---|---|
| `paper_en.md` | text template (numbers as registry keys, labels, citations) |
| `registry.py` | number registry from the run outputs |
| `fmt.py` | rounding (half up) and number formatting |
| `tables.py`, `figures.py` | tables and figures, from the registry only |
| `build.py` | template rendering and docx build |
| `build_tacl.py` | LaTeX build in the TACL submission format (tacl2021v1, A4, two columns, BibTeX with acl_natbib) |
| `verify.py`, `claims.py` | checks |
| `refs.py`, `references.bib` | references (author-year text for the docx; BibTeX for the PDF) |
| `reference_verification.md`, `reference_records/` | primary record of every reference (ACL Anthology BibTeX exports, arXiv and proceedings pages) |
| `ooxml_fix.py` | docx post-processing |
| `assets/orcid.png` | ORCID icon of the author block (docx) |
