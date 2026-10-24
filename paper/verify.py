# -*- coding: utf-8 -*-
"""Number and rule check of a built manuscript (version 3.0).

1. Every digit in the document (body text, tables, captions, equations) must be explained: either it is part of a
   string rendered from the registry (recorded in build/used_<lang>.json), or it belongs to a whitelisted structural
   pattern (section, table, figure, equation and proposition labels, identifiers, dates, math indices). Anything else
   is reported as UNEXPLAINED.
2. Every registry value used in the document is re-read from its source file where the source is a JSON file with a
   simple path (seed means, per-run values, gate, rules, float64 seed means) and compared with the registry.
3. Long dashes (U+2014) are reported (rule 13).
4. The keys used in the English and the Turkish document are compared.
Usage: python verify.py en|tr|both
"""
import os
import json
import re
import sys
import zipfile
from pathlib import Path

from lxml import etree

HERE = Path(__file__).resolve().parent
BUILD = HERE / 'build'
# root that holds the archived inputs (see paper/README.md for the expected layout); override with CACSA_DATA_ROOT
_REPO = Path(__file__).resolve().parent.parent
# data root that holds the archived inputs (see README.md); override with CACSA_DATA_ROOT
S = Path(os.environ.get('CACSA_DATA_ROOT', _REPO / 'outputs' if (_REPO / 'outputs').is_dir() else _REPO))
MAIN = S / 'main_v3/np_v3_main'
FU = S / 'main_v3/followup/np_v3_main'
REG = json.load(open(HERE / 'registry.json', encoding='utf-8'))
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'


def paragraphs(docx):
    """(style, text, math) per paragraph, with OMML text separated; hyperlink targets are not text."""
    z = zipfile.ZipFile(docx)
    root = etree.fromstring(z.read('word/document.xml'))
    out = []
    for p in root.iter(f'{{{W}}}p'):
        st = p.find(f'{{{W}}}pPr/{{{W}}}pStyle')
        style = st.get(f'{{{W}}}val') if st is not None else ''
        prose, math = [], []
        for el in p.iter(f'{{{W}}}t', f'{{{M}}}t'):
            in_math = el.tag == f'{{{M}}}t' or any(a.tag == f'{{{M}}}oMath' for a in el.iterancestors())
            (math if in_math else prose).append(el.text or '')
        out.append((style, ' '.join(prose), ' '.join(math)))
    return out


IDENT = [
    'claude-haiku-4-5-20251001', 'claude-sonnet-5', 'claude-opus-5', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen2.5-7B-Instruct', 'Qwen2.5',
    'paraphrase-multilingual-MiniLM-L12-v2', 'all-mpnet-base-v2', 'bge-base-en-v1.5', 'colbertv2.0', 'colbert-ir', 'glove.6B.300d', 'glove.6B.zip',
    'microsoft/deberta-v3-base', 'DeBERTa-v3-base', 'DeBERTa-v3-large', 'deberta-v3-base', 'DeBERTaV3', 'BERT-base', 'BERT-large', 'RoBERTa-large',
    'np_main_20261004T154358Z', 'main_20261003T050337Z', 'np_v3_main', 'NegPairs_v3_build_colab.ipynb', 'NegPairs_v3_main_colab.ipynb',
    'FILE_MANIFEST_SHA256.csv', 'SHA-256', 'Float64', 'float64', 'float32', 'bfloat16', 'TF32', 'L2-normalized', 'L2', 'A100-SXM4-40GB', 'A100', 'L4', 'T4',
    'CACSA v2', 'v1.1', 'v2.0', 'v3.0', 'v3', 'v2', 'v1', 'cacsa2', 'ColBERTv2', 'ColBERTv1', 'BERTScore', 'MiniLM-L12', 'MiniLM', 'MPNet', 'BGE', 'GloVe',
    'en_core_web_sm', '45140', '0000-0002-6660-0565', 'Manisa', 'SEM 2012', '*SEM 2012', 'SEM 2023', 'ECIR 2021', 'SIGIR 2024', 'EMNLP 2020', 'EMNLP 2022',
    'ICLR 2020', 'ICLR 2021', 'ICLR 2019', 'ICLR 2023', 'NeurIPS 2020', 'NeurIPS 2019', 'NIPS 2017', 'ACL-IJCNLP 2021', 'EMNLP-IJCNLP', 'CoNLL-2009',
    'PMLR 80', 'arXiv:2309.07597', 'arXiv:2412.15115', 'arXiv:2012.09650', 'orionweller/NevIR', 'stanfordnlp/glove', 'perm-aff-00007', 'fresh-syn-00034',
    'Proposition 1', 'Proposition 2', 'Propositions 1 and 2', 'K-fold', 'perm_predictions_f64.jsonl', 'gate.json', 'rules.json', 'nevir_results.json',
    'probing.json', 'qwen_audit_v1.json', 'Perm (≥', 'SS_perm,bal', 'SS_perm', 'SS_bal', 'flip_O→N', 'flip_O→S', 'flip_O→P', 'acc_OP', 'EMNLP 2014',
    'Attribution 4.0', 'Atıf 4.0', 'CC BY 4.0',
    # released systems (Section 5) and fine-tuning (Section 6)
    'BAAI/bge-m3', 'bge-m3', 'BGE-M3', 'intfloat/e5-base-v2', 'e5-base-v2', 'cross-encoder/ms-marco-MiniLM-L6-v2', 'ms-marco-MiniLM-L6-v2', 'MiniLM-L6',
    'castorini/monot5-base-msmarco-10k', 'castorini/monot5-3b-msmarco-10k', 'monot5-base-msmarco-10k', 'monot5-3b-msmarco-10k', 'monoT5-base', 'monoT5-3B',
    'monoT5', 'allenai/OLMo-2-1124-7B-Instruct', 'OLMo-2-1124-7B-Instruct', 'OLMo-2-7B-Instruct', 'OLMo-2-7B', 'OLMo 2', 'Qwen/Qwen3-8B', 'Qwen3-8B', 'Qwen3',
    'microsoft/Phi-4-mini-instruct', 'Phi-4-mini-instruct', 'Phi-4-mini', 'Phi-4', 'Claude Haiku 4.5', 'Claude Sonnet 5', 'BAAI/bge-reranker-v2-m3',
    'bge-reranker-v2-m3', 'BAAI/bge-reranker-base', 'bge-reranker-base', 'answerdotai/answerai-colbert-small-v1', 'answerai-colbert-small-v1',
    'lightonai/GTE-ModernColBERT-v1', 'GTE-ModernColBERT-v1', 'GTE-ModernColBERT', 'Alibaba-NLP/gte-modernbert-base', 'gte-modernbert-base',
    'mixedbread-ai/mxbai-rerank-base-v1', 'mxbai-rerank-base-v1', 'naver/splade-v3', 'splade-v3', 'SPLADE-v3', 'sentence-transformers/all-mpnet-base-v2',
    'BAAI/bge-base-en-v1.5', 'colbert-ir/colbertv2.0', 'Sentence-Transformers', 'ModernBERT', 'XLM-RoBERTa', 'tacl2021v1', 'NevIR-FT', 'NegPairs-FT',
    'float16', 'e5',
]
LABEL_WORDS_EN = r'(?:Tables?|Figures?|Eqs?\.|Sections?|Appendix|Appendices|Corollary|Corollaries|Remark|Rule|Rules|Propositions?)'
LABEL_WORDS_TR = r'(?:Tablo(?:lar)?|Şekil|Denk\.|Bölüm(?:ler)?|Ek|Sonuç|Açıklama|Kural(?:lar)?|Önerme(?:ler)?)'

_RX = {}


def _rm_used(t, used_strings, long_only):
    key = (id(used_strings), long_only)
    if key not in _RX:
        cand = sorted({s_ for s_ in used_strings if s_ and any(c.isdigit() for c in s_) and (len(s_) >= 5 or not long_only)}, key=len, reverse=True)
        alt = '|'.join(re.escape(s_) for s_ in cand)
        _RX[key] = re.compile(r'(?<![0-9A-Za-z])(?<![0-9][.,])(?:' + alt + r')(?![0-9A-Za-z])(?![.,][0-9])')
    return _RX[key].sub(' ', t)


def strip_explained(text, used_strings, lang):
    t = _rm_used(text, used_strings, True)
    for s_ in sorted(IDENT, key=len, reverse=True):
        t = t.replace(s_, ' ')
    t = re.sub(r'\b[3-7] (?:October|Ekim) 2026\b', ' ', t)
    t = re.sub(r'\b(?:[Vv]ersion|[Mm]anifest|[Ss]ürüm|[Mm]anifesto?) [123]\.[01]\b', ' ', t)
    t = re.sub(r'\b(?:19|20)\d\d[a-z]?\b', ' ', t)                       # years in author-year citations and references
    t = re.sub(r'\(pp\. \d+(?:–\d+)?\)|\bpp\. \d+(?:–\d+)?|\b\d+\(\d+\), \d+–\d+|, \d+, \d+–\d+|\bvolume \d+|\b\d+ \(\d+\)', ' ', t)   # pages, volumes
    t = re.sub(r'https?://\S+', ' ', t)
    t = re.sub(r'\b\d+(?:st|nd|rd|th)\b', ' ', t)                          # ordinals in venue names
    lw = LABEL_WORDS_TR if lang == 'tr' else LABEL_WORDS_EN
    t = re.sub(lw + r'\s*\(?[A-E]?\d+(?:\.\d+)?\)?(?:\s*(?:and|to|ile|ve|,|–)\s*\(?[A-E]?\d+(?:\.\d+)?\)?)*', ' ', t)
    t = re.sub(r'\b(?:Table|Tablo|Figure|Şekil)\s+[A-E]?\d+\.', ' ', t)
    t = re.sub(r'^\s*(?:Appendix |Ek )?[A-E]?\d+(?:\.\d+)?\.?\s', ' ', t)   # heading numbers
    t = re.sub(r'\b(?:A[1-6]|D1|C[0-2]|E[0-6]|R[1-8]|M[01]|M0-native|G)\b', ' ', t)
    t = re.sub(r'\b[ABC]\.\d\b', ' ', t)                                   # corollary and remark labels
    t = re.sub(r'(?<!\d)95 ?%|%95(?!\d)|(?<!\d)10 ?%|%10(?!\d)', ' ', t)     # confidence level; audit share
    t = re.sub(r'(?<![\d.,])(?:100|50|0|1|2|3|4)(?![\d.,])', ' ', t)         # scale, chance, exact zeros and small structural counts
    t = re.sub(r'\bq_?[12]\b|\bd_?[12]\b|\bo\^?\{?1-y\}?', ' ', t)
    t = re.sub(r'KOTAN1,\*|\b1 Department|\b1 Yapay', ' ', t)
    t = re.sub(r'10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+', ' ', t)
    t = _rm_used(t, used_strings, False)
    return t


def strip_math(m, used_strings):
    t = m
    t = re.sub(r'\s\d{1,2}\s*$', ' ', t)                       # equation number
    t = re.sub(r'(?<![\d.])100(?![\d.])|(?<!\d)4\s*d|(?<![\d.])0\.05(?![\d.])|(?<![\d.])128(?![\d.])', ' ', t)
    t = _rm_used(t, used_strings, False)
    t = re.sub(r'[0-9]', lambda mm: ' ' if mm.group(0) in '012' else mm.group(0), t)   # indices and small constants
    return t


def check(lang):
    docx = BUILD / f'paper_{lang}.docx'
    used = json.load(open(BUILD / f'used_{lang}.json', encoding='utf-8'))
    used_strings = sorted({s for v in used.values() for s in v}, key=len, reverse=True)
    meta = json.load(open(BUILD / f'meta_{lang}.json', encoding='utf-8'))
    paras = paragraphs(docx)
    problems = []
    in_refs = False
    for style, text, mtext in paras:
        if text.strip() in ('References', 'Kaynaklar'):
            in_refs = True
            continue
        if text.startswith('Appendix') or text.startswith('Ek '):
            in_refs = False
        if style == 'Bibliography' or in_refs:
            continue
        if '—' in text:
            problems.append(('LONG DASH', text[:120]))
        rest = strip_explained(text, used_strings, lang)
        for m in re.finditer(r'\d+(?:[.,]\d+)*', rest):
            ctx = rest[max(0, m.start() - 40):m.end() + 40].replace('\n', ' ')
            problems.append(('UNEXPLAINED', f'{m.group(0)!r} in: {ctx}'))
        if mtext.strip():
            mrest = strip_math(mtext, used_strings)
            if re.search(r'\d', mrest):
                problems.append(('UNEXPLAINED-MATH', mtext[:160]))
    # re-read registry values from simple JSON sources
    sm = {}
    for short, name in (('v1', 'v1test'), ('fresh', 'fresh'), ('perm', 'perm')):
        for r in json.load(open(MAIN / f'seed_means_{name}.json')):
            sm[(name, r['encoder'], r['aggregator'])] = r
    f64 = {(r['encoder'], r['aggregator']): r for r in json.load(open(FU / 'seed_means_perm_f64_static.json'))}
    pr = {}
    for name in ('v1test', 'fresh', 'perm'):
        for r in json.load(open(MAIN / f'per_run_{name}.json')):
            pr[(name, r['encoder'], r['aggregator'], r['seed'])] = r
    gate = json.load(open(MAIN / 'gate.json'))
    rules = json.load(open(MAIN / 'rules.json'))
    checked = 0
    for key in used:
        e = REG[key]
        how, src = e['how'], e['src']
        m = re.match(r'\[([^,\]]+),([^\]]+)\]\.(.+)$', how)
        if src.endswith('seed_means_perm_f64_static.json') and m:
            enc, agg, metric = m.groups()
            metric = metric.replace(' (float64 scores)', '')
            v = f64[(enc, agg)].get(metric)
            checked += 1
            if v is None or abs(v - e['value']) > 1e-12:
                problems.append(('MISMATCH', key))
        elif re.search(r'seed_means_(v1test|fresh|perm)\.json$', src) and m and 'seed=' not in how:
            name = re.search(r'seed_means_(v1test|fresh|perm)\.json$', src).group(1)
            enc, agg, metric = m.groups()
            v = sm[(name, enc, agg)].get(metric)
            checked += 1
            if v is None or abs(v - e['value']) > 1e-12:
                problems.append(('MISMATCH', key))
        elif re.search(r'per_run_(v1test|fresh|perm)\.json$', src) and 'seed=' in how and how.startswith('['):
            name = re.search(r'per_run_(v1test|fresh|perm)\.json$', src).group(1)
            mm = re.match(r'\[([^,]+),([^,]+),seed=(\d+)\]\.(.+)$', how)
            if mm:
                enc, agg, sd, metric = mm.groups()
                v = pr[(name, enc, agg, int(sd))].get(metric)
                checked += 1
                if v is None or abs(v - e['value']) > 1e-12:
                    problems.append(('MISMATCH', key))
        elif src.endswith('gate.json') and how.startswith('encoders.') and how.endswith('A1_seed_mean_dev_SS'):
            enc = how.split('.')[1]
            checked += 1
            if abs(gate['encoders'][enc]['A1_seed_mean_dev_SS'] - e['value']) > 1e-12:
                problems.append(('MISMATCH', key))
        elif src.endswith('rules.json') and re.match(r'(R2|R3|R8)\.([^.]+)\.t_interval\.(mean|low|high)$', how):
            rule, enc, field = re.match(r'(R2|R3|R8)\.([^.]+)\.t_interval\.(mean|low|high)$', how).groups()
            checked += 1
            if abs(rules[rule][enc]['t_interval'][field] - e['value']) > 1e-12:
                problems.append(('MISMATCH', key))
    # re-read the released-system and fine-tuning values (revision step 1 and 3) by walking the JSON path in 'how'
    checked += reread_json_paths(used, problems)
    return paras, used, meta, problems, checked


_JSON_CACHE = {}


def _jload(path):
    if path not in _JSON_CACHE:
        _JSON_CACHE[path] = json.load(open(path, encoding='utf-8'))
    return _JSON_CACHE[path]


def _walk(obj, path):
    """Follow a dotted path with [key] / [index] brackets; dict keys may themselves contain dots (joined tokens)."""
    tokens = []
    for m in re.finditer(r'\[([^\]]+)\]|([^.\[\]]+)', path):
        if m.group(1) is not None:
            s = m.group(1)
            tokens.append(('idx', int(s)) if re.fullmatch(r'-?\d+', s) else ('key', s))
        else:
            tokens.append(('key', m.group(2)))
    i = 0
    while i < len(tokens):
        kind, s = tokens[i]
        if kind == 'idx':
            obj = obj[s]
            i += 1
            continue
        if isinstance(obj, list):
            obj = {r['name']: r for r in obj}                     # metrics_full.json is a list of rows keyed by name
        if s in obj:
            obj = obj[s]
            i += 1
            continue
        j = i + 1
        joined = s
        while j < len(tokens) and tokens[j][0] == 'key':
            joined = joined + '.' + tokens[j][1]
            if joined in obj:
                break
            j += 1
        if joined in obj:
            obj = obj[joined]
            i = j + 1
        else:
            raise KeyError(path)
    return obj


def reread_json_paths(used, problems):
    """Re-read every used pub.* / ft.* / hc.* value whose 'how' is a plain JSON path into its source file."""
    n = 0
    for key in used:
        if not (key.startswith('pub.') or key.startswith('ft.') or key.startswith('nevir.')):
            continue
        e = REG[key]
        src, how = e['src'], e['how']
        if not str(src).endswith('.json'):
            continue
        src = str(src) if os.path.isabs(str(src)) else str(S / str(src))       # registry paths are relative to the data root
        if not os.path.exists(src):
            continue
        how = re.sub(r'\s*\([^)]*\)\s*$', '', how).strip()           # drop trailing remarks such as "(over seeds)"
        if not re.fullmatch(r'[\w\-/|:\[\]. ]+', how) or ' ' in how.strip() or how.startswith('sum') or ' - ' in how or how.startswith('ft.'):
            continue
        slice_ = None
        m = re.match(r'^(.*)\[:(\d+)\]$', how)
        if m:
            how, slice_ = m.group(1), int(m.group(2))
        if any(w in how for w in ('values min', 'values max', 'mean /', 'min', 'max')) and not re.search(r'\.(min|max)$', how):
            continue
        try:
            v = _walk(_jload(src), how)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if slice_ is not None and isinstance(v, str):
            v = v[:slice_]
        n += 1
        want = e['value']
        if isinstance(want, str):
            if isinstance(v, list):
                v = ', '.join(str(x) for x in v)
            ok = isinstance(v, str) and (v == want or v.startswith(want))
        elif isinstance(want, (int, float)) and isinstance(v, (int, float)) and not isinstance(v, bool):
            ok = abs(v - want) <= 1e-9 * max(1.0, abs(want))
        else:
            ok = v == want
        if not ok:
            problems.append(('MISMATCH', f'{key}: registry {want!r} vs {src}:{how} -> {v!r}'))
    return n


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'en'
    langs = ['en', 'tr'] if which == 'both' else [which]
    res = {}
    for lg in langs:
        paras, used, meta, problems, checked = check(lg)
        res[lg] = used
        print(f'== {lg}: {len(paras)} paragraphs, {len(used)} registry keys used, {checked} values re-read from source files, {len(problems)} problems')
        for kind, ctx in problems:
            print(f'  {kind}: {ctx}')
    if len(langs) == 2:
        a, b = set(res['en']), set(res['tr'])
        print('keys only in EN:', sorted(a - b)[:80])
        print('keys only in TR:', sorted(b - a)[:80])
