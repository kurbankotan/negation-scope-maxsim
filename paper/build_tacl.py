# -*- coding: utf-8 -*-
"""Typeset the English manuscript for TACL (tacl2021v1.sty, A4, two columns, anonymous submission by default).

Input: the template paper_en.md rendered by build.process('en', cite_tex=True) (numbers, labels resolved; citations
as natbib commands) -> build/paper_en_tex.md. Output: build/tacl/paper_en.tex and paper_en.pdf (pdflatex + bibtex).
Tables become LaTeX floats sized to a column or to the full width; the LaTeX numbers are checked against the labels
that build.py resolved in the text. The appendices follow the references (TACL appendix policy of 1 March 2024).

usage: python build_tacl.py [--final]      (--final: camera-ready option with the author block)
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / 'build'
TEX = BUILD / 'tacl'
# TACL style files (tacl2021v1.sty, acl_natbib.bst; from https://transacl.org, Author Guidelines): paper/tacl_style/
STYLE = next((p for p in (HERE / 'tacl_style', HERE.parent / 'tacl_style') if (p / 'tacl2021v1.sty').exists()), HERE / 'tacl_style')
BIB = next((p for p in (HERE / 'references.bib', HERE.parent / 'bibfetch' / 'references.bib') if p.exists()), HERE / 'references.bib')
sys.path.insert(0, str(HERE))
import build as paper_build  # noqa: E402
from refs import REFS  # noqa: E402

COLUMN_PT = 219.3          # 7.7 cm
TEXTWIDTH_PT = 455.1       # 16.0 cm
MD_FORMAT = 'markdown+pipe_tables+tex_math_dollars+fenced_divs+bracketed_spans+implicit_figures+superscript+subscript+raw_attribute'
SIZES = [('\\small', 10.0), ('\\footnotesize', 9.0), ('\\scriptsize', 8.0)]      # \scriptsize is redefined to 8pt below
LONG_SPLIT = 52            # rows above which an appendix table is split into two side-by-side or stacked parts


# ------------------------------------------------------------------------------------------------ pandoc helpers
def pandoc(md):
    r = subprocess.run(['pandoc', '-f', MD_FORMAT, '-t', 'latex', '--wrap=none'], input=md, capture_output=True, text=True, check=True)
    return r.stdout.strip()


SEP = 'ZZCELLSEPZZ'


def pandoc_cells(cells):
    if not cells:
        return []
    out = pandoc(('\n\n' + SEP + '\n\n').join(c if c.strip() else '~' for c in cells))
    parts = [p.strip() for p in out.split(SEP)]
    if len(parts) != len(cells):
        raise ValueError(f'cell conversion mismatch: {len(parts)} parts for {len(cells)} cells')
    return [('' if cells[i].strip() == '' else p) for i, p in enumerate(parts)]


def fix_unicode(tex):
    sup = {'⁻': '-', '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'}
    return re.sub('[' + ''.join(sup) + ']+', lambda m: '\\textsuperscript{' + ''.join(sup[c] for c in m.group(0)) + '}', tex)


# ------------------------------------------------------------------------------------------------ tables
def split_row(line):
    line = line.strip()
    if line.startswith('|'):
        line = line[1:]
    if line.endswith('|'):
        line = line[:-1]
    return [c.strip().replace('\\|', '|') for c in re.split(r'(?<!\\)\|', line)]


def parse_sep(line):
    aligns, widths = [], []
    for c in split_row(line):
        left, right = c.startswith(':'), c.endswith(':')
        aligns.append('c' if (left and right) else 'r' if right else 'l')
        widths.append(c.count('-'))
    return aligns, widths


def plain(cell):
    s = re.sub(r'\*\*?', '', cell)
    s = re.sub(r'\$([^$]*)\$', r'\1', s)
    s = s.replace('\\', '')
    return s


def est_width(text, size_pt):
    """Rough width of a text in pt: Times digits and lowercase ~0.5 em, uppercase ~0.68 em, thin marks ~0.3 em;
    code spans (Courier) 0.6 em per character."""
    w = 0.0
    for m in re.finditer(r'`([^`]*)`|([^`]+)', text):
        if m.group(1) is not None:
            w += 0.6 * len(m.group(1))
            continue
        for ch in m.group(2):
            if ch.isupper():
                w += 0.68
            elif ch in '.,;:|il\'()[] ':
                w += 0.3
            else:
                w += 0.5
    return w * size_pt


def is_numeric(cell):
    """A cell without letters (numbers, intervals, signs, dashes) is never wrapped."""
    return not re.search(r'[A-Za-z]', plain(cell))


def longest_piece(cell, size_pt):
    """Width of the longest unbreakable part of a cell: a numeric cell as a whole; otherwise the longest piece between
    spaces, hyphens and slashes (LaTeX breaks after an explicit hyphen, and \\nolinkurl breaks at / and -)."""
    if is_numeric(cell) or len(plain(cell)) <= 26:
        return est_width(plain(cell), size_pt)
    code = cell.strip().startswith('`')
    pieces = re.split(r'\s+|(?<=[-/])', plain(cell))
    return max((est_width(('`' + w + '`') if code else w, size_pt) for w in pieces if w), default=0.0)


def breakable_code(tex):
    def repl(m):
        inner = m.group(1)
        if len(inner) > 14 and '\\' not in inner and ('/' in inner or '-' in inner or '_' in inner):
            return '\\nolinkurl{' + inner + '}'
        return m.group(0)
    return re.sub(r'\\texttt\{([^{}]*)\}', repl, tex)


def split_header(h):
    """A header longer than its data column is set on two lines (split at the middle space)."""
    words = h.split()
    if len(words) < 2:
        return None
    best, bi = None, 0
    for i in range(1, len(words)):
        a, b = ' '.join(words[:i]), ' '.join(words[i:])
        d = abs(len(a) - len(b))
        if best is None or d < best:
            best, bi = d, i
    return ' '.join(words[:bi]), ' '.join(words[bi:])


def layout(header, rows, aligns, force_wide=False):
    """Choose column or full width, the font size and the column widths.
    Returns (wide, size, colsep, widths or None, header_lines) where widths is a list of p-column widths (None for
    natural width) and header_lines maps column index -> (line1, line2) for two-line headers."""
    ncols = len(header)
    body_rows = [r for r in rows if not r[0].startswith('§ ')]
    data_cols = list(zip(*body_rows)) if body_rows else [()] * ncols

    def plan(pt, target, colsep):
        hl = {}
        dat = [max((est_width(plain(c), pt) for c in col), default=0.0) for col in data_cols]
        lw = [max((longest_piece(c, pt) for c in col), default=0.0) for col in data_cols]
        hw = []
        for i, h in enumerate(header):
            w = est_width(plain(h), pt)
            if w > dat[i] + 6 and split_header(plain(h)):
                a, b = split_header(plain(h))
                w = max(est_width(a, pt), est_width(b, pt))
                hl[i] = split_header(h)
            hw.append(w)
        nat = [max(d, h) for d, h in zip(dat, hw)]
        total = sum(nat) + 2 * colsep * ncols
        if total <= target:
            return None, hl, nat
        # wrap the text columns: numeric and short columns keep their natural width
        fixed = [i for i in range(ncols) if nat[i] <= 56 or all(is_numeric(c) for c in data_cols[i])]
        flex = [i for i in range(ncols) if i not in fixed]
        if not flex:
            return False, hl, nat
        avail = target - 2 * colsep * ncols - 2.0 - sum(nat[i] for i in fixed)
        need = sum(max(lw[i], hw[i]) for i in flex)
        if avail < need:
            return False, hl, nat
        tot = sum(nat[i] for i in flex)
        widths = list(nat)
        for i in flex:
            widths[i] = max(max(lw[i], hw[i]), avail * nat[i] / tot)
        over = sum(widths[i] for i in flex) - avail
        if over > 0:   # take the excess from the columns above their minimum
            room = {i: widths[i] - max(lw[i], hw[i]) for i in flex}
            tot_room = sum(room.values())
            for i in flex:
                widths[i] -= over * room[i] / tot_room if tot_room > 0 else 0
        return widths, hl, nat

    if not force_wide:
        for size, pt in SIZES:
            colsep = 3.0 if pt >= 9 else 2.4
            widths, hl, nat = plan(pt, COLUMN_PT, colsep)
            if widths is None:
                return False, size, colsep, None, hl
    for size, pt in SIZES:
        colsep = 3.5 if pt >= 9 else 2.6
        widths, hl, nat = plan(pt, TEXTWIDTH_PT, colsep)
        if widths is None:
            return True, size, colsep, None, hl
    for size, pt in SIZES[1:]:
        colsep = 3.0 if pt >= 9 else 2.4
        widths, hl, nat = plan(pt, TEXTWIDTH_PT, colsep)
        if widths:
            return True, size, colsep, widths, hl
    raise ValueError('table does not fit the text width: ' + ' | '.join(header))


def tabular(header_t, rows_t, aligns, widths, size, colsep, header_lines):
    cmd = {'l': '\\raggedright', 'r': '\\raggedleft', 'c': '\\centering'}
    if widths:
        spec = ''.join(f'>{{{cmd[a]}\\arraybackslash}}p{{{w:.1f}pt}}' for a, w in zip(aligns, widths))
    else:
        spec = ''.join(aligns)
    cells = []
    for i, h in enumerate(header_t):
        if i in header_lines:
            a, b = header_lines[i]
            cells.append(f'\\makecell[{aligns[i]}]{{\\textbf{{{a}}}\\\\\\textbf{{{b}}}}}')
        else:
            cells.append('\\textbf{' + h + '}')
    head = ' & '.join(cells) + ' \\\\'
    body = '\n'.join((f'\\multicolumn{{{len(r)}}}{{@{{}}l@{{}}}}{{\\textit{{{r[0][2:].strip()}}}}} \\\\' if r[0].startswith('§') else ' & '.join(r) + ' \\\\') for r in rows_t)
    return f'{{{size}\\setlength\\tabcolsep{{{colsep}pt}}\\begin{{tabular}}{{{spec}}}\n\\toprule\n{head}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}}}'


def convert_table_block(caption_md, tables, name, wide_names, placement='t'):
    cap = pandoc(caption_md)
    parts = []
    any_wide = False
    for header, rows, aligns, widths_md, label_md in tables:
        ncols = len(header)
        flat = list(header) + [c for r in rows for c in r]
        conv = [breakable_code(c).replace(' ZZNLZZ ', '\\newline ') for c in pandoc_cells(flat)]
        header_t = conv[:ncols]
        rows_t = [conv[ncols + i * ncols: ncols + (i + 1) * ncols] for i in range(len(rows))]
        for r in rows_t:
            if len(r) != ncols:
                raise ValueError(f'row length {len(r)} != {ncols} in table {name}')
        wide, size, colsep, widths, hl = layout(header, rows, aligns, force_wide=name in wide_names)
        header_lines = {}
        for i, (a, b) in hl.items():
            ta, tb = pandoc_cells([a, b])
            header_lines[i] = (ta, tb)
        any_wide = any_wide or wide
        label_t = pandoc(label_md) if label_md else None
        parts.append((label_t, tabular(header_t, rows_t, aligns, widths, size, colsep, header_lines)))
    inner = []
    for label_t, tab in parts:
        if label_t:
            inner.append(f'{{\\footnotesize\\textit{{{label_t}}}}}\\par\\vskip2pt\n')
        inner.append(f'\\centering {tab}\\par\n')
    if placement == 'H':      # non-floating (one-column appendix): placed exactly here, in order
        return '\\begin{center}\\vskip2pt\n' + '\\vskip4pt\n'.join(inner) + f'\\captionof{{table}}{{{cap}}}\\label{{tab:{name}}}\n\\end{{center}}\\vskip6pt\n', any_wide
    env = 'table*' if any_wide else 'table'
    return f'\\begin{{{env}}}[{placement}]\n' + '\\vskip4pt\n'.join(inner) + f'\\caption{{{cap}}}\\label{{tab:{name}}}\n\\end{{{env}}}\n', any_wide


# ------------------------------------------------------------------------------------------------ document parsing
def parse_divs(md):
    divs = {}
    for m in re.finditer(r'::: \{custom-style="(\w+)"\}\n(.*?)\n:::', md, flags=re.S):
        divs.setdefault(m.group(1), []).append(m.group(2).strip())
    return divs


def transform_body(md, table_names, fig_names, wide_names, report, placement='t'):
    lines = md.split('\n')
    out = []
    i = 0
    tcount = fcount = 0
    first_appendix = True
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^: \*\*Table ([A-Z]?\d+)\.\*\* (.*)$', line)
        if m:
            cap_md = m.group(2)
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            tables, pending_label = [], None
            while j < len(lines) and (lines[j].startswith('|') or (lines[j].startswith('*(') and lines[j].endswith('*'))):
                if lines[j].startswith('*('):
                    pending_label = lines[j].strip('*')
                    j += 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    continue
                tl = []
                while j < len(lines) and lines[j].startswith('|'):
                    tl.append(lines[j])
                    j += 1
                header = split_row(tl[0])
                aligns, widths = parse_sep(tl[1])
                rows = [split_row(x) for x in tl[2:]]
                tables.append((header, rows, aligns, widths, pending_label))
                pending_label = None
                while j < len(lines) and not lines[j].strip():
                    j += 1
            name = table_names[tcount]
            tcount += 1
            tex, wide = convert_table_block(cap_md, tables, name, wide_names, placement)
            report.append((name, 'wide' if wide else 'column', sum(len(t[1]) for t in tables), max(len(t[0]) for t in tables)))
            out += ['```{=latex}', tex, '```', '']
            i = j
            continue
        m = re.match(r'^!\[\*\*Figure (\d+)\.\*\* (.*)\]\((fig/[^)]+)\)\{width=16cm\}$', line)
        if m:
            cap = pandoc(m.group(2))
            name = fig_names[fcount]
            fcount += 1
            col = HERE / 'fig' / f'fig_{name}_col_en.pdf'
            if placement == 'H':
                out += ['```{=latex}', f'\\begin{{center}}\\vskip2pt\n\\includegraphics[width=7.7cm]{{fig/fig_{name}_col_en.pdf}}\n\\captionof{{figure}}{{{cap}}}\\label{{fig:{name}}}\n\\end{{center}}\\vskip6pt', '```']
            elif col.exists():
                out += ['```{=latex}', f'\\begin{{figure}}[{placement if placement == "t" else "tbp"}]\n\\centering\n\\includegraphics[width=\\columnwidth]{{fig/fig_{name}_col_en.pdf}}\n\\caption{{{cap}}}\\label{{fig:{name}}}\n\\end{{figure}}', '```']
            else:
                out += ['```{=latex}', f'\\begin{{figure*}}[{placement}]\n\\centering\n\\includegraphics[width=\\textwidth]{{fig/fig_{name}_en.pdf}}\n\\caption{{{cap}}}\\label{{fig:{name}}}\n\\end{{figure*}}', '```']
            i += 1
            continue
        m = re.match(r'^(#+) (.*)$', line)
        if m:
            level, text = len(m.group(1)), m.group(2)
            ma = re.match(r'^Appendix ([A-E])\. (.*)$', text)
            if ma:
                pre = '\\clearpage\\appendix\n' if first_appendix else ('\\clearpage\\onecolumn\n' if ma.group(2).startswith('Complementary') else '')
                if ma.group(2).startswith('Complementary'):
                    placement = 'H'
                out += ['```{=latex}', pre + f'\\section{{{pandoc(ma.group(2))}}}', '```']
                first_appendix = False
                i += 1
                continue
            text = re.sub(r'^(?:\d+(?:\.\d+)*\.?|[A-E]\.\d+)\s+', '', text)
            out.append('#' * level + ' ' + text)
            i += 1
            continue
        out.append(line)
        i += 1
    if tcount != len(table_names) or fcount != len(fig_names):
        raise ValueError(f'found {tcount} tables and {fcount} figures, expected {len(table_names)} and {len(fig_names)}')
    return '\n'.join(out)


def bibliography(entries_md, meta):
    order = meta['citation_order']
    if len(order) != len(entries_md):
        raise ValueError('reference count mismatch')
    conv = pandoc_cells(entries_md)
    items = []
    for key, tex in zip(order, conv):
        r = REFS[key]
        items.append(f'\\bibitem[{r["cite"]}({r["year"]})]{{{key}}} {tex}')
    return '\\begin{thebibliography}{}\n' + '\n\n'.join(items) + '\n\\end{thebibliography}\n'


PREAMBLE = r'''\documentclass[11pt,a4paper]{article}
\usepackage{times,latexsym}
\usepackage[hyphens]{url}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[%(opt)s]{tacl2021v1}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{array}
\usepackage{booktabs}
\usepackage{makecell}
\usepackage{placeins}
\usepackage{newunicodechar}
\usepackage[font=small,labelfont=bf,skip=4pt]{caption}
\makeatletter
\renewcommand\scriptsize{\@setfontsize\scriptsize{8}{9.5}}   % smallest table size: 8 pt
\makeatother
\hypersetup{pdftitle={Negation Scope in MaxSim-Family Scorers},pdfauthor={},pdfsubject={},pdfkeywords={},pdfcreator={},pdfproducer={}}
\urlstyle{tt}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{…}{\ldots{}}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{±}{\ensuremath{\pm}}
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\dbltextfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\floatsep}{6pt plus 2pt minus 2pt}
\setlength{\dblfloatsep}{6pt plus 2pt minus 2pt}
\renewcommand{\topfraction}{0.95}\renewcommand{\dbltopfraction}{0.95}\renewcommand{\textfraction}{0.05}\renewcommand{\floatpagefraction}{0.85}\renewcommand{\dblfloatpagefraction}{0.85}
\setcounter{topnumber}{3}\setcounter{dbltopnumber}{3}\setcounter{totalnumber}{4}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
'''


def main(final=False):
    md, meta = paper_build.process('en', cite_tex=True, anonymous=not final)
    tpl = (HERE / 'paper_en.md').read_text(encoding='utf-8')
    table_names = re.findall(r'<<TABLE:(\w+)(?::[A-Z])?>>', tpl)
    fig_names = re.findall(r'<<FIG:(\w+)\|', tpl)
    wide_names = set()
    divs = parse_divs(md)
    title = divs['Title'][0]
    subtitle = divs['Subtitle'][0]
    abstract = re.sub(r'^\*\*Abstract\.\*\*\s*', '', divs['Abstract'][0])
    start = md.index('\n# 1. ')
    ref_head = md.index('\n# References\n')
    body_md = md[start:ref_head]
    rest = md[ref_head + len('\n# References\n'):]
    app_start = rest.index('\n# Appendix A.')
    bib_md, app_md = rest[:app_start], rest[app_start:]
    entries = re.findall(r'::: \{custom-style="Bibliography"\}\n(.*?)\n:::', bib_md, flags=re.S)
    n_main_tables = len(re.findall(r'^: \*\*Table \d+\.\*\*', body_md, flags=re.M))
    n_main_figs = len(re.findall(r'^!\[\*\*Figure \d+\.\*\*', body_md, flags=re.M))
    report = []
    body_tex = pandoc(transform_body(body_md, table_names[:n_main_tables], fig_names[:n_main_figs], wide_names, report))
    app_tex = pandoc(transform_body(app_md, table_names[n_main_tables:], fig_names[n_main_figs:], wide_names, report, placement='tp'))
    bib_tex = '\\bibliographystyle{acl_natbib}\n\\bibliography{references}\n'
    opt = 'acceptedWithA' if final else ''
    doc = PREAMBLE.replace('%(opt)s', opt)
    doc += '\\begin{document}\n'
    doc += f'\\title{{{pandoc(title)}: {pandoc(subtitle)}}}\n'
    if final:
        doc += ('\\author{Kurban Kotan \\\\ Department of Artificial Intelligence and Machine Learning \\\\ '
                'Faculty of Computer and Information Sciences \\\\ Manisa Celal Bayar University, Manisa, Turkey \\\\ '
                '\\texttt{kurban.kotan@cbu.edu.tr}}\n')
    else:
        doc += '\\author{Anonymous}\n'
    doc += '\\date{}\n\\maketitle\n'
    doc += f'\\begin{{abstract}}\n{pandoc(abstract)}\n\\end{{abstract}}\n\n'
    doc += body_tex + '\n\n' + bib_tex + '\n' + app_tex + '\n\\end{document}\n'
    doc = fix_unicode(doc)
    doc = breakable_code(doc)          # long identifiers in prose may break at / and -
    doc = doc.replace('\\mathbb{1}', '\\mathbf{1}')
    # manual equation numbers "\qquad (n)" become amsmath tags at the right margin
    doc = re.sub(r',?\\qquad \((\d+)\)\s*\\\]', r'\\tag{\1}\\]', doc)
    doc = re.sub(r'\\\]\n\n([a-z])', r'\\]\n\n\\noindent \1', doc)
    TEX.mkdir(parents=True, exist_ok=True)
    for f in ('tacl2021v1.sty', 'acl_natbib.bst'):
        if not (STYLE / f).exists():
            raise SystemExit(f'{f} not found: download the TACL style files (https://transacl.org, Author Guidelines) into {STYLE}')
        shutil.copy(STYLE / f, TEX / f)
    (TEX / 'fig').mkdir(exist_ok=True)
    for f in (HERE / 'fig').glob('fig_*_en.pdf'):
        shutil.copy(f, TEX / 'fig' / f.name)
    (TEX / 'paper_en.tex').write_text(doc, encoding='utf-8')
    shutil.copy(BIB, TEX / 'references.bib')

    def latex():
        r = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'paper_en.tex'], cwd=TEX, capture_output=True)
        if r.returncode != 0:
            log = (TEX / 'paper_en.log').read_text(encoding='utf-8', errors='replace')
            i = log.find('\n!')
            print(log[i:i + 3000])
            raise SystemExit('pdflatex failed')
    latex()
    r = subprocess.run(['bibtex', 'paper_en'], cwd=TEX, capture_output=True, text=True)
    blg = (TEX / 'paper_en.blg').read_text(encoding='utf-8', errors='replace')
    if r.returncode != 0 or 'error' in blg.lower():
        print(blg[-3000:])
        raise SystemExit('bibtex failed')
    for _ in range(3):
        latex()
    aux = (TEX / 'paper_en.aux').read_text(encoding='utf-8', errors='replace')
    bad = []
    for k, v in meta['labels'].items():
        kind, name = k.split(':', 1)
        if kind not in ('tab', 'fig'):
            continue
        m = re.search(r'\\newlabel\{' + kind + ':' + re.escape(name) + r'\}\{\{([^}]*)\}', aux)
        got = m.group(1) if m else None
        if got != v:
            bad.append((k, v, got))
    log = (TEX / 'paper_en.log').read_text(encoding='utf-8', errors='replace')
    n_over = len(re.findall(r'Overfull \\hbox', log))
    n_bibwarn = len(re.findall(r'^Warning--', blg, flags=re.M))
    print('bibtex warnings:', n_bibwarn)
    n_pages = re.search(r'Output written on paper_en.pdf \((\d+) pages', log)
    print('tables:', ', '.join(f'{n}:{w}:{r}x{c}' for n, w, r, c in report))
    print('pages:', n_pages.group(1) if n_pages else '?', '| overfull hboxes:', n_over, '| label mismatches:', bad)
    if bad:
        raise SystemExit(1)


if __name__ == '__main__':
    main(final='--final' in sys.argv)
