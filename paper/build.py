# -*- coding: utf-8 -*-
"""Build the English and Turkish manuscripts: template -> Markdown (numbers, labels, citations) -> pandoc -> docx
(native OMML equations) -> styling. Usage: python build.py [en|tr|both]"""
import json
import os
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

import fmt
from refs import REFS, sort_key
from tables import TABLES

HERE = Path(__file__).resolve().parent
BUILD = HERE / 'build'
BUILD.mkdir(exist_ok=True)
FONT = 'Times New Roman'
DARK = RGBColor(0x16, 0x21, 0x3E)


# ============================================================================================ reference document
def make_reference_docx(path, lang):
    tmp = BUILD / 'pandoc_default_reference.docx'
    subprocess.run(['pandoc', '-o', str(tmp), '--print-default-data-file', 'reference.docx'], check=True)
    doc = Document(tmp)
    st = doc.styles
    lang_code = 'tr-TR' if lang == 'tr' else 'en-US'

    def font(s, size=None, bold=None, italic=None, color=None, name=FONT):
        f = s.font
        f.name = name
        rpr = s.element.get_or_add_rPr()
        rf = rpr.find(qn('w:rFonts'))
        if rf is None:
            rf = OxmlElement('w:rFonts')
            rpr.append(rf)
        for a in ('w:ascii', 'w:hAnsi', 'w:cs', 'w:eastAsia'):
            rf.set(qn(a), name)
        for a in ('w:asciiTheme', 'w:hAnsiTheme', 'w:cstheme', 'w:eastAsiaTheme'):
            if rf.get(qn(a)) is not None:
                del rf.attrib[qn(a)]
        if size is not None:
            f.size = Pt(size)
        if bold is not None:
            f.bold = bold
        if italic is not None:
            f.italic = italic
        if color is not None:
            f.color.rgb = color

    def para(s, align=None, before=None, after=None, line=None, first=None, left=None, hanging=None, keep_next=None):
        pf = s.paragraph_format
        if align is not None:
            pf.alignment = align
        if before is not None:
            pf.space_before = Pt(before)
        if after is not None:
            pf.space_after = Pt(after)
        if line is not None:
            pf.line_spacing = line
        if first is not None:
            pf.first_line_indent = Pt(first)
        if left is not None:
            pf.left_indent = Pt(left)
        if hanging is not None:
            pf.left_indent = Pt(hanging)
            pf.first_line_indent = Pt(-hanging)
        if keep_next is not None:
            pf.keep_with_next = keep_next

    def get(name, base='Normal', kind=1):
        try:
            return st[name]
        except KeyError:
            s = st.add_style(name, kind)
            s.base_style = st[base]
            return s

    normal = st['Normal']
    font(normal, 11)
    para(normal, WD_ALIGN_PARAGRAPH.JUSTIFY, 0, 4, 1.15)
    rpr = normal.element.get_or_add_rPr()
    lg = OxmlElement('w:lang')
    lg.set(qn('w:val'), lang_code)
    rpr.append(lg)
    for nm in ('Body Text', 'First Paragraph'):
        s = get(nm)
        font(s, 11)
        para(s, WD_ALIGN_PARAGRAPH.JUSTIFY, 2, 4, 1.15, first=18 if nm == 'Body Text' else 0)
    s = get('Compact')
    font(s, 11)
    para(s, WD_ALIGN_PARAGRAPH.LEFT, 1, 2, 1.1, first=0)
    for nm, size in (('Heading 1', 12.5), ('Heading 2', 11.5), ('Heading 3', 11)):
        s = get(nm)
        font(s, size, bold=True, italic=(nm == 'Heading 3'), color=DARK)
        para(s, WD_ALIGN_PARAGRAPH.LEFT, 12 if nm == 'Heading 1' else 9, 4, 1.0, first=0, keep_next=True)
    s = get('Title')
    font(s, 18, bold=True, color=DARK)
    para(s, WD_ALIGN_PARAGRAPH.CENTER, 24, 6, 1.0, first=0)
    s = get('Subtitle')
    font(s, 13.5, bold=False, italic=False, color=DARK)
    para(s, WD_ALIGN_PARAGRAPH.CENTER, 0, 14, 1.0, first=0)
    s = get('Author')
    font(s, 11.5)
    para(s, WD_ALIGN_PARAGRAPH.CENTER, 0, 4, 1.0, first=0)
    s = get('Affiliation')
    font(s, 9.5)
    para(s, WD_ALIGN_PARAGRAPH.CENTER, 0, 2, 1.0, first=0)
    s = get('Abstract')
    font(s, 10)
    para(s, WD_ALIGN_PARAGRAPH.JUSTIFY, 0, 0, 1.1, first=0)
    s.paragraph_format.left_indent = Cm(0.4)
    s.paragraph_format.right_indent = Cm(0.4)
    ppr = s.element.get_or_add_pPr()
    bdr = OxmlElement('w:pBdr')
    for edge in ('top', 'left', 'bottom', 'right'):
        e = OxmlElement(f'w:{edge}')
        e.set(qn('w:val'), 'single')
        e.set(qn('w:sz'), '6')
        e.set(qn('w:space'), '6')
        e.set(qn('w:color'), '5B6B8C')
        bdr.append(e)
    ppr.append(bdr)
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), 'F3F5F9')
    ppr.append(shd)
    s = get('Keywords')
    font(s, 10)
    para(s, WD_ALIGN_PARAGRAPH.LEFT, 10, 10, 1.0, first=0)
    for nm in ('Table Caption', 'Image Caption', 'Caption'):
        s = get(nm)
        font(s, 9.5, italic=False)
        para(s, WD_ALIGN_PARAGRAPH.JUSTIFY, 6, 6, 1.0, first=0, keep_next=(nm == 'Table Caption'))
    s = get('Bibliography')
    font(s, 9.5)
    para(s, WD_ALIGN_PARAGRAPH.LEFT, 0, 3, 1.0, hanging=22)
    s = get('Captioned Figure')
    para(s, WD_ALIGN_PARAGRAPH.CENTER, 6, 0, 1.0, first=0, keep_next=True)
    s = get('Figure')
    para(s, WD_ALIGN_PARAGRAPH.CENTER, 6, 0, 1.0, first=0, keep_next=True)
    s = get('Block Text')
    font(s, 10.5)
    para(s, WD_ALIGN_PARAGRAPH.JUSTIFY, 4, 6, 1.1, first=0)
    s.paragraph_format.left_indent = Cm(0.6)
    s.paragraph_format.right_indent = Cm(0.6)
    # inline code (model identifiers, hashes): monospace two points smaller than the body text, so that the
    # wider monospace glyphs do not stand out from the surrounding serif text
    s = st['Verbatim Char']
    font(s, 9, name='Consolas')
    s = get('Source Code')
    font(s, 9, name='Consolas')
    para(s, WD_ALIGN_PARAGRAPH.LEFT, 2, 2, 1.0, first=0)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.3)
    sec.top_margin = sec.bottom_margin = Cm(2.3)
    doc.save(path)


# ============================================================================================ template processing
REPO_URL = 'https://github.com/kurbankotan/negation-scope-maxsim'
AUTHOR_BLOCK = {
    'named': ('::: {custom-style="Author"}\nKurban KOTAN^1,\\*^ [![](assets/orcid.png){width=0.32cm}](https://orcid.org/0000-0002-6660-0565)\n:::\n\n'
              '::: {custom-style="Affiliation"}\n^1^ Department of Artificial Intelligence and Machine Learning, Faculty of Computer and Information '
              'Sciences, Manisa Celal Bayar University, Manisa, 45140, Turkey\n:::\n\n'
              '::: {custom-style="Affiliation"}\n\\*Corresponding author: kurban.kotan@cbu.edu.tr\n:::'),
    'anonymous': '::: {custom-style="Author"}\nAnonymous submission\n:::',
}
REPO_TEXT = {
    'named': f'the GitHub repository <{REPO_URL}>',
    'anonymous': 'a GitHub repository (link withheld for anonymous review)',
}


def process(lang, cite_tex=False, anonymous=False):
    """Render the template. cite_tex=True writes natbib citation commands (for the LaTeX build) instead of author-year
    text and writes build/paper_<lang>_tex.md; tables are numbered continuously through the appendices (TACL style)."""
    tpl = (HERE / f'paper_{lang}.md').read_text(encoding='utf-8')
    suffix = '_tex' if cite_tex else ''
    # author block and repository link: anonymous for a double-blind submission, named otherwise
    mode = 'anonymous' if anonymous else 'named'
    for marker, text in (('<<AUTHORS>>', AUTHOR_BLOCK[mode]), ('<<REPO>>', REPO_TEXT[mode])):
        if marker not in tpl:
            raise ValueError(f'{marker} missing in the template')
        tpl = tpl.replace(marker, text)

    # tables: <<TABLE:name>> or <<TABLE:name:X>> (the appendix letter is accepted and ignored: numbering is continuous)
    def tab(m):
        name = m.group(1)
        cap, md = TABLES[name](lang)
        word = 'Tablo' if lang == 'tr' else 'Table'
        return f'\n: **{word} {{{{def:tab:{name}}}}}.** {cap}\n\n{md}\n'
    src = re.sub(r'<<TABLE:(\w+)(?::([A-Z]))?>>', tab, tpl)
    if not cite_tex:
        src = src.replace(' ZZNLZZ ', ' ')     # cell line break marker: a space in the docx, \newline in LaTeX
        src = src.replace('| § ', '| ')        # group header rows of a table: a plain first cell in the docx

    # figures: <<FIG:name|caption>>
    def fig(m):
        name, cap = m.group(1), m.group(2).strip()
        word = 'Şekil' if lang == 'tr' else 'Figure'
        return f'\n![**{word} {{{{def:fig:{name}}}}}.** {cap}](fig/fig_{name}_{lang}.png){{width=16cm}}\n'
    src = re.sub(r'<<FIG:(\w+)\|(.*?)>>', fig, src, flags=re.S)

    # pending blocks: <<PENDING:tag>> is rendered as a visible note in drafts; the final build refuses them
    def pending(m):
        if os.environ.get('PAPER_FINAL'):
            raise ValueError(f'pending block {m.group(1)} in a final build')
        return ('\n**[Bekleyen: ' if lang == 'tr' else '\n**[Pending: ') + m.group(1) + ' results are inserted here when available.]**\n'
    src = re.sub(r'<<PENDING:(\w+)>>', pending, src)

    # numbers from the registry
    src = fmt.render(src, lang)

    # labels: {{def:kind:name}} and {{ref:kind:name}}
    labels, counters = {}, {}
    for m in re.finditer(r'\{\{def:(\w+):([\w-]+)\}\}', src):
        kind, name = m.group(1), m.group(2)
        counters[kind] = counters.get(kind, 0) + 1
        prefix = kind[3:] if kind.startswith('tab') else ''
        key = (kind if not kind.startswith('tab') else 'tab', name)
        if key in labels:
            raise ValueError(f'duplicate label {key}')
        labels[key] = f'{prefix}{counters[kind]}'
    src = re.sub(r'\{\{def:(\w+):([\w-]+)\}\}', lambda m: labels[(m.group(1) if not m.group(1).startswith('tab') else 'tab', m.group(2))], src)

    def ref(m):
        kind, name = m.group(1), m.group(2)
        k = (kind, name)
        if k not in labels:
            raise KeyError(f'undefined label {k}')
        return labels[k]
    src = re.sub(r'\{\{ref:(\w+):([\w-]+)\}\}', ref, src)
    if '{{' in src:
        raise ValueError('unresolved label: ' + src[src.index('{{'):src.index('{{') + 60])

    # citations: author-year. [@a; @b] -> (Cite Year; Cite Year); bare @key -> Cite (Year)
    used = []

    def note(k):
        if k not in REFS:
            raise KeyError(f'unknown reference {k}')
        if k not in used:
            used.append(k)

    def paren(m):
        keys = re.findall(r'@(\w+)', m.group(1))
        for k in keys:
            note(k)
        if cite_tex:
            return '`\\citep{' + ','.join(keys) + '}`{=latex}'
        return '(' + '; '.join(f"{REFS[k]['cite']} {REFS[k]['year']}" for k in keys) + ')'
    def noparen(m):
        k = m.group(1)
        note(k)
        if cite_tex:
            return '`\\citealp{' + k + '}`{=latex}'
        return f"{REFS[k]['cite']} {REFS[k]['year']}"
    src = re.sub(r'\[-@(\w+)\]', noparen, src)
    src = re.sub(r'\[(@[\w]+(?:;\s*@[\w]+)*)\]', paren, src)

    def bare(m):
        k = m.group(1)
        note(k)
        if cite_tex:
            return '`\\citet{' + k + '}`{=latex}'
        return f"{REFS[k]['cite']} ({REFS[k]['year']})"
    src = re.sub(r'(?<![\w@])@(\w+)', bare, src)
    unused = [k for k in REFS if k not in used]
    head = 'Kaynaklar' if lang == 'tr' else 'References'
    bib = [f'\n# {head}\n']
    for k in sorted(used, key=sort_key):
        bib.append(f'::: {{custom-style="Bibliography"}}\n{REFS[k]["text"]}\n:::\n')
    src = src.replace('<<REFERENCES>>', '\n'.join(bib))
    meta = {'labels': {f'{k[0]}:{k[1]}': v for k, v in labels.items()}, 'citation_order': sorted(used, key=sort_key), 'unused_refs': unused}
    (BUILD / f'paper_{lang}{suffix}.md').write_text(src, encoding='utf-8')
    (BUILD / f'meta_{lang}{suffix}.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    return src, meta


# ============================================================================================ docx post-processing
def set_cell_border(cell, **kw):
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.find(qn('w:tcBorders'))
    if b is None:
        b = OxmlElement('w:tcBorders')
        tcPr.append(b)
    for edge, (sz, color) in kw.items():
        e = b.find(qn(f'w:{edge}'))
        if e is None:
            e = OxmlElement(f'w:{edge}')
            b.append(e)
        e.set(qn('w:val'), 'single' if sz else 'nil')
        if sz:
            e.set(qn('w:sz'), str(sz))
            e.set(qn('w:space'), '0')
            e.set(qn('w:color'), color)


def style_tables(doc):
    for t in doc.tables:
        tblPr = t._tbl.tblPr
        for old in tblPr.findall(qn('w:tblBorders')):
            tblPr.remove(old)
        borders = OxmlElement('w:tblBorders')
        for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
            e = OxmlElement(f'w:{edge}')
            e.set(qn('w:val'), 'nil')
            borders.append(e)
        tblPr.append(borders)
        w = tblPr.find(qn('w:tblW'))
        if w is None:
            w = OxmlElement('w:tblW')
            tblPr.append(w)
        w.set(qn('w:type'), 'pct')
        w.set(qn('w:w'), '5000')
        jc = tblPr.find(qn('w:jc'))
        if jc is None:
            jc = OxmlElement('w:jc')
            tblPr.append(jc)
        jc.set(qn('w:val'), 'center')
        rows = t.rows
        n = len(rows)
        for ri, row in enumerate(rows):
            trPr = row._tr.get_or_add_trPr()
            cs = OxmlElement('w:cantSplit')
            trPr.append(cs)
            if ri == 0:
                th = OxmlElement('w:tblHeader')
                trPr.append(th)
            for cell in row.cells:
                kw = {}
                if ri == 0:
                    kw['top'] = (8, '1F2A44')
                    kw['bottom'] = (6, '1F2A44')
                if ri == n - 1:
                    kw['bottom'] = (8, '1F2A44')
                if kw:
                    set_cell_border(cell, **kw)
                for p in cell.paragraphs:
                    p.paragraph_format.space_before = Pt(1.5)
                    p.paragraph_format.space_after = Pt(1.5)
                    p.paragraph_format.first_line_indent = Pt(0)
                    for r in p.runs:
                        r.font.size = Pt(8.5)
                        if ri == 0:
                            r.font.bold = True
                if ri == 0:
                    tcPr = cell._tc.get_or_add_tcPr()
                    shd = OxmlElement('w:shd')
                    shd.set(qn('w:val'), 'clear')
                    shd.set(qn('w:color'), 'auto')
                    shd.set(qn('w:fill'), 'E8ECF3')
                    tcPr.append(shd)
        # math inside table cells: smaller size is set on runs; OMML text follows the paragraph font size in Word


def add_page_numbers(doc):
    for sec in doc.sections:
        p = sec.footer.paragraphs[0] if sec.footer.paragraphs else sec.footer.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run()
        for tag, text in (('begin', None), (None, 'PAGE'), ('end', None)):
            if tag:
                fc = OxmlElement('w:fldChar')
                fc.set(qn('w:fldCharType'), tag)
                r._r.append(fc)
            else:
                it = OxmlElement('w:instrText')
                it.set(qn('xml:space'), 'preserve')
                it.text = text
                r._r.append(it)
        r.font.size = Pt(9)


M_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'


def merge_math_letters(doc):
    """Merge runs of single upright letters (pandoc splits \\mathrm{softplus} into one run per letter)."""
    from lxml import etree
    body = doc.element.body
    merged = 0
    for parent in body.iter(f'{{{M_NS}}}oMath', f'{{{M_NS}}}e', f'{{{M_NS}}}sub', f'{{{M_NS}}}sup', f'{{{M_NS}}}num',
                            f'{{{M_NS}}}den', f'{{{M_NS}}}fName', f'{{{M_NS}}}lim'):
        kids = list(parent)
        i = 0
        while i < len(kids):
            k = kids[i]
            def is_letter_run(el):
                if el.tag != f'{{{M_NS}}}r':
                    return False
                t = el.find(f'{{{M_NS}}}t')
                sty = el.find(f'{{{M_NS}}}rPr/{{{M_NS}}}sty')
                return t is not None and t.text is not None and len(t.text) == 1 and t.text.isalpha() and sty is not None and sty.get(f'{{{M_NS}}}val') == 'p'
            if is_letter_run(k):
                j = i + 1
                while j < len(kids) and is_letter_run(kids[j]):
                    j += 1
                if j - i > 1:
                    t0 = k.find(f'{{{M_NS}}}t')
                    t0.text = ''.join(kids[x].find(f'{{{M_NS}}}t').text for x in range(i, j))
                    for x in range(i + 1, j):
                        parent.remove(kids[x])
                    merged += 1
                    kids = list(parent)
            i += 1
    return merged


def size_table_math(doc, half_points=17):
    from lxml import etree
    for t in doc.tables:
        for r in t._tbl.iter(f'{{{M_NS}}}r'):
            rpr = r.find(f'{{{W_NS}}}rPr')
            if rpr is None:
                rpr = etree.SubElement(r, f'{{{W_NS}}}rPr')
                r.remove(rpr)
                t_el = r.find(f'{{{M_NS}}}t')
                r.insert(list(r).index(t_el), rpr)
            sz = rpr.find(f'{{{W_NS}}}sz')
            if sz is None:
                sz = etree.SubElement(rpr, f'{{{W_NS}}}sz')
            sz.set(f'{{{W_NS}}}val', str(half_points))


def keep_part_labels(doc):
    """A table part label such as '(b) ...' stays on the page of the table it introduces."""
    for para in doc.paragraphs:
        if para.text.startswith('(b) '):
            para.paragraph_format.keep_with_next = True
            para.paragraph_format.space_before = Pt(6)
            para.paragraph_format.space_after = Pt(2)


def postprocess(path, lang, anonymous=False):
    doc = Document(path)
    merge_math_letters(doc)
    size_table_math(doc)
    style_tables(doc)
    keep_part_labels(doc)
    add_page_numbers(doc)
    # equations: make display equations slightly larger spacing; nothing else to change (OMML is native)
    cp = doc.core_properties
    cp.author = '' if anonymous else 'Kurban Kotan'
    cp.title = ('MaxSim ailesi puanlayıcılarda olumsuzluk kapsamı' if lang == 'tr'
                else 'Negation Scope in MaxSim-Family Scorers')
    cp.language = 'tr-TR' if lang == 'tr' else 'en-US'
    cp.comments = ''
    doc.save(path)
    import ooxml_fix
    ooxml_fix.fix_docx(path)


def build(lang, anonymous=False):
    # draw only the figures the template places (<<FIG:name|caption>>)
    names = re.findall(r'<<FIG:(\w+)\|', (HERE / f'paper_{lang}.md').read_text(encoding='utf-8'))
    if names:
        import figures
        for name in names:
            figures.FIGS[name](lang)
    ref = BUILD / f'reference_{lang}.docx'
    make_reference_docx(ref, lang)
    src, meta = process(lang, anonymous=anonymous)
    out = BUILD / f'paper_{lang}.docx'
    cmd = ['pandoc', str(BUILD / f'paper_{lang}.md'), '-f', 'markdown+pipe_tables+tex_math_dollars+fenced_divs+bracketed_spans+implicit_figures+superscript+subscript',
           '-t', 'docx', '--reference-doc', str(ref), '-o', str(out), '--resource-path', str(HERE)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit(1)
    if r.stderr.strip():
        print('pandoc:', r.stderr.strip()[:2000])
    postprocess(out, lang, anonymous)
    used = {k: sorted(v) for k, v in fmt.USED[lang].items()}
    (BUILD / f'used_{lang}.json').write_text(json.dumps(used, indent=0, ensure_ascii=False), encoding='utf-8')
    print(f'{lang}: {out}  citations={len(meta["citation_order"])} unused={meta["unused_refs"]} keys={len(used)}')
    return out


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    which = args[0] if args else 'en'
    for lg in (['en', 'tr'] if which == 'both' else [which]):
        build(lg, anonymous='--anon' in sys.argv)
