# -*- coding: utf-8 -*-
"""Number formatting and placeholder rendering for the manuscripts (EN and TR).

Placeholders in templates:
  «key»        number in the entry's default format (percentages without the % sign)
  «key|%»      with the percent sign (EN 53.07%, TR %53,07)
  «key|+»      explicit sign for positive values (+11.93)
  «key|+%»     both
  «ci:prefix»  "[low, high]" from prefix.low / prefix.high (TR "[low; high]")
  «cl:prefix»  "[low, high]" from prefix.cl_low / prefix.cl_high
  «rng:a:b»    "a to b" range of two keys (EN "a to b", TR "a ile b")
Every key that is rendered is recorded in USED with the exact string produced, so that the verifier can check that
each number in the document came from the registry.
"""
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

HERE = Path(__file__).resolve().parent
REG = json.load(open(HERE / 'registry.json', encoding='utf-8'))
USED = {'en': {}, 'tr': {}}
MINUS = '−'
SUP = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')


def rnd(x, dec):
    q = Decimal(1).scaleb(-dec)
    d = Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)
    if d == 0:
        d = abs(d)
    return d


def fnum(x, dec, lang, sign=False, group=True):
    d = rnd(x, dec)
    s = f'{abs(d):,.{dec}f}' if group else f'{abs(d):.{dec}f}'
    if lang == 'tr':
        s = s.replace(',', '\x00').replace('.', ',').replace('\x00', '.')
    if d < 0:
        s = MINUS + s
    elif sign and d > 0:
        s = '+' + s
    return s


def fsci(x, lang):
    """1e-3 -> '10⁻³'; 2e-5 -> '2 × 10⁻⁵'; 6.84e-5 -> '6.84 × 10⁻⁵' (two significant decimals at most)."""
    m, e = f'{float(x):.2e}'.split('e')
    e = int(e)
    m = m.rstrip('0').rstrip('.')
    if lang == 'tr':
        m = m.replace('.', ',')
    exp = str(e).translate(SUP).replace('⁻', '⁻')
    if m == '1':
        return f'10{exp}'
    return f'{m} × 10{exp}'


def fmt_key(key, lang, mods=''):
    if key not in REG:
        raise KeyError(f'unknown registry key: {key}')
    e = REG[key]
    v, kind, dec = e['value'], e['kind'], e['dec']
    sign = '+' in mods
    pct = '%' in mods
    if kind in ('pct', 'pp'):
        s = fnum(v, 2, lang, sign)
    elif kind == 'int':
        s = fnum(v, 0, lang, sign)
    elif kind == 'float':
        s = fnum(v, dec, lang, sign)
    elif kind == 'usd':
        s = fnum(v, 2, lang, sign)
    elif kind == 'sci':
        s = fsci(v, lang)
    elif kind == 'text':
        s = str(v)
    elif kind == 'id':
        s = str(int(v))
    else:
        raise ValueError(f'{key}: kind {kind} cannot be printed')
    if pct:
        s = ('%' + s) if lang == 'tr' else (s + '%')
    USED[lang].setdefault(key, set()).add(s)
    return s


def fmt_interval(prefix, lang, kind='ci'):
    lo, hi = (f'{prefix}.low', f'{prefix}.high') if kind == 'ci' else (f'{prefix}.cl_low', f'{prefix}.cl_high')
    a, b = fmt_key(lo, lang), fmt_key(hi, lang)
    return f'[{a}; {b}]' if lang == 'tr' else f'[{a}, {b}]'


PH = re.compile(r'«([^»]+)»')


def render(text, lang):
    def sub(m):
        tok = m.group(1)
        if tok.startswith('ci:'):
            return fmt_interval(tok[3:], lang, 'ci')
        if tok.startswith('cl:'):
            return fmt_interval(tok[3:], lang, 'cl')
        if tok.startswith('rng:'):
            _, a, b = tok.split(':')
            sa, sb = fmt_key(a, lang), fmt_key(b, lang)
            return f'{sa} ile {sb}' if lang == 'tr' else f'{sa} to {sb}'
        if '|' in tok:
            key, mods = tok.split('|', 1)
        else:
            key, mods = tok, ''
        return fmt_key(key, lang, mods)
    return PH.sub(sub, text)


def val(key):
    return REG[key]['value']
