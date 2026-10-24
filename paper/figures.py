# -*- coding: utf-8 -*-
"""Figures of the manuscripts, drawn from registry.json only (no number is typed here)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from pathlib import Path  # noqa: E402

from fmt import REG, fmt_key, val  # noqa: E402

OUT = Path(__file__).resolve().parent / 'fig'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'savefig.dpi': 300})

ENCS = ['E0', 'E3', 'E1', 'E2', 'E4', 'E5', 'E6']
STATIC = ('E0', 'E3')
COL = {'E0': '#7f7f7f', 'E3': '#b0b0b0', 'E1': '#1f5fa8', 'E2': '#c4622d', 'E4': '#2e8b57', 'E5': '#8e44ad', 'E6': '#d4a017'}
MK = {'E0': 'o', 'E3': 'v', 'E1': 's', 'E2': 'D', 'E4': '^', 'E5': 'P', 'E6': 'X'}
L = {
    'en': {'ylab': 'Scope sensitivity SS (percentage points)', 'seen': 'Seen families (syntactic, quantifier)', 'unseen': 'Unseen families (affixal, lexical)',
           'xlab': 'Aggregation rule', 'E0': 'E0 hash (static)', 'E3': 'E3 GloVe (static)', 'E1': 'E1 MiniLM', 'E2': 'E2 DeBERTa-v3-base', 'E4': 'E4 MPNet',
           'E5': 'E5 BGE', 'E6': 'E6 ColBERTv2', 'C1': 'C1 fine-tuned cross-encoder', 'first': 'decisive clause first', 'second': 'decisive clause second',
           'bal': 'order-balanced', 'ylab_perm': 'Permutation scope sensitivity (percentage points)', 'xlab_perm': 'Encoder (A1) and C1',
           'ylab_nevir': 'Paired accuracy (%)', 'all': 'all instances', 'nested': 'nested subset', 'complement': 'complement', 'random': 'random ranking',
           'ylab_probe': 'Probe accuracy (%)', 'cv': 'cross-validation (seen)', 'transfer': 'transfer (seen to unseen)', 'syn': 'syntactic: negation', 'twins': 'syntactic: neutral twins',
           'chance': 'chance', 'native': 'M0-native'},
    'tr': {'ylab': 'Kapsam duyarlılığı SS (yüzde puan)', 'seen': 'Görülen aileler (sözdizimsel, niceleyici)', 'unseen': 'Görülmeyen aileler (ekli, sözcüksel)',
           'xlab': 'Birleştirme kuralı', 'E0': 'E0 karma (statik)', 'E3': 'E3 GloVe (statik)', 'E1': 'E1 MiniLM', 'E2': 'E2 DeBERTa-v3-base', 'E4': 'E4 MPNet',
           'E5': 'E5 BGE', 'E6': 'E6 ColBERTv2', 'C1': 'C1 ince ayarlı çapraz kodlayıcı', 'first': 'belirleyici yan tümce önce', 'second': 'belirleyici yan tümce sonra',
           'bal': 'sıra dengeli', 'ylab_perm': 'Permütasyon kapsam duyarlılığı (yüzde puan)', 'xlab_perm': 'Kodlayıcı (A1) ve C1',
           'ylab_nevir': 'Eşleştirilmiş doğruluk (%)', 'all': 'tüm örnekler', 'nested': 'iç içe alt küme', 'complement': 'tümleyen', 'random': 'rastgele sıralama',
           'ylab_probe': 'Sonda doğruluğu (%)', 'cv': 'çapraz doğrulama (görülen)', 'transfer': 'aktarım (görülenden görülmeyene)', 'syn': 'sözdizimsel: olumsuzluk', 'twins': 'sözdizimsel: nötr ikizler',
           'chance': 'şans', 'native': 'M0-native'},
}


def fig_ssfresh(lang):
    t = L[lang]
    rules = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'D1', 'M1', 'M0']
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.0), sharex=True)
    for ax, sub in zip(axes, ('seen', 'unseen')):
        for k, e in enumerate(ENCS):
            xs, ys, lo, hi = [], [], [], []
            for i, a in enumerate(rules):
                m = val(f'sm.fresh.{e}.{a}.{sub}.SS')
                xs.append(i + (k - 3) * 0.1)
                ys.append(m)
                if a == 'M0':
                    lo.append(0.0)
                    hi.append(0.0)
                else:
                    lo.append(m - val(f'ci.fresh.{e}.{a}.{sub}.SS.low'))
                    hi.append(val(f'ci.fresh.{e}.{a}.{sub}.SS.high') - m)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=MK[e], ms=3.8, color=COL[e], ecolor=COL[e], elinewidth=0.7, capsize=1.5, label=t[e], lw=0)
        c1 = val(f'sm.fresh.C1.C1.{sub}.SS')
        ax.axhline(c1, ls='--', lw=0.8, color='black', label=t['C1'])
        ax.axhline(0, lw=0.5, color='black')
        ax.set_title(t[sub], fontsize=9, loc='left')
        ax.set_ylim(-45, 100)
        ax.grid(axis='y', lw=0.3, alpha=0.6)
    axes[1].set_xticks(range(len(rules)))
    axes[1].set_xticklabels(rules)
    axes[1].set_xlabel(t['xlab'])
    fig.text(0.015, 0.58, t['ylab'], rotation=90, va='center', ha='left', fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(t[k]) for k in ENCS + ['C1']]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc='lower center', ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.53, -0.01))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.2, hspace=0.25)
    p = OUT / f'fig_ssfresh_{lang}.png'
    fig.savefig(p)
    plt.close(fig)
    return p


def _perm(e, metric):
    return f'f64.{e}.A1.{metric}' if e in STATIC else f'sm.perm.{e}.A1.{metric}'


def _perm_ci(e, metric):
    return f'cif64.{e}.A1.{metric}' if e in STATIC else f'ci.perm.{e}.A1.{metric}'


def fig_permorder(lang):
    t = L[lang]
    groups = ENCS + ['C1']
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    w = 0.34
    for i, e in enumerate(groups):
        for j, (metric, color, lab) in enumerate((('seen.first.SS', '#1f5fa8', t['first']), ('seen.second.SS', '#d9a441', t['second']))):
            if e == 'C1':
                m = val(f'sm.perm.C1.C1.{metric}')
                lo, hi = val(f'ci.perm.C1.C1.{metric}.low'), val(f'ci.perm.C1.C1.{metric}.high')
            else:
                m = val(_perm(e, metric))
                lo, hi = val(_perm_ci(e, metric) + '.low'), val(_perm_ci(e, metric) + '.high')
            x = i + (j - 0.5) * w
            ax.bar(x, m, width=w, color=color, edgecolor='black', linewidth=0.4, label=lab if i == 0 else None)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt='none', ecolor='black', elinewidth=0.6, capsize=1.5)
        bal = val('sm.perm.C1.C1.seen.SS_bal') if e == 'C1' else val(_perm(e, 'seen.SS_bal'))
        ax.plot([i], [bal], marker='D', ms=4, color='black', ls='none', label=t['bal'] if i == 0 else None)
    ax.axhline(0, lw=0.5, color='black')
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_xlabel(t['xlab_perm'])
    ax.set_ylabel(t['ylab_perm'])
    ax.set_ylim(-25, 100)
    ax.grid(axis='y', lw=0.3, alpha=0.6)
    ax.legend(frameon=False, fontsize=7.5, loc='upper left', ncol=3)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.96, bottom=0.17)
    p = OUT / f'fig_permorder_{lang}.png'
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_nevir(lang):
    t = L[lang]
    scorers = [('E1', 'M0'), ('E5', 'M0'), ('E6', 'M0'), ('E4', 'M0'), ('E2', 'M0'), ('E6', 'M0n'), ('E3', 'M0'), ('E0', 'M0')]
    labels = [t[e] if a == 'M0' else t['native'] for e, a in scorers]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    w = 0.27
    for j, (sub, color, lab) in enumerate((('all', '#1f5fa8', t['all']), ('nested', '#c4622d', t['nested']), ('complement', '#9fb3c8', t['complement']))):
        for i, (e, a) in enumerate(scorers):
            m = val(f'nevir.{e}.{a}.{sub}.paired')
            lo, hi = val(f'nevir.{e}.{a}.{sub}.low'), val(f'nevir.{e}.{a}.{sub}.high')
            x = i + (j - 1) * w
            ax.bar(x, m, width=w, color=color, edgecolor='black', linewidth=0.4, label=lab if i == 0 else None)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt='none', ecolor='black', elinewidth=0.6, capsize=1.2)
    ax.axhline(val('nevir.published.random'), ls=':', lw=0.9, color='black', label=t['random'])
    ax.set_xticks(range(len(scorers)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=7.5)
    ax.set_ylabel(t['ylab_nevir'])
    ax.set_ylim(0, 35)
    ax.grid(axis='y', lw=0.3, alpha=0.6)
    ax.legend(frameon=False, fontsize=7.5, loc='upper right', ncol=2)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.96, bottom=0.3)
    p = OUT / f'fig_nevir_{lang}.png'
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_probing(lang):
    t = L[lang]
    encs = ['E2', 'E1', 'E5', 'E4', 'E6', 'E0', 'E3']
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    series = (('cv_mean', '#1f5fa8', t['cv']), ('transfer', '#9fb3c8', t['transfer']), ('syn_cv', '#2e8b57', t['syn']), ('twins_cv', '#d9a441', t['twins']))
    w = 0.2
    for j, (key, color, lab) in enumerate(series):
        for i, e in enumerate(encs):
            k = f'probe.{e}.{key}'
            if k not in REG:
                continue
            x = i + (j - 1.5) * w
            ax.bar(x, val(k), width=w, color=color, edgecolor='black', linewidth=0.4, label=lab if i == 0 else None)
    ax.axhline(50, ls=':', lw=0.9, color='black', label=t['chance'])
    ax.set_xticks(range(len(encs)))
    ax.set_xticklabels([t[e] for e in encs], rotation=20, ha='right', fontsize=7.5)
    ax.set_ylabel(t['ylab_probe'])
    ax.set_ylim(40, 100)
    ax.grid(axis='y', lw=0.3, alpha=0.6)
    ax.legend(frameon=False, fontsize=7.5, loc='upper right', ncol=2)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.96, bottom=0.3)
    p = OUT / f'fig_probing_{lang}.png'
    fig.savefig(p)
    plt.close(fig)
    return p


FIGS = {'ssfresh': fig_ssfresh, 'permorder': fig_permorder, 'nevir': fig_nevir, 'probing': fig_probing}


PUB_ORDER = ['colbertv2', 'answerai', 'gtemc', 'bgem3c', 'bgem3d', 'bgem3s', 'bgebase', 'e5', 'mpnet', 'gtemb', 'splade', 'minilm', 'bgerr', 'bgerr2', 'mxbai', 'monot5b', 'monot5l', 'olmo', 'phi', 'qwen3']
PUB_SHORT = {'colbertv2': 'ColBERTv2', 'answerai': 'answerai-colbert-small', 'gtemc': 'GTE-ModernColBERT', 'bgem3c': 'bge-m3 multi-vector', 'bgem3d': 'bge-m3 dense', 'bgem3s': 'bge-m3 sparse',
             'bgebase': 'bge-base-en-v1.5', 'e5': 'e5-base-v2', 'mpnet': 'all-mpnet-base-v2', 'gtemb': 'gte-modernbert-base', 'splade': 'SPLADE', 'minilm': 'ms-marco-MiniLM-L6',
             'bgerr': 'bge-reranker-base', 'bgerr2': 'bge-reranker-v2-m3', 'mxbai': 'mxbai-rerank-base', 'monot5b': 'monoT5-base', 'monot5l': 'monoT5-3B', 'olmo': 'OLMo-2-7B', 'phi': 'Phi-4-mini', 'qwen3': 'Qwen3-8B'}
PUB_GROUP = {'colbertv2': 'li', 'answerai': 'li', 'gtemc': 'li', 'bgem3c': 'li', 'bgem3d': 'dense', 'bgem3s': 'sparse', 'bgebase': 'dense', 'e5': 'dense', 'mpnet': 'dense', 'gtemb': 'dense',
             'splade': 'sparse', 'minilm': 'ce', 'bgerr': 'ce', 'bgerr2': 'ce', 'mxbai': 'ce', 'monot5b': 'ce', 'monot5l': 'ce', 'olmo': 'llm', 'phi': 'llm', 'qwen3': 'llm'}
PUB_COL = {'li': '#d4a017', 'dense': '#1f5fa8', 'sparse': '#2e8b57', 'ce': '#c4622d', 'llm': '#8e44ad'}
PUB_GL = {'en': {'li': 'late interaction', 'dense': 'dense', 'sparse': 'learned sparse', 'ce': 'cross-encoder / monoT5', 'llm': 'LLM judge', 'xlab': 'Order-balanced permutation scope sensitivity (points)',
                 'static': 'static token states (exact 0)'},
          'tr': {'li': 'geç etkileşim', 'dense': 'yoğun', 'sparse': 'öğrenilmiş seyrek', 'ce': 'çapraz kodlayıcı / monoT5', 'llm': 'LLM yargıç', 'xlab': 'Sıra dengeli permütasyon kapsam duyarlılığı (puan)',
                 'static': 'statik token durumları (tam 0)'}}


def fig_pubss(lang, narrow=False):
    """Dot plot of the order-balanced permutation scope sensitivity of the released systems with 95% intervals.
    narrow=True draws the single-column version for the two-column typesetting (PDF, 7.7 cm wide)."""
    t = PUB_GL[lang]
    order = [k for g in ('li', 'dense', 'sparse', 'ce', 'llm') for k in PUB_ORDER if PUB_GROUP[k] == g]
    fig, ax = plt.subplots(figsize=(3.03, 2.75) if narrow else (6.8, 4.6))
    fs = 6.3 if narrow else 7.5
    ys = list(range(len(order)))[::-1]
    seen = set()
    for y, k in zip(ys, order):
        g = PUB_GROUP[k]
        m, lo, hi = val(f'pub.{k}.perm.SS_bal'), val(f'pub.{k}.perm.SS_bal.low'), val(f'pub.{k}.perm.SS_bal.high')
        ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt='o', color=PUB_COL[g], ecolor=PUB_COL[g], elinewidth=0.8 if narrow else 0.9, capsize=1.5 if narrow else 2,
                    markersize=3.2 if narrow else 4.5, label=t[g] if g not in seen else None)
        seen.add(g)
    ax.axvline(0, ls=':', lw=0.9, color='black', label=t['static'])
    ax.set_yticks(ys)
    ax.set_yticklabels([PUB_SHORT[k] for k in order], fontsize=fs)
    ax.set_xlabel(t['xlab'], fontsize=7 if narrow else 9)
    ax.tick_params(axis='x', labelsize=fs)
    ax.grid(axis='x', lw=0.3, alpha=0.6)
    ax.legend(frameon=False, fontsize=5.8 if narrow else 7.5, loc='upper right', handletextpad=0.4, borderaxespad=0.3)
    if narrow:
        fig.subplots_adjust(left=0.40, right=0.98, top=0.99, bottom=0.17)
        p = OUT / f'fig_pubss_col_{lang}.pdf'
    else:
        fig.subplots_adjust(left=0.27, right=0.98, top=0.98, bottom=0.11)
        p = OUT / f'fig_pubss_{lang}.png'
    fig.savefig(p)
    if not narrow:
        fig.savefig(OUT / f'fig_pubss_{lang}.pdf')
        fig_pubss(lang, narrow=True)
    plt.close(fig)
    return p


FIGS['pubss'] = fig_pubss

if __name__ == '__main__':
    for lang in ('en', 'tr'):
        for name, fn in FIGS.items():
            print(fn(lang))
