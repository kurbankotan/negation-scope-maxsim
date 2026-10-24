# -*- coding: utf-8 -*-
"""Tables of the manuscripts as pandoc pipe tables; every number comes from registry.json via fmt."""
import os
import json
from pathlib import Path

from fmt import fmt_key as F, fmt_interval, val, REG

# root that holds the archived inputs (see paper/README.md for the expected layout); override with CACSA_DATA_ROOT
_REPO = Path(__file__).resolve().parent.parent
# data root that holds the archived inputs (see README.md); override with CACSA_DATA_ROOT
S = Path(os.environ.get('CACSA_DATA_ROOT', _REPO / 'outputs' if (_REPO / 'outputs').is_dir() else _REPO))
DATA3 = S / 'np_data_v3/negpairs_v3'
ENCS = ['E0', 'E3', 'E1', 'E2', 'E4', 'E5', 'E6']
STATIC = ('E0', 'E3')
RULES_ALL = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'D1', 'M1', 'M0', 'C0', 'C2']
ENC_NAME = {'E0': 'E0 Hash', 'E3': 'E3 GloVe', 'E1': 'E1 MiniLM', 'E2': 'E2 DeBERTa', 'E4': 'E4 MPNet', 'E5': 'E5 BGE', 'E6': 'E6 ColBERTv2', 'C1': 'C1'}


def pipe(header, rows, align, widths=None):
    if widths is None:
        widths = [1] * len(header)

    def sep(al, w):
        d = '-' * max(3, 4 * w)
        return {'l': ':' + d, 'r': d + ':', 'c': ':' + d + ':'}[al]
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(sep(x, w) for x, w in zip(align, widths)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(r) + ' |')
    return '\n'.join(out)


def T(lang, en, tr):
    return tr if lang == 'tr' else en


def perm_key(e, a, metric):
    """Perm seed means: float64 values for the static encoders, GPU values otherwise."""
    if e in STATIC:
        return f'f64.{e}.{a}.{metric}'
    return f'sm.perm.{e}.{a}.{metric}'


def perm_ci(e, a, metric, lang):
    pre = f'cif64.{e}.{a}.{metric}' if e in STATIC else f'ci.perm.{e}.{a}.{metric}'
    return fmt_interval(pre, lang)


def ci(prefix, lang):
    return fmt_interval(prefix, lang)


def yes_no(v, lang):
    return T(lang, 'met', 'sağlandı') if v else T(lang, 'not met', 'sağlanmadı')


# ------------------------------------------------------------------------------------------- example scenarios
def rows_of(path, sid):
    arms = {}
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        if r['scenario_id'] == sid:
            arms[r['arm']] = r
    return arms


ARM_NAMES = {'O': ('O (affirmative)', 'O (olumlu)'), 'P': ('P (paraphrase)', 'P (yeniden ifade)'),
             'N': ('N (decisive negation)', 'N (belirleyici olumsuzluk)'), 'S': ('S (irrelevant negation)', 'S (ilgisiz olumsuzluk)')}


def t_examplefresh(lang):
    a = rows_of(DATA3 / 'fresh.jsonl', 'fresh-syn-00034')
    opts = a['O']['options']
    head = T(lang, ['Arm', 'Context', 'Cue', 'Answer'], ['Kol', 'Bağlam', 'Belirteç', 'Yanıt'])
    rows = []
    for arm in 'OPNS':
        r = a[arm]
        cue = f'*{r["cue"]}*' if r['cue'] else '–'
        rows.append([T(lang, *ARM_NAMES[arm]), r['context'], cue, opts[r['label']]])
    cap = T(lang,
            f'A NegPairs-Fresh scenario (`{a["O"]["scenario_id"]}`, syntactic family) shown verbatim. Question: "{a["O"]["question"]}" '
            f'Options: "{opts[0]}" and "{opts[1]}". The N cue negates the decisive words; the S cue negates an irrelevant detail. '
            f'The N and S arms differ in their words, so on such pairs a scorer may separate them through word identity (Remark A.4).',
            f'Bir NegPairs-Fresh senaryosu (`{a["O"]["scenario_id"]}`, sözdizimsel aile), veri setindeki İngilizce metniyle. '
            f'Soru: "{a["O"]["question"]}" Seçenekler: "{opts[0]}" ve "{opts[1]}". N kolundaki belirteç belirleyici sözcükleri, '
            f'S kolundaki belirteç ilgisiz bir ayrıntıyı olumsuzlar. N ve S kolları sözcükleriyle ayrılır; bu tür çiftlerde bir '
            f'puanlayıcı iki kolu sözcük kimliğiyle ayırabilir (Açıklama A.4).')
    return cap, pipe(head, rows, 'llll', [3, 9, 2, 3])


def t_exampleperm(lang):
    a = rows_of(DATA3 / 'perm.jsonl', 'perm-aff-00007')
    opts = a['O']['options']
    head = T(lang, ['Arm', 'Context', 'Edit', 'Answer'], ['Kol', 'Bağlam', 'Düzenleme', 'Yanıt'])
    rows = []
    for arm in 'OPNS':
        r = a[arm]
        if arm == 'N':
            ed = T(lang, '*sufficient* → *insufficient* in the decisive clause', 'belirleyici yan tümcede *sufficient* → *insufficient*')
        elif arm == 'S':
            ed = T(lang, '*sufficient* → *insufficient* in the other clause', 'öteki yan tümcede *sufficient* → *insufficient*')
        else:
            ed = '–'
        rows.append([T(lang, *ARM_NAMES[arm]), r['context'], ed, opts[r['label']]])
    cap = T(lang,
            f'A NegPairs-Perm scenario (`{a["O"]["scenario_id"]}`, affixal family, decisive clause second) shown verbatim, including its '
            f'lower-case start (Section 4.1). Question: "{a["O"]["question"]}" Options: "{opts[0]}" and "{opts[1]}". '
            f'N and S contain exactly the same words; Proposition A says that every unmasked scorer over static token states gives them identical scores.',
            f'Bir NegPairs-Perm senaryosu (`{a["O"]["scenario_id"]}`, ekli aile, belirleyici yan tümce ikinci sırada), küçük harfle başlayan '
            f'biçimiyle (Bölüm 4.1). Soru: "{a["O"]["question"]}" Seçenekler: "{opts[0]}" ve "{opts[1]}". N ve S tam olarak aynı '
            f'sözcükleri içerir; Önerme A\'ya göre statik token durumları üzerindeki maskesiz her puanlayıcı ikisine aynı puanı verir.')
    return cap, pipe(head, rows, 'llll', [3, 9, 4, 3])


# ------------------------------------------------------------------------------------------- aggregation rules
def t_aggs(lang):
    head = T(lang, ['ID', 'Aggregation', 'Mask', 'Parameters (*d* = 384)', 'Exact statement (static states)'],
             ['Kimlik', 'Birleştirme', 'Maske', 'Parametre (*d* = 384)', 'Tam ifade (statik durumlar)'])
    P = lambda a: F(f'param.E1.{a}', lang)
    rows = [
        ['A1', T(lang, 'maximum over context tokens', 'bağlam token\'ları üzerinde maksimum'), '–', P('A1'), T(lang, 'A, B', 'A, B')],
        ['A2', T(lang, 'mean over context tokens', 'bağlam token\'ları üzerinde ortalama'), '–', P('A2'), 'A'],
        ['A3', T(lang, 'log-sum-exp, learned temperature', 'log-sum-exp, öğrenilen sıcaklık'), '–', P('A3'), T(lang, 'A, B', 'A, B')],
        ['A4', T(lang, 'signed maximum', 'işaretli maksimum'), T(lang, 'resolver scope', 'çözümleyici kapsamı'), P('A4'), T(lang, 'C (possibility)', 'C (olabilirlik)')],
        ['A5', T(lang, 'signed maximum, learned gate', 'işaretli maksimum, öğrenilen geçit'), T(lang, 'learned from the state', 'durumdan öğrenilen'), P('A5'), 'A'],
        ['A6', T(lang, 'signed maximum', 'işaretli maksimum'), T(lang, 'sentence-wide', 'cümle çapında'), P('A6'), T(lang, 'A.3 (equal marked multisets)', 'A.3 (eşit işaretli çoklu kümeler)')],
        ['D1', T(lang, 'signed maximum', 'işaretli maksimum'), T(lang, 'dataset scope (oracle)', 'veri seti kapsamı (kâhin)'), P('D1'), T(lang, 'C (possibility)', 'C (olabilirlik)')],
        ['M1', T(lang, 'learned MaxSim (linear map, maximum)', 'öğrenilen MaxSim (doğrusal dönüşüm, maksimum)'), '–', P('M1'), T(lang, 'A, B', 'A, B')],
        ['M0', T(lang, 'zero-shot MaxSim (cosine, maximum)', 'sıfır atışlı MaxSim (kosinüs, maksimum)'), '–', F('param.E1.M0', lang), T(lang, 'A, B', 'A, B')],
        ['M0-native', T(lang, 'ColBERTv2 scoring (E6 only)', 'ColBERTv2 puanlaması (yalnızca E6)'), '–', F('param.E6.M0n', lang), T(lang, 'none (contextual)', 'yok (bağlamsal)')],
        ['C0', T(lang, 'head on mean states (control)', 'ortalama durumlar üzerinde baş (kontrol)'), '–', P('C0'), 'A'],
        ['C2', T(lang, 'option only (control)', 'yalnızca seçenek (kontrol)'), '–', P('C2'), T(lang, 'context not read', 'bağlam okunmaz')],
        ['C1', T(lang, 'fine-tuned cross-encoder (reference)', 'ince ayarlı çapraz kodlayıcı (başvuru)'), '–', F('param.C1.C1', lang), T(lang, 'outside the setting', 'çerçevenin dışında')],
    ]
    cap = T(lang,
            'Scoring rules, controls and reference. The pair scorer of Eq. (2) is shared by A1 to A6 and D1; M1 and M0 remove the affinity network. '
            'Parameter counts are for 384-dimensional states (E0, E1); the last column names the statement of Section 3 that applies with a static encoder.',
            'Puanlama kuralları, kontroller ve başvuru. Denk. (2)\'deki çift puanlayıcı A1-A6 ve D1 tarafından paylaşılır; M1 ve M0 yakınlık ağını kaldırır. '
            'Parametre sayıları 384 boyutlu durumlar içindir (E0, E1); son sütun statik bir kodlayıcıyla geçerli olan Bölüm 3 ifadesini verir.')
    return cap, pipe(head, rows, 'lllrl', [1, 4, 3, 2, 3])


# ------------------------------------------------------------------------------------------- encoders
def t_encoders(lang):
    head = T(lang, ['ID', 'Model', 'Kind', '*d*', 'Revision'], ['Kimlik', 'Model', 'Tür', '*d*', 'Sürüm'])
    rows = [
        ['E0', T(lang, 'hash: SHA-256-seeded unit vector per lower-cased token type', 'karma: küçük harfli token türü başına SHA-256 tohumlu birim vektör'), T(lang, 'static', 'statik'), F('enc.E0.dim', lang), '–'],
        ['E3', '`glove.6B.300d` (`stanfordnlp/glove`)', T(lang, 'static', 'statik'), F('enc.E3.dim', lang), F('enc.E3.revision_short', lang) + '…'],
        ['E1', '`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`', T(lang, 'contextual; paraphrase sentence encoder', 'bağlamsal; yeniden ifade cümle kodlayıcısı'), F('enc.E1.dim', lang), F('enc.E1.revision_short', lang) + '…'],
        ['E2', '`microsoft/deberta-v3-base`', T(lang, 'contextual; masked language model', 'bağlamsal; maskeli dil modeli'), F('enc.E2.dim', lang), F('enc.E2.revision_short', lang) + '…'],
        ['E4', '`sentence-transformers/all-mpnet-base-v2`', T(lang, 'contextual; sentence encoder', 'bağlamsal; cümle kodlayıcısı'), F('enc.E4.dim', lang), F('enc.E4.revision_short', lang) + '…'],
        ['E5', '`BAAI/bge-base-en-v1.5`', T(lang, 'contextual; retrieval bi-encoder', 'bağlamsal; erişim çift kodlayıcısı'), F('enc.E5.dim', lang), F('enc.E5.revision_short', lang) + '…'],
        ['E6', '`colbert-ir/colbertv2.0` (BERT-base and the checkpoint\'s projection)', T(lang, 'contextual; late-interaction retriever', 'bağlamsal; geç etkileşimli erişim modeli'), F('enc.E6.dim', lang), F('enc.E6.revision_short', lang) + '…'],
    ]
    cap = T(lang, 'Frozen encoders. Token states are the last-layer states, L2-normalized, with special tokens removed; E6 uses ColBERTv2\'s document mode. '
                  'Revisions are the first eight hexadecimal digits of the Hugging Face commit recorded at run time.',
            'Dondurulmuş kodlayıcılar. Token durumları son katman durumlarıdır; L2 ile normalleştirilmiş, özel token\'lar çıkarılmıştır; E6 ColBERTv2\'nin belge kipini kullanır. '
            'Sürümler, koşu sırasında kaydedilen Hugging Face commit değerinin ilk sekiz onaltılık hanesidir.')
    return cap, pipe(head, rows, 'lllrl', [1, 5, 3, 1, 2])


# ------------------------------------------------------------------------------------------- data sets
def t_data(lang):
    head = T(lang, ['Set', 'Split', 'Syntactic', 'Quantifier', 'Affixal', 'Lexical', 'Scenarios', 'Items'],
             ['Set', 'Bölüm', 'Sözdizimsel', 'Niceleyici', 'Ekli', 'Sözcüksel', 'Senaryo', 'Öğe'])
    rows = []
    for sp, name in (('train', T(lang, 'train', 'eğitim')), ('dev', T(lang, 'dev', 'geliştirme')), ('test', T(lang, 'test (previously read)', 'test (daha önce okundu)'))):
        syn, qua = F(f'v2.data.n.{sp}.syntactic', lang), F(f'v2.data.n.{sp}.quantifier', lang)
        aff = F('v2.data.n.test.affixal', lang) if sp == 'test' else '–'
        lex = F('v2.data.n.test.lexical', lang) if sp == 'test' else '–'
        n = F(f'data.v1.{sp}.scenarios', lang)
        items = F(f'data.v1.{sp}.items', lang)
        rows.append(['NegPairs v1', name, syn, qua, aff, lex, n, items])
    rows.append(['NegPairs-Fresh', T(lang, 'test', 'test'), F('build.fresh.accepted.syntactic', lang), F('build.fresh.accepted.quantifier', lang),
                 F('build.fresh.accepted.affixal', lang), F('build.fresh.accepted.lexical', lang), F('data.fresh.scenarios', lang), F('data.fresh.items', lang)])
    for order, name in (('first', T(lang, 'test, decisive clause first', 'test, belirleyici yan tümce önce')), ('second', T(lang, 'test, decisive clause second', 'test, belirleyici yan tümce sonra'))):
        rows.append(['NegPairs-Perm', name, F(f'build.perm.accepted.syntactic.{order}', lang), F(f'build.perm.accepted.quantifier.{order}', lang),
                     F(f'build.perm.accepted.affixal.{order}', lang), F(f'build.perm.accepted.lexical.{order}', lang), F(f'data.perm.order.{order}', lang), '–'])
    rows.append(['NegPairs-Perm', T(lang, 'test, total', 'test, toplam'), F('data.perm.cue.syntactic.not', lang), F('data.perm.cue.quantifier.no', lang),
                 F('build.perm.affixal_total', lang), F('build.perm.lexical_total', lang), F('data.perm.scenarios', lang), F('data.perm.items', lang)])
    cap = T(lang, 'Scenario counts by set, split and negation family (each scenario has four items). The affixal and lexical families are unseen in training. '
                  'NegPairs v1 counts are those of manifest v2.0; the v1 test split had been read before this study was designed.',
            'Set, bölüm ve olumsuzluk ailesine göre senaryo sayıları (her senaryonun dört öğesi vardır). Ekli ve sözcüksel aileler eğitimde görülmez. '
            'NegPairs v1 sayıları manifest v2.0\'dakilerdir; v1 test bölümü bu çalışma tasarlanmadan önce okunmuştu.')
    return cap, pipe(head, rows, 'llrrrrrr', [4, 6, 3, 3, 2, 3, 3, 2])


# ------------------------------------------------------------------------------------------- gate and rules
def t_rules(lang):
    head = T(lang, ['Rule', 'Encoder', 'Seed mean', '*t* interval', 'Positive seeds', 'Cluster interval', 'Outcome'],
             ['Kural', 'Kodlayıcı', 'Tohum ort.', '*t* aralığı', 'Pozitif tohum', 'Kümeli aralık', 'Sonuç'])
    rows = []
    GROUP = '§ '   # a row whose first cell starts with this marker is a group header (set across all columns in LaTeX)
    g_desc = T(lang, f'G: A1 dev SS, seen families (≥ {F("thr.gate_SS_seen_dev", lang)})', f'G: A1 geliştirme SS, görülen aileler (≥ {F("thr.gate_SS_seen_dev", lang)})')
    rows.append([GROUP + g_desc, '', '', '', '', '', ''])
    for i, e in enumerate(['E6', 'E1', 'E4', 'E5', 'E2', 'E0', 'E3']):
        rows.append(['', e, F(f'G.{e}.mean', lang), T(lang, f'seeds {F(f"G.{e}.min", lang)} to {F(f"G.{e}.max", lang)}', f'tohumlar {F(f"G.{e}.min", lang)} ile {F(f"G.{e}.max", lang)}'),
                     f'{F(f"G.{e}.n_pos", lang)}/{F("n.seeds", lang)}', '–', T(lang, 'passed', 'geçti') if val(f'G.{e}.passed') else T(lang, 'not passed', 'geçemedi')])
    for rule, desc_en, desc_tr, order in (
            ('R2', f'R2: A4 − A1, unseen SS, Fresh (≥ {F("thr.R2_min_gain_unseen_SS", lang)}; level {F("stat.conf_R2R3", lang)}%)',
             f'R2: A4 − A1, görülmeyen SS, Fresh (≥ {F("thr.R2_min_gain_unseen_SS", lang)}; düzey %{F("stat.conf_R2R3", lang)})', ['E0', 'E6', 'E2', 'E5', 'E4', 'E1', 'E3']),
            ('R3', f'R3: A6 − A4, flip_O→S on S-same, Fresh (≥ {F("thr.R3_min_gain_flipS", lang)}; level {F("stat.conf_R2R3", lang)}%)',
             f'R3: A6 − A4, S-aynı üzerinde flip_O→S, Fresh (≥ {F("thr.R3_min_gain_flipS", lang)}; düzey %{F("stat.conf_R2R3", lang)})', ['E2', 'E6', 'E0', 'E5', 'E4', 'E1', 'E3']),
            ('R8', f'R8: A1 SS_perm,bal, seen families, Perm (≥ {F("thr.R8_min_SS_perm_bal_seen", lang)}; level {F("stat.conf_R8", lang)}%)',
             f'R8: A1 SS_perm,bal, görülen aileler, Perm (≥ {F("thr.R8_min_SS_perm_bal_seen", lang)}; düzey %{F("stat.conf_R8", lang)})', ['E4', 'E6', 'E5', 'E1', 'E2'])):
        rows.append([GROUP + T(lang, desc_en, desc_tr), '', '', '', '', '', ''])
        for i, e in enumerate(order):
            evaluated = val(f'{rule}.{e}.evaluated')
            if evaluated:
                out = yes_no(val(f'{rule}.{e}.met'), lang)
                clu = fmt_interval(f'{rule}.{e}', lang, 'cl') if f'{rule}.{e}.cl_low' in REG else '–'
            else:
                out = T(lang, 'not evaluated (gate)', 'değerlendirilmedi (geçit)')
                clu = '–'
            rows.append(['', e, F(f'{rule}.{e}.mean', lang, '+' if rule != 'R8' else ''), ci(f'{rule}.{e}', lang),
                         f'{F(f"{rule}.{e}.pos", lang)}/{F(f"{rule}.{e}.n", lang)}', clu, out])
    rows.append([GROUP + T(lang, 'Exact rules (static encoders, float64): checks and violations', 'Tam kurallar (statik kodlayıcılar, float64): denetimler ve ihlaller'), '', '', '', '', '', ''])
    rows.append([T(lang, 'R5: Proposition B, insertion pairs', 'R5: Önerme B, ekleme çiftleri'), 'E0, E3',
                 T(lang, f'{F("R5.checks", lang)} checks', f'{F("R5.checks", lang)} denetim'), T(lang, f'{F("R5.violations_total", lang)} violations', f'{F("R5.violations_total", lang)} ihlal'), '–', '–', yes_no(val('R5.met'), lang)])
    rows.append([T(lang, 'R6: Proposition A, permutation pairs', 'R6: Önerme A, permütasyon çiftleri'), 'E0, E3',
                 T(lang, f'{F("R6.checks", lang)} checks', f'{F("R6.checks", lang)} denetim'), T(lang, f'{F("R6.score_violations_total", lang)} violations, {F("R6.decision_disagreements_total", lang)} disagreements', f'{F("R6.score_violations_total", lang)} ihlal, {F("R6.decision_disagreements_total", lang)} uyuşmazlık'), '–', '–', yes_no(val('R6.met'), lang)])
    rows.append([T(lang, 'R7: Corollary B.1, nested NevIR instances', 'R7: Sonuç B.1, iç içe NevIR örnekleri'), 'E0, E3',
                 T(lang, f'{F("R7.checks", lang)} checks', f'{F("R7.checks", lang)} denetim'), T(lang, f'{F("R7.violations_total", lang)} violations', f'{F("R7.violations_total", lang)} ihlal'), '–', '–', yes_no(val('R7.met'), lang)])
    cap = T(lang,
            'Gate and prespecified rules (manifest v3.0). Values in percentage points; seed means over ten seeds; the *t* interval is the seed-level interval at the '
            'stated level (nine degrees of freedom; paired within seeds for R2 and R3); the cluster interval is the scenario-cluster bootstrap percentile interval at the same level. '
            'A statistical rule is met only if the mean reaches the threshold, at least nine of ten seeds are positive and both intervals lie above zero. '
            'For the gate the interval column gives the range over seeds; for the exact rules R5 to R7 the mean column gives the number of checks and the interval column the number of violations. '
            'E3 failed the gate, so its R2 and R3 values are descriptive; R8 concerns the contextual encoders only.',
            'Geçit ve önceden belirlenmiş kurallar (manifest v3.0). Değerler yüzde puan; on tohum üzerinden tohum ortalamaları; *t* aralığı belirtilen düzeydeki tohum düzeyinde '
            'aralıktır (dokuz serbestlik derecesi; R2 ve R3 için tohum içinde eşleştirilmiş); kümeli aralık aynı düzeydeki senaryo kümeli bootstrap yüzdelik aralığıdır. '
            'Bir istatistiksel kural ancak ortalama eşiğe ulaşırsa, on tohumun en az dokuzu pozitifse ve her iki aralık sıfırın üstündeyse sağlanmış sayılır. '
            'Geçit için aralık sütunu tohumlar üzerindeki aralığı verir; tam kurallar R5-R7 için ortalama sütunu denetim sayısını, aralık sütunu ihlal sayısını verir. '
            'E3 geçidi geçemedi; R2 ve R3 değerleri betimseldir; R8 yalnızca bağlamsal kodlayıcıları ilgilendirir.')
    return cap, pipe(head, rows, 'llrlrll', [10, 5, 4, 6, 5, 6, 6])


# ------------------------------------------------------------------------------------------- Fresh main table
def t_fresh(lang):
    head = T(lang, ['Encoder', 'A1 seen', 'A1 unseen', 'A4 unseen', 'A6 unseen', 'D1 unseen', 'M1 unseen', 'M0 unseen', 'acc (A1)'],
             ['Kodlayıcı', 'A1 görülen', 'A1 görülmeyen', 'A4 görülmeyen', 'A6 görülmeyen', 'D1 görülmeyen', 'M1 görülmeyen', 'M0 görülmeyen', 'doğruluk (A1)'])
    rows = []
    for e in ENCS:
        rows.append([ENC_NAME[e], F(f'sm.fresh.{e}.A1.seen.SS', lang), F(f'sm.fresh.{e}.A1.unseen.SS', lang), F(f'sm.fresh.{e}.A4.unseen.SS', lang),
                     F(f'sm.fresh.{e}.A6.unseen.SS', lang), F(f'sm.fresh.{e}.D1.unseen.SS', lang), F(f'sm.fresh.{e}.M1.unseen.SS', lang),
                     F(f'sm.fresh.{e}.M0.unseen.SS', lang), F(f'sm.fresh.{e}.A1.all.acc', lang)])
    rows.append(['C1', F('sm.fresh.C1.C1.seen.SS', lang), F('sm.fresh.C1.C1.unseen.SS', lang), '–', '–', '–', '–', '–', F('sm.fresh.C1.C1.all.acc', lang)])
    cap = T(lang,
            'NegPairs-Fresh: scope sensitivity SS (percentage points; seed means over ten seeds, M0 scored once) on the seen families (A1) and the unseen families (the listed rules), '
            'and overall accuracy of A1 (percent). The values of every rule on both family groups and on v1 test are in the released run outputs.',
            'NegPairs-Fresh: görülen ailelerde (A1) ve görülmeyen ailelerde (listelenen kurallar) kapsam duyarlılığı SS (yüzde puan; on tohum üzerinden tohum ortalamaları, M0 bir kez puanlandı) '
            've A1\'in genel doğruluğu (yüzde). Her kuralın iki aile grubundaki ve v1 testindeki değerleri yayımlanan koşu çıktılarındadır.')
    return cap, pipe(head, rows, 'lrrrrrrrr', [2, 1, 1, 1, 1, 1, 1, 1, 1])


# ------------------------------------------------------------------------------------------- Perm main table
def t_perm(lang):
    head = T(lang, ['Encoder', 'A1 SS_bal', 'A1 SS, first', 'A1 SS, second', 'A1 joint acc', 'A4 SS_bal', 'A6 SS_bal', 'D1 SS_bal', 'D1 joint acc'],
             ['Kodlayıcı', 'A1 SS_bal', 'A1 SS, önce', 'A1 SS, sonra', 'A1 ortak doğr.', 'A4 SS_bal', 'A6 SS_bal', 'D1 SS_bal', 'D1 ortak doğr.'])
    rows = []
    for e in ENCS:
        rows.append([ENC_NAME[e] + (T(lang, ' (float64)', ' (float64)') if e in STATIC else ''),
                     F(perm_key(e, 'A1', 'seen.SS_bal'), lang), F(perm_key(e, 'A1', 'seen.first.SS'), lang), F(perm_key(e, 'A1', 'seen.second.SS'), lang),
                     F(perm_key(e, 'A1', 'seen.pooled.joint_NS_acc'), lang), F(perm_key(e, 'A4', 'seen.SS_bal'), lang), F(perm_key(e, 'A6', 'seen.SS_bal'), lang),
                     F(perm_key(e, 'D1', 'seen.SS_bal'), lang), F(perm_key(e, 'D1', 'seen.pooled.joint_NS_acc'), lang)])
    rows.append(['C1', F('sm.perm.C1.C1.seen.SS_bal', lang), F('sm.perm.C1.C1.seen.first.SS', lang), F('sm.perm.C1.C1.seen.second.SS', lang),
                 F('sm.perm.C1.C1.seen.pooled.joint_NS_acc', lang), '–', '–', '–', '–'])
    cap = T(lang,
            'NegPairs-Perm, seen families (' + '{{ref:tab:perm_n}}' + ' scenarios per clause order): order-balanced permutation scope sensitivity SS_bal, SS with the decisive clause first and second, '
            'and joint (N, S) accuracy (percent), for the maximum (A1) and the signed maximum with resolver scope (A4), a sentence-wide mask (A6) and dataset scope (D1). '
            'Seed means over ten seeds; the values of the static encoders E0 and E3 are computed from float64 scores. Every unmasked rule of E0 and E3 is at exactly 0 (Section 7.2).',
            'NegPairs-Perm, görülen aileler (yan tümce sırası başına ' + '{{ref:tab:perm_n}}' + ' senaryo): sıra dengeli permütasyon kapsam duyarlılığı SS_bal, belirleyici yan tümce önce ve sonra iken SS '
            've ortak (N, S) doğruluğu (yüzde); maksimum (A1) ile çözümleyici kapsamlı (A4), cümle çapında maskeli (A6) ve veri seti kapsamlı (D1) işaretli maksimum için. '
            'On tohum üzerinden tohum ortalamaları; statik kodlayıcılar E0 ve E3\'ün değerleri float64 puanlardan hesaplanmıştır. E0 ve E3\'ün maskesiz her kuralı tam olarak 0\'dadır (Bölüm 7.2).')
    cap = cap.replace('{{ref:tab:perm_n}}', F('subset.perm_seen_first', lang))
    return cap, pipe(head, rows, 'lrrrrrrrr', [2, 1, 1, 1, 1, 1, 1, 1, 1])


# ------------------------------------------------------------------------------------------- NevIR
def t_nevir(lang):
    head = T(lang, ['Scorer', 'All (n = ' + F('nevir.instances', lang) + ')', '95% interval', 'Nested (n = ' + F('nevir.nested', lang) + ')', 'Complement (n = ' + F('nevir.complement', lang) + ')', 'Per-query, all'],
             ['Puanlayıcı', 'Tümü (n = ' + F('nevir.instances', lang) + ')', '%95 aralık', 'İç içe (n = ' + F('nevir.nested', lang) + ')', 'Tümleyen (n = ' + F('nevir.complement', lang) + ')', 'Sorgu başına, tümü'])
    rows = []
    for e in ['E1', 'E5', 'E6', 'E4', 'E2', 'E3', 'E0']:
        rows.append([T(lang, f'M0 with {ENC_NAME[e]}', f'M0, {ENC_NAME[e]}'), F(f'nevir.{e}.M0.all.paired', lang), ci(f'nevir.{e}.M0.all', lang),
                     F(f'nevir.{e}.M0.nested.paired', lang), F(f'nevir.{e}.M0.complement.paired', lang), F(f'nevir.{e}.M0.all.per_query', lang)])
    rows.append([T(lang, 'M0-native (ColBERTv2 scoring)', 'M0-native (ColBERTv2 puanlaması)'), F('nevir.E6.M0n.all.paired', lang), ci('nevir.E6.M0n.all', lang),
                 F('nevir.E6.M0n.nested.paired', lang), F('nevir.E6.M0n.complement.paired', lang), F('nevir.E6.M0n.all.per_query', lang)])
    for e in STATIC:
        for a in ('A1', 'A3', 'M1'):
            rows.append([T(lang, f'{a} with {ENC_NAME[e]} (ten seeds)', f'{a}, {ENC_NAME[e]} (on tohum)'),
                         T(lang, f'{F(f"nevir.{e}.{a}.all.paired.min", lang)} to {F(f"nevir.{e}.{a}.all.paired.max", lang)}', f'{F(f"nevir.{e}.{a}.all.paired.min", lang)} ile {F(f"nevir.{e}.{a}.all.paired.max", lang)}'),
                         '–', F('nevir.static_trained.nested.max', lang), '–', '–'])
    rows.append([T(lang, 'Random ranking', 'Rastgele sıralama'), F('nevir.published.random', lang), '–', F('nevir.published.random', lang), F('nevir.published.random', lang), '–'])
    rows.append([T(lang, 'ColBERTv2 as published ([-@weller2024])', 'Yayımlandığı biçimiyle ColBERTv2 ([-@weller2024])'), F('nevir.published.ColBERTv2', lang), '–', '–', '–', '–'])
    cap = T(lang,
            'NevIR test split: paired accuracy (percent) on all instances, on the nested subset (documents nested as bags of tokens) and on its complement, with instance-level '
            'bootstrap intervals for the zero-shot scorers; per-query accuracy on all instances. Static scorers use float64 scores with a margin of ' + F('cfg.nevir_margin', lang) + '. '
            'For the trained static scorers the first column gives the range over ten seeds. Per-query accuracy counts a tie as an error, which is why the static scorers fall below 50. '
            'The published ColBERTv2 value was obtained with the benchmark\'s own evaluation code and configuration and is given for orientation only.',
            'NevIR test bölümü: tüm örneklerde, iç içe alt kümede (belgeleri token torbası olarak iç içe) ve tümleyeninde eşleştirilmiş doğruluk (yüzde); sıfır atışlı puanlayıcılar için '
            'örnek düzeyinde bootstrap aralıkları; tüm örneklerde sorgu başına doğruluk. Statik puanlayıcılar ' + F('cfg.nevir_margin', lang) + ' paylı float64 puanlar kullanır. '
            'Eğitilen statik puanlayıcılar için ilk sütun on tohum üzerindeki aralığı verir. Sorgu başına doğruluk eşitliği hata sayar; statik puanlayıcıların 50\'nin altına düşmesinin nedeni budur. '
            'Yayımlanmış ColBERTv2 değeri ölçütün kendi değerlendirme kodu ve yapılandırmasıyla elde edilmiştir ve yalnızca yönlendirme içindir.')
    return cap, pipe(head, rows, 'lrlrrr', [5, 2, 3, 2, 3, 3])


# ------------------------------------------------------------------------------------------- probing
def t_probing(lang):
    head = T(lang, ['Encoder', 'CV accuracy (SD)', 'Transfer', 'Syntactic: negation', 'Syntactic: twins', 'Selectivity', 'A5 gate AUC'],
             ['Kodlayıcı', 'ÇD doğruluğu (SS)', 'Aktarım', 'Sözdizimsel: olumsuzluk', 'Sözdizimsel: ikizler', 'Seçicilik', 'A5 geçit AUC'])
    rows = []
    for e in ['E2', 'E1', 'E5', 'E4', 'E6', 'E0', 'E3']:
        tw = F(f'probe.{e}.twins_cv', lang) if f'probe.{e}.twins_cv' in REG else '–'
        sel = F(f'probe.{e}.selectivity', lang, '+') if f'probe.{e}.selectivity' in REG else '–'
        syn = F(f'probe.{e}.syn_cv', lang) if f'probe.{e}.syn_cv' in REG else '–'
        auc = F(f'probe.{e}.auc_mean', lang) if f'probe.{e}.auc_mean' in REG else '–'
        rows.append([ENC_NAME[e], f'{F(f"probe.{e}.cv_mean", lang)} ({F(f"probe.{e}.cv_sd", lang)})', F(f'probe.{e}.transfer', lang), syn, tw, sel, auc])
    cap = T(lang,
            'Probing the clause of the edit in the frozen token states (percent; chance 50): accuracy of five-fold cross-validation (CV) grouped by scenario on the seen families (' + F('probe.E1.n_words_seen', lang) + ' words), with its standard deviation (SD) over folds, '
            'transfer from the seen to the unseen families (' + F('probe.E1.n_words_unseen', lang) + ' words), and on the syntactic scenarios (' + F('probe.E1.syn_n_words', lang) + ' words) the negation probe against the neutral-twin probe '
            '(*also* inserted instead of *not*); selectivity is their difference. The last column is the mean area under the ROC curve of the A5 gate value for the same words and labels (ten seeds).',
            'Dondurulmuş token durumlarında düzenlemenin yan tümcesini sondalama (yüzde; şans 50): görülen ailelerde senaryoya göre gruplanmış beş katlı çapraz doğrulama (ÇD) doğruluğu (' + F('probe.E1.n_words_seen', lang) + ' sözcük) ve katlar üzerindeki standart sapması (SS), '
            'görülen ailelerden görülmeyenlere aktarım (' + F('probe.E1.n_words_unseen', lang) + ' sözcük) ve sözdizimsel senaryolarda (' + F('probe.E1.syn_n_words', lang) + ' sözcük) olumsuzluk sondası ile nötr ikiz sondası '
            '(*not* yerine *also* eklenmiş); seçicilik ikisinin farkıdır. Son sütun aynı sözcükler ve etiketler için A5 geçit değerinin ROC eğrisi altındaki alanının ortalamasıdır (on tohum).')
    return cap, pipe(head, rows, 'lrrrrrr', [4, 4, 2, 2, 2, 3, 2])


# ------------------------------------------------------------------------------------------- CondaQA (appendix B)
def t_condaqa(lang):
    head = T(lang, ['System', 'Setting', 'Test accuracy (%)', 'Source'], ['Sistem', 'Ayar', 'Test doğruluğu (%)', 'Kaynak'])
    src_pub = T(lang, '[-@ravichander2022], their Table 4', '[-@ravichander2022], kendi Tablo 4\'ü')
    here = T(lang, 'this study', 'bu çalışma')
    rows = [
        [T(lang, 'Majority label', 'Çoğunluk etiketi'), T(lang, 'all test questions', 'tüm test soruları'), F('v2.lit.cq.majority', lang), src_pub],
        ['BERT-large', T(lang, 'fully fine-tuned', 'tam ince ayar'), F('v2.lit.cq.bert_large', lang), src_pub],
        ['RoBERTa-large', T(lang, 'fully fine-tuned', 'tam ince ayar'), F('v2.lit.cq.roberta_large', lang), src_pub],
        ['DeBERTa-v3-large', T(lang, 'fully fine-tuned', 'tam ince ayar'), F('v2.lit.cq.debertav3_large', lang), src_pub],
        [T(lang, 'Majority label (NO)', 'Çoğunluk etiketi (NO)'), T(lang, f'{F("v2.cq.test.eligible", lang)} eligible rows', f'{F("v2.cq.test.eligible", lang)} uygun satır'), F('v2.cq.label.NO', lang), here],
        [T(lang, f'CACSA modules and C0 ({F("v2.cq.v11.n_conditions", lang)} conditions)', f'CACSA modülleri ve C0 ({F("v2.cq.v11.n_conditions", lang)} koşul)'), T(lang, 'frozen encoders, seed means', 'dondurulmuş kodlayıcılar, tohum ortalamaları'),
         F('v2.cq.v11.acc_min', lang) + T(lang, ' to ', ' ile ') + F('v2.cq.v11.acc_max', lang), here],
        ['C1 DeBERTa-v3-base', T(lang, 'fully fine-tuned (float32 re-run)', 'tam ince ayar (float32 yeniden koşu)'), F('v2.cq.fu.C1.acc', lang), here],
    ]
    cap = T(lang, 'CondaQA: published reference values and the results of the v1.1 grid. The published values are reproduced as printed; the eligible rows of this study exclude the free-text answers.',
            'CondaQA: yayımlanmış başvuru değerleri ve v1.1 ızgarasının sonuçları. Yayımlanmış değerler basıldığı biçimde aktarılmıştır; bu çalışmadaki uygun satırlar serbest metin yanıtları içermez.')
    return cap, pipe(head, rows, 'llrl', [3, 3, 2, 3])


# ------------------------------------------------------------------------------------------- v2.0 outcomes (appendix C)
def t_v2outcomes(lang):
    head = T(lang, ['Rule (manifest v2.0)', 'Encoder', 'Value', 'Seed interval (95%)', 'Positive seeds', 'Outcome'],
             ['Kural (manifest v2.0)', 'Kodlayıcı', 'Değer', 'Tohum aralığı (%95)', 'Pozitif tohum', 'Sonuç'])
    rows = []
    for i, e in enumerate(['E1', 'E0', 'E2']):
        rows.append([T(lang, 'G: A1 test SS, seen families (≥ 20)', 'G: A1 test SS, görülen aileler (≥ 20)') if i == 0 else '', e, F(f'v2.G.{e}.mean', lang),
                     T(lang, f'seeds {F(f"v2.G.{e}.min", lang)} to {F(f"v2.G.{e}.max", lang)}', f'tohumlar {F(f"v2.G.{e}.min", lang)} ile {F(f"v2.G.{e}.max", lang)}'), '–',
                     T(lang, 'passed', 'geçti') if val(f'v2.G.{e}.passed') else T(lang, 'not passed', 'geçemedi')])
    rows.append([T(lang, 'R1: A1 unseen SS (≤ 5)', 'R1: A1 görülmeyen SS (≤ 5)'), 'E0', F('v2.R1.E0.mean', lang), '–', '–', T(lang, 'not evaluated (gate)', 'değerlendirilmedi (geçit)')])
    rows.append([T(lang, 'R2: A4 − A1 unseen SS (≥ 15)', 'R2: A4 − A1 görülmeyen SS (≥ 15)'), 'E1', F('v2.R2.E1.mean', lang, '+'), ci('v2.R2.E1', lang), f'{F("v2.R2.E1.pos", lang)}/{F("v2.R2.E1.n", lang)}', yes_no(val('v2.R2.E1.met'), lang)])
    rows.append([T(lang, 'R3: A6 − A4 flip_O→S, S-same (≥ 10)', 'R3: A6 − A4 flip_O→S, S-aynı (≥ 10)'), 'E1', F('v2.R3.E1.mean', lang, '+'), ci('v2.R3.E1', lang), f'{F("v2.R3.E1.pos", lang)}/{F("v2.R3.E1.n", lang)}', yes_no(val('v2.R3.E1.met'), lang)])
    rows.append([T(lang, 'R5: Proposition 1 of manifest v2.0 (Proposition B here), insertion pairs (exact)', 'R5: manifest v2.0\'ın Önerme 1\'i (burada Önerme B), ekleme çiftleri (tam)'), 'E0', T(lang, f'{F("v2.R5.violations_total", lang)} violations', f'{F("v2.R5.violations_total", lang)} ihlal'), '–', '–', yes_no(val('v2.R5.met'), lang)])
    rows.append([T(lang, 'D1 − A1 unseen SS (descriptive)', 'D1 − A1 görülmeyen SS (betimsel)'), 'E1', F('v2.D1.E1.D1mA1.unseen.mean', lang, '+'), ci('v2.D1.E1.D1mA1.unseen', lang), f'{F("v2.D1.E1.D1mA1.unseen.pos", lang)}/{F("v2.D1.E1.D1mA1.unseen.n", lang)}', '–'])
    rows.append([T(lang, 'C1: seen / unseen SS, accuracy (reference)', 'C1: görülen / görülmeyen SS, doğruluk (başvuru)'), 'C1',
                 f'{F("v2.sm.C1.C1.seen.SS", lang)} / {F("v2.sm.C1.C1.unseen.SS", lang)}; {F("v2.sm.C1.C1.all.acc", lang)}%' if lang == 'en' else f'{F("v2.sm.C1.C1.seen.SS", lang)} / {F("v2.sm.C1.C1.unseen.SS", lang)}; %{F("v2.sm.C1.C1.all.acc", lang)}', '–', '–', '–'])
    cap = T(lang, 'Outcomes of the v2.0 confirmatory run on the NegPairs v1 test split (run np_main_20261004T154358Z; three encoders, five seeds; percentage points). '
                  'The gate was computed on the test split; R2 and R3 required a positive contrast in all five seeds and a 95% seed interval above zero.',
            'v2.0 teyit koşusunun NegPairs v1 test bölümündeki sonuçları (koşu np_main_20261004T154358Z; üç kodlayıcı, beş tohum; yüzde puan). '
            'Geçit test bölümünde hesaplanmıştı; R2 ve R3 beş tohumun hepsinde pozitif bir karşıtlık ve sıfırın üstünde %95 tohum aralığı gerektiriyordu.')
    return cap, pipe(head, rows, 'llrlrl', [6, 3, 3, 3, 3, 3])


# ------------------------------------------------------------------------------------------- build funnel (appendix D)
def t_funnel(lang):
    head = T(lang, ['Set', 'Stratum', 'Candidates', 'Filters', 'Near-dup.', 'Verifier 1', 'Verifier 2', 'Other', 'Accepted', 'Cap'],
             ['Set', 'Katman', 'Aday', 'Süzgeç', 'Yakın kopya', 'Doğrulayıcı 1', 'Doğrulayıcı 2', 'Diğer', 'Kabul', 'Üst sınır'])
    rows = []
    for fam in ('syntactic', 'quantifier', 'affixal', 'lexical'):
        for order in ('first', 'second'):
            k = f'build.perm.stages.{fam}.{order}'
            g = lambda st: F(f'{k}.{st}', lang) if f'{k}.{st}' in REG else '0'
            rows.append(['NegPairs-Perm', T(lang, f'{fam}, decisive {order}', f'{fam}, belirleyici {"önce" if order == "first" else "sonra"}'), g('candidates'), g('filter'), g('duplicate'), g('verifier'), g('qwen'), g('other'), g('accepted'), F('build.perm.cap_per_stratum', lang)])
    rows.append(['NegPairs-Perm', T(lang, 'total', 'toplam'), F('build.perm.candidates', lang), F('build.perm.stage.filter', lang), F('build.perm.stage.duplicate', lang), F('build.perm.stage.verifier', lang), F('build.perm.stage.qwen', lang),
                 F('build.perm.stage.other', lang), F('build.perm.stage.accepted', lang), F('build.perm.cap_total', lang)])
    for fam in ('syntactic', 'quantifier', 'affixal', 'lexical'):
        k = f'build.fresh.stages.{fam}'
        g = lambda st: F(f'{k}.{st}', lang) if f'{k}.{st}' in REG else '0'
        rows.append(['NegPairs-Fresh', fam, g('candidates'), g('filter'), g('duplicate'), g('verifier'), g('qwen'), g('other'), g('accepted'), F('build.fresh.cap_per_family', lang)])
    rows.append(['NegPairs-Fresh', T(lang, 'total', 'toplam'), F('build.fresh.candidates', lang), F('build.fresh.stage.filter', lang), F('build.fresh.stage.duplicate', lang), F('build.fresh.stage.verifier', lang), F('build.fresh.stage.qwen', lang), F('build.fresh.stage.other', lang), F('build.fresh.stage.accepted', lang), F('build.fresh.cap_total', lang)])
    cap = T(lang, 'Construction funnel of the v3.0 sets: processed candidates per stratum and the stage at which each was discarded, in pipeline order (deterministic filters; near-duplicate check; first verifier `claude-haiku-4-5-20251001`; '
                  'second verifier Qwen2.5-7B-Instruct; other = generator failure or stratum cap reached at acceptance). The Perm phase stopped at its budget cap; the Fresh phase reached every family cap.',
            'v3.0 setlerinin yapım hunisi: katman başına işlenen adaylar ve her birinin elendiği aşama, boru hattı sırasıyla (belirlenimci süzgeçler; yakın kopya denetimi; birinci doğrulayıcı `claude-haiku-4-5-20251001`; '
            'ikinci doğrulayıcı Qwen2.5-7B-Instruct; diğer = üretici hatası veya kabul anında dolan katman üst sınırı). Perm evresi bütçe üst sınırında durdu; Fresh evresi her aile üst sınırına ulaştı.')
    return cap, pipe(head, rows, 'llrrrrrrrr', [7, 8, 5, 4, 4, 4, 4, 3, 4, 2])


def t_discard(lang):
    head = T(lang, ['Set', 'Reason (as recorded)', 'Candidates'], ['Set', 'Gerekçe (kaydedildiği biçimde)', 'Aday'])
    rows = []
    for setname, label in (('perm', 'NegPairs-Perm'), ('fresh', 'NegPairs-Fresh')):
        for i in range(8):
            rows.append([label if i == 0 else '', '`' + val(f'build.{setname}.discard.{i}.reason').replace('|', '\\|') + '`', F(f'build.{setname}.discard.{i}.n', lang)])
    cap = T(lang, 'The eight most frequent discard reasons per set, as recorded by the build (verifier verdicts name the arm and the option letter or verdict; numbered reasons are the deterministic checks).',
            'Set başına en sık sekiz eleme gerekçesi, yapımda kaydedildiği biçimde (doğrulayıcı kararları kolu ve seçenek harfini veya kararı adlandırır; numaralı gerekçeler belirlenimci denetimlerdir).')
    return cap, pipe(head, rows, 'llr', [2, 5, 2])


# ------------------------------------------------------------------------------------------- appendix E tables
def t_allfresh(lang):
    head = T(lang, ['Encoder', 'Rule', 'Fresh seen SS', 'Fresh unseen SS', 'Fresh unseen flip_O→N', 'Fresh unseen flip_O→S', 'Fresh acc', 'v1 test seen SS', 'v1 test unseen SS'],
             ['Kodlayıcı', 'Kural', 'Fresh görülen SS', 'Fresh görülmeyen SS', 'Fresh görülmeyen flip_O→N', 'Fresh görülmeyen flip_O→S', 'Fresh doğr.', 'v1 test görülen SS', 'v1 test görülmeyen SS'])
    rows = []
    for e in ENCS:
        for i, a in enumerate(RULES_ALL + (['M0n'] if e == 'E6' else [])):
            rows.append([ENC_NAME[e] if i == 0 else '', a if a != 'M0n' else 'M0-native', F(f'sm.fresh.{e}.{a}.seen.SS', lang), F(f'sm.fresh.{e}.{a}.unseen.SS', lang), F(f'sm.fresh.{e}.{a}.unseen.flip_O_N', lang),
                         F(f'sm.fresh.{e}.{a}.unseen.flip_O_S', lang), F(f'sm.fresh.{e}.{a}.all.acc', lang), F(f'sm.v1.{e}.{a}.seen.SS', lang), F(f'sm.v1.{e}.{a}.unseen.SS', lang)])
    rows.append(['C1', 'C1', F('sm.fresh.C1.C1.seen.SS', lang), F('sm.fresh.C1.C1.unseen.SS', lang), F('sm.fresh.C1.C1.unseen.flip_O_N', lang), F('sm.fresh.C1.C1.unseen.flip_O_S', lang),
                 F('sm.fresh.C1.C1.all.acc', lang), F('sm.v1.C1.C1.seen.SS', lang), F('sm.v1.C1.C1.unseen.SS', lang)])
    cap = T(lang, 'All conditions on NegPairs-Fresh and on the v1 test split (seed means over ten seeds; M0 and M0-native scored once): scope sensitivity SS (percentage points) by family group, unseen-family decision changes (percent of scenarios) and overall accuracy (percent).',
            'NegPairs-Fresh ve v1 test bölümündeki tüm koşullar (on tohum üzerinden tohum ortalamaları; M0 ve M0-native bir kez puanlandı): aile grubuna göre kapsam duyarlılığı SS (yüzde puan), görülmeyen ailelerde karar değişiklikleri (senaryoların yüzdesi) ve genel doğruluk (yüzde).')
    return cap, pipe(head, rows, 'llrrrrrrr', [2, 1, 1, 1, 1, 1, 1, 1, 1])


def t_allperm(lang):
    head = T(lang, ['Encoder', 'Rule', 'Seen SS_bal', 'Seen first SS', 'Seen second SS', 'Seen joint acc', 'Unseen SS_bal', 'Unseen joint acc', 'flip_O→P'],
             ['Kodlayıcı', 'Kural', 'Görülen SS_bal', 'Görülen önce SS', 'Görülen sonra SS', 'Görülen ortak doğr.', 'Görülmeyen SS_bal', 'Görülmeyen ortak doğr.', 'flip_O→P'])
    rows = []
    for e in ENCS:
        for i, a in enumerate(RULES_ALL + (['M0n'] if e == 'E6' else [])):
            rows.append([(ENC_NAME[e] + (' (float64)' if e in STATIC else '')) if i == 0 else '', a if a != 'M0n' else 'M0-native', F(perm_key(e, a, 'seen.SS_bal'), lang), F(perm_key(e, a, 'seen.first.SS'), lang), F(perm_key(e, a, 'seen.second.SS'), lang),
                         F(perm_key(e, a, 'seen.pooled.joint_NS_acc'), lang), F(perm_key(e, a, 'unseen.SS_bal'), lang), F(perm_key(e, a, 'unseen.pooled.joint_NS_acc'), lang), F(perm_key(e, a, 'all.pooled.flip_O_P'), lang)])
    rows.append(['C1', 'C1', F('sm.perm.C1.C1.seen.SS_bal', lang), F('sm.perm.C1.C1.seen.first.SS', lang), F('sm.perm.C1.C1.seen.second.SS', lang), F('sm.perm.C1.C1.seen.pooled.joint_NS_acc', lang),
                 F('sm.perm.C1.C1.unseen.SS_bal', lang), F('sm.perm.C1.C1.unseen.pooled.joint_NS_acc', lang), F('sm.perm.C1.C1.all.pooled.flip_O_P', lang)])
    cap = T(lang, 'All conditions on NegPairs-Perm (seed means over ten seeds; M0 and M0-native scored once; E0 and E3 from float64 scores): order-balanced SS and SS by clause order on the seen families, joint (N, S) accuracy (percent) on the seen and unseen families, and decision changes under paraphrase (percent of scenarios).',
            'NegPairs-Perm üzerindeki tüm koşullar (on tohum üzerinden tohum ortalamaları; M0 ve M0-native bir kez puanlandı; E0 ve E3 float64 puanlardan): görülen ailelerde sıra dengeli SS ve yan tümce sırasına göre SS, görülen ve görülmeyen ailelerde ortak (N, S) doğruluğu (yüzde) ve yeniden ifadede karar değişiklikleri (senaryoların yüzdesi).')
    return cap, pipe(head, rows, 'llrrrrrrr', [2, 1, 1, 1, 1, 1, 1, 1, 1])


def t_family(lang):
    head = T(lang, ['Encoder', 'Rule', 'Syntactic', 'Quantifier', 'Affixal', 'Lexical'], ['Kodlayıcı', 'Kural', 'Sözdizimsel', 'Niceleyici', 'Ekli', 'Sözcüksel'])
    rows = []
    for e in ENCS:
        for i, a in enumerate(('A1', 'A4', 'D1')):
            rows.append([(ENC_NAME[e] + (' (float64)' if e in STATIC else '')) if i == 0 else '', a] + [F(perm_key(e, a, f'family.{f}.SS_bal'), lang) for f in ('syntactic', 'quantifier', 'affixal', 'lexical')])
    rows.append(['C1', 'C1'] + [F(f'sm.perm.C1.C1.family.{f}.SS_bal', lang) for f in ('syntactic', 'quantifier', 'affixal', 'lexical')])
    cap = T(lang, 'NegPairs-Perm: order-balanced permutation scope sensitivity per family (percentage points; seed means) for A1, A4 and D1. The affixal family has ' + F('build.perm.affixal_total', lang) + ' scenarios (' + F('build.perm.accepted.affixal.first', lang) + ' with the decisive clause first) and the lexical family a single cue, so these two columns are descriptive.',
            'NegPairs-Perm: aile başına sıra dengeli permütasyon kapsam duyarlılığı (yüzde puan; tohum ortalamaları), A1, A4 ve D1 için. Ekli ailede ' + F('build.perm.affixal_total', lang) + ' senaryo (' + F('build.perm.accepted.affixal.first', lang) + '\'i belirleyici yan tümce önce) ve sözcüksel ailede tek belirteç vardır; bu iki sütun betimseldir.')
    return cap, pipe(head, rows, 'llrrrr', [2, 1, 1, 1, 1, 1])


def t_seeds(lang):
    seeds = [val(f'seed.{i}') for i in range(1, 11)]
    head = T(lang, ['Rule', 'Encoder'] + [F(f'seed.{i}', lang) for i in range(1, 11)], ['Kural', 'Kodlayıcı'] + [F(f'seed.{i}', lang) for i in range(1, 11)])
    rows = []
    for rule, encs in (('R2', ['E0', 'E6', 'E2', 'E5', 'E4', 'E1', 'E3']), ('R3', ['E2', 'E6', 'E0', 'E5', 'E4', 'E1', 'E3']), ('R8', ['E4', 'E6', 'E5', 'E1', 'E2'])):
        for i, e in enumerate(encs):
            rows.append([rule if i == 0 else '', e] + [F(f'{rule}.{e}.s{s}', lang, '+' if rule != 'R8' else '') for s in seeds])
    cap = T(lang, 'Seed-level values of the rule quantities (percentage points): R2 (A4 − A1 unseen-family SS on Fresh), R3 (A6 − A4 flip_O→S on the S-same Fresh scenarios) and R8 (A1 order-balanced SS on the seen Perm families), by seed. Seeds are listed in manifest order.',
            'Kural niceliklerinin tohum düzeyindeki değerleri (yüzde puan): R2 (Fresh\'te A4 − A1 görülmeyen aile SS), R3 (S-aynı Fresh senaryolarında A6 − A4 flip_O→S) ve R8 (görülen Perm ailelerinde A1 sıra dengeli SS), tohuma göre. Tohumlar manifest sırasıyla verilmiştir.')
    return cap, pipe(head, rows, 'll' + 'r' * 10, [2, 3] + [2] * 10)


def t_lr(lang):
    head = T(lang, ['Encoder', 'Dev accuracy at 3 × 10⁻⁴', 'at 10⁻³', 'at 3 × 10⁻³', 'Chosen'], ['Kodlayıcı', '3 × 10⁻⁴\'te geliştirme doğruluğu', '10⁻³\'te', '3 × 10⁻³\'te', 'Seçilen'])
    rows = []
    for e in ENCS:
        rows.append([ENC_NAME[e], F(f'lr.{e}.acc.3e-4', lang), F(f'lr.{e}.acc.1e-3', lang), F(f'lr.{e}.acc.3e-3', lang), F(f'lr.{e}.chosen', lang)])
    cap = T(lang, 'Learning-rate grid of the pilot (A1, seeds 13, 42 and 71; mean dev accuracy in percent over all dev items; ties would go to the smaller rate). The chosen rate applied to every trained condition of the encoder.',
            'Pilotun öğrenme oranı ızgarası (A1, tohumlar 13, 42 ve 71; tüm geliştirme öğeleri üzerinde ortalama geliştirme doğruluğu, yüzde; eşitlikte küçük oran seçilirdi). Seçilen oran kodlayıcının eğitilen her koşuluna uygulandı.')
    return cap, pipe(head, rows, 'lrrrr', [2, 2, 1, 1, 1])


def t_resolver(lang):
    head = T(lang, ['Set', 'Families', 'Arm', 'n', 'Cue detected', 'Decisive in scope', 'Scope in decisive clause', 'Scope in other clause'],
             ['Set', 'Aileler', 'Kol', 'n', 'Belirteç bulundu', 'Belirleyici kapsamda', 'Kapsam belirleyici yan tümcede', 'Kapsam öteki yan tümcede'])
    rows = []
    for name, label in (('fresh', 'NegPairs-Fresh'), ('perm', 'NegPairs-Perm')):
        for g, gl in (('seen', T(lang, 'seen', 'görülen')), ('unseen', T(lang, 'unseen', 'görülmeyen'))):
            for arm in ('N', 'S'):
                k = f'res.{name}.{g}.{arm}'
                dec = f'{F(f"{k}.decisive_in_scope.pct", lang)}% ({F(f"{k}.decisive_in_scope.num", lang)}/{F(f"{k}.decisive_in_scope.den", lang)})' if lang == 'en' else f'%{F(f"{k}.decisive_in_scope.pct", lang)} ({F(f"{k}.decisive_in_scope.num", lang)}/{F(f"{k}.decisive_in_scope.den", lang)})'
                cue = F(f'{k}.cue_detected.pct', lang) + ('%' if lang == 'en' else '')
                if lang == 'tr':
                    cue = '%' + F(f'{k}.cue_detected.pct', lang)
                sdc = (F(f'{k}.scope_in_decisive_clause.pct', lang) + ('%' if lang == 'en' else '')) if f'{k}.scope_in_decisive_clause.pct' in REG else '–'
                soc = (F(f'{k}.scope_in_other_clause.pct', lang) + ('%' if lang == 'en' else '')) if f'{k}.scope_in_other_clause.pct' in REG else '–'
                if lang == 'tr':
                    sdc = ('%' + F(f'{k}.scope_in_decisive_clause.pct', lang)) if f'{k}.scope_in_decisive_clause.pct' in REG else '–'
                    soc = ('%' + F(f'{k}.scope_in_other_clause.pct', lang)) if f'{k}.scope_in_other_clause.pct' in REG else '–'
                rows.append([label, gl, arm, F(f'{k}.n', lang), cue, dec, sdc, soc])
    cap = T(lang, 'Scope resolver diagnostics on the N and S arms (spaCy `' + val('res.spacy') + '`). "Decisive in scope" is computed on the items whose decisive words were located in the text; the clause columns exist for NegPairs-Perm, whose clause spans are stored.',
            'N ve S kollarında kapsam çözümleyicisi tanıları (spaCy `' + val('res.spacy') + '`). "Belirleyici kapsamda" belirleyici sözcükleri metinde bulunan öğeler üzerinde hesaplanır; yan tümce sütunları, yan tümce aralıkları kayıtlı olan NegPairs-Perm için vardır.')
    return cap, pipe(head, rows, 'lllrrrrr', [2, 1, 1, 1, 1, 2, 1, 1])


def t_qwen(lang):
    head = T(lang, ['Group', 'Agreement (all four arms, or the arm)', 'n', 'Share'], ['Grup', 'Uyuşma (dört kol veya ilgili kol)', 'n', 'Pay'])
    rows = []
    for k, en, tr in (('all', 'all scenarios', 'tüm senaryolar'), ('split.train', 'train split', 'eğitim bölümü'), ('split.dev', 'dev split', 'geliştirme bölümü'), ('split.test', 'test split', 'test bölümü'),
                      ('family.syntactic', 'syntactic family', 'sözdizimsel aile'), ('family.quantifier', 'quantifier family', 'niceleyici aile'), ('family.affixal', 'affixal family', 'ekli aile'), ('family.lexical', 'lexical family', 'sözcüksel aile'),
                      ('arm.O', 'arm O', 'O kolu'), ('arm.P', 'arm P', 'P kolu'), ('arm.N', 'arm N', 'N kolu'), ('arm.S', 'arm S', 'S kolu')):
        pct = F(f'qwen.v1.{k}.pct', lang) + '%' if lang == 'en' else '%' + F(f'qwen.v1.{k}.pct', lang)
        rows.append([T(lang, en, tr), F(f'qwen.v1.{k}.agree', lang), F(f'qwen.v1.{k}.n', lang), pct])
    rows2 = []
    for e in ENCS:
        rows2.append([ENC_NAME[e], F(f'qwenpass.{e}.A1.seen.SS', lang), F(f'sm.v1.{e}.A1.seen.SS', lang), F(f'qwenpass.{e}.A1.unseen.SS', lang), F(f'sm.v1.{e}.A1.unseen.SS', lang)])
    rows2.append(['C1', F('qwenpass.C1.C1.seen.SS', lang), F('sm.v1.C1.C1.seen.SS', lang), F('qwenpass.C1.C1.unseen.SS', lang), F('sm.v1.C1.C1.unseen.SS', lang)])
    head2 = T(lang, ['Encoder (A1)', 'Seen SS, passing', 'Seen SS, all', 'Unseen SS, passing', 'Unseen SS, all'], ['Kodlayıcı (A1)', 'Görülen SS, geçen', 'Görülen SS, tümü', 'Görülmeyen SS, geçen', 'Görülmeyen SS, tümü'])
    cap = T(lang, 'Qwen2.5-7B-Instruct audit of NegPairs v1. Part (a), this table: scenarios in which the verdict equals the expected option in all four arms (by split and family) and agreement per arm. Part (b), the table below: A1 scope sensitivity (percentage points; seed means) on the v1 test scenarios that pass the audit, next to the value on the whole test split.',
            'NegPairs v1\'in Qwen2.5-7B-Instruct denetimi. (a) kısmı, bu tablo: dört kolun hepsinde kararın beklenen seçeneğe eşit olduğu senaryolar (bölüm ve aileye göre) ve kol başına uyuşma. (b) kısmı, alttaki tablo: denetimi geçen v1 test senaryolarında A1 kapsam duyarlılığı (yüzde puan; tohum ortalamaları), test bölümünün tümündeki değerin yanında.')
    part_b = T(lang, '(b) A1 scope sensitivity on the v1 test scenarios that pass the audit and on the whole test split', '(b) Denetimi geçen v1 test senaryolarında ve test bölümünün tümünde A1 kapsam duyarlılığı')
    return cap, pipe(head, rows, 'lrrr', [3, 2, 1, 1]) + f'\n\n*{part_b}*\n\n' + pipe(head2, rows2, 'lrrrr', [2, 1, 1, 1, 1])


def t_acc(lang):
    head = T(lang, ['Encoder', 'A1 acc_OP', 'A1 acc_OP, prior-incongruent', 'C2 acc_OP', 'A1 − C2', 'Draft gate'], ['Kodlayıcı', 'A1 acc_OP', 'A1 acc_OP, önsel-uyumsuz', 'C2 acc_OP', 'A1 − C2', 'Taslak geçit'])
    rows = []
    for e in ENCS:
        rows.append([ENC_NAME[e], F(f'acc2.G.{e}.A1_acc_OP', lang), F(f'acc2.G.{e}.A1_incong', lang), F(f'acc2.G.{e}.C2_acc_OP', lang), F(f'acc2.G.{e}.margin', lang, '+'),
                     T(lang, 'passed', 'geçti') if val(f'acc2.G.{e}.passed') else T(lang, 'not passed', 'geçemedi')])
    cap = T(lang, 'Secondary accuracy view on NegPairs-Fresh (percent; seed means): accuracy on the affirmative arms O and P (acc_OP) of A1 over all scenarios and over the ' + F('strata.fresh.prior_incongruent', lang) + ' prior-incongruent scenarios (those whose correct option is not the option a context-blind reading favors), acc_OP of the context-blind control C2, the difference, and the draft gate of manifest v2.0 (acc_OP ≥ ' + F('thr2.gate_acc_OP', lang) + ' and prior-incongruent acc_OP ≥ ' + F('thr2.gate_incongruent_acc_OP', lang) + ').',
            'NegPairs-Fresh\'te ikincil doğruluk görünümü (yüzde; tohum ortalamaları): A1\'in olumlu kollar O ve P\'deki doğruluğu (acc_OP), tüm senaryolarda ve önsel-uyumsuz ' + F('strata.fresh.prior_incongruent', lang) + ' senaryoda (doğru seçeneği, bağlamı okumayan bir okumanın yeğlediği seçenek olmayan senaryolar), bağlamı okumayan kontrol C2\'nin acc_OP değeri, fark ve manifest v2.0\'ın taslak geçidi (acc_OP ≥ ' + F('thr2.gate_acc_OP', lang) + ' ve önsel-uyumsuz acc_OP ≥ ' + F('thr2.gate_incongruent_acc_OP', lang) + ').')
    return cap, pipe(head, rows, 'lrrrrl', [3, 2, 3, 2, 2, 2])


def t_f64gpu(lang):
    head = T(lang, ['Encoder', 'Rule', 'SS_bal float64', 'SS_bal GPU', 'Joint acc float64', 'Joint acc GPU'], ['Kodlayıcı', 'Kural', 'SS_bal float64', 'SS_bal GPU', 'Ortak doğr. float64', 'Ortak doğr. GPU'])
    rows = []
    for e in STATIC:
        for i, a in enumerate(('A1', 'A4', 'A6', 'D1')):
            rows.append([ENC_NAME[e] if i == 0 else '', a, F(f'f64.{e}.{a}.seen.SS_bal', lang), F(f'sm.perm.{e}.{a}.seen.SS_bal', lang), F(f'f64.{e}.{a}.seen.pooled.joint_NS_acc', lang), F(f'sm.perm.{e}.{a}.seen.pooled.joint_NS_acc', lang)])
    cap = T(lang, 'Float64 against bfloat16 GPU values for the static encoders on the seen families of NegPairs-Perm (seed means). The GPU decisions differed from the float64 decisions in ' + F('f64.item_disagreements', lang) + ' of ' + F('n.f64_item_predictions', lang) + ' item predictions over all conditions, at near-ties; ' + F('f64.n_seed_mean_differences', lang) + ' seed means differed, by at most ' + F('f64.max_abs_seed_mean_diff', lang) + ' points. The article reports the float64 values.',
            'Statik kodlayıcılar için NegPairs-Perm\'in görülen ailelerinde float64 ile bfloat16 GPU değerleri (tohum ortalamaları). GPU kararları tüm koşullardaki ' + F('n.f64_item_predictions', lang) + ' öğe tahmininin ' + F('f64.item_disagreements', lang) + '\'sinde, neredeyse eşit puanlarda float64 kararlarından ayrıldı; ' + F('f64.n_seed_mean_differences', lang) + ' tohum ortalaması en çok ' + F('f64.max_abs_seed_mean_diff', lang) + ' puan farkla değişti. Makale float64 değerlerini verir.')
    return cap, pipe(head, rows, 'llrrrr', [2, 1, 1, 1, 1, 1])


def t_params(lang):
    dims = ['*d* = ' + F('enc.E6.dim', lang) + ' (E6)', '*d* = ' + F('enc.E3.dim', lang) + ' (E3)', '*d* = ' + F('enc.E1.dim', lang) + ' (E0, E1)', '*d* = ' + F('enc.E2.dim', lang) + ' (E2, E4, E5)']
    head = T(lang, ['Rule'] + dims, ['Kural'] + dims)
    rows = []
    for a in ('A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'D1', 'M1', 'C0', 'C2'):
        rows.append([a, F(f'param.E6.{a}', lang), F(f'param.E3.{a}', lang), F(f'param.E1.{a}', lang), F(f'param.E2.{a}', lang)])
    cap = T(lang, 'Trainable parameter counts per aggregation rule and state dimension (M0 and M0-native have none; C1 has ' + F('param.C1.C1', lang) + ').',
            'Puanlama kuralı ve durum boyutu başına eğitilebilir parametre sayıları (M0 ve M0-native parametresizdir; C1\'in ' + F('param.C1.C1', lang) + ' parametresi vardır).')
    return cap, pipe(head, rows, 'lrrrr', [1, 1, 1, 1, 1])


TABLES = {'examplefresh': t_examplefresh, 'exampleperm': t_exampleperm, 'aggs': t_aggs, 'encoders': t_encoders, 'data': t_data, 'rules': t_rules,
          'fresh': t_fresh, 'perm': t_perm, 'nevir': t_nevir, 'probing': t_probing, 'condaqa': t_condaqa, 'v2outcomes': t_v2outcomes,
          'funnel': t_funnel, 'discard': t_discard, 'allfresh': t_allfresh, 'allperm': t_allperm, 'family': t_family, 'seeds': t_seeds,
          'lr': t_lr, 'resolver': t_resolver, 'qwen': t_qwen, 'acc': t_acc, 'f64gpu': t_f64gpu, 'params': t_params}


# ------------------------------------------------------------------------------------------- released systems (revision step 1)
PUB_ORDER = ['colbertv2', 'answerai', 'gtemc', 'bgem3c', 'bgem3d', 'bgem3s', 'bgebase', 'e5', 'mpnet', 'gtemb', 'splade', 'minilm', 'bgerr', 'bgerr2', 'mxbai', 'monot5b', 'monot5l', 'olmo', 'phi', 'qwen3']
PUB_LABEL = {'colbertv2': 'ColBERTv2 (native)', 'answerai': 'answerai-colbert-small-v1', 'gtemc': 'GTE-ModernColBERT-v1', 'bgem3c': 'bge-m3, multi-vector', 'bgem3d': 'bge-m3, dense',
             'bgem3s': 'bge-m3, sparse', 'bgebase': 'bge-base-en-v1.5', 'e5': 'e5-base-v2', 'mpnet': 'all-mpnet-base-v2', 'gtemb': 'gte-modernbert-base', 'splade': 'SPLADE (cocondenser-ensembledistil)',
             'minilm': 'ms-marco-MiniLM-L6-v2', 'bgerr': 'bge-reranker-base', 'bgerr2': 'bge-reranker-v2-m3', 'mxbai': 'mxbai-rerank-base-v1', 'monot5b': 'monoT5-base', 'monot5l': 'monoT5-3B',
             'olmo': 'OLMo-2-7B-Instruct (judge)', 'phi': 'Phi-4-mini-instruct (judge)', 'qwen3': 'Qwen3-8B (judge)'}
PUB_TYPE = {'colbertv2': ('late interaction', 'geç etkileşim'), 'answerai': ('late interaction', 'geç etkileşim'), 'gtemc': ('late interaction', 'geç etkileşim'), 'bgem3c': ('late interaction', 'geç etkileşim'),
            'bgem3d': ('dense', 'yoğun'), 'bgem3s': ('learned sparse', 'öğrenilmiş seyrek'), 'bgebase': ('dense', 'yoğun'), 'e5': ('dense', 'yoğun'), 'mpnet': ('dense', 'yoğun'), 'gtemb': ('dense', 'yoğun'),
            'splade': ('learned sparse', 'öğrenilmiş seyrek'), 'minilm': ('cross-encoder', 'çapraz kodlayıcı'), 'bgerr': ('cross-encoder', 'çapraz kodlayıcı'), 'bgerr2': ('cross-encoder', 'çapraz kodlayıcı'),
            'mxbai': ('cross-encoder', 'çapraz kodlayıcı'), 'monot5b': ('monoT5', 'monoT5'), 'monot5l': ('monoT5', 'monoT5'), 'olmo': ('LLM judge', 'LLM yargıç'), 'phi': ('LLM judge', 'LLM yargıç'), 'qwen3': ('LLM judge', 'LLM yargıç')}
PUB_REPO = {'colbertv2': 'colbert-ir/colbertv2.0', 'answerai': 'answerdotai/answerai-colbert-small-v1', 'gtemc': 'lightonai/GTE-ModernColBERT-v1', 'bgem3c': 'BAAI/bge-m3', 'bgem3d': 'BAAI/bge-m3', 'bgem3s': 'BAAI/bge-m3',
            'bgebase': 'BAAI/bge-base-en-v1.5', 'e5': 'intfloat/e5-base-v2', 'mpnet': 'sentence-transformers/all-mpnet-base-v2', 'gtemb': 'Alibaba-NLP/gte-modernbert-base',
            'splade': 'naver/splade-cocondenser-ensembledistil', 'minilm': 'cross-encoder/ms-marco-MiniLM-L6-v2', 'bgerr': 'BAAI/bge-reranker-base', 'bgerr2': 'BAAI/bge-reranker-v2-m3',
            'mxbai': 'mixedbread-ai/mxbai-rerank-base-v1', 'monot5b': 'castorini/monot5-base-msmarco-10k', 'monot5l': 'castorini/monot5-3b-msmarco-10k',
            'olmo': 'allenai/OLMo-2-1124-7B-Instruct', 'phi': 'microsoft/Phi-4-mini-instruct', 'qwen3': 'Qwen/Qwen3-8B'}


def t_pubsys(lang):
    head = T(lang, ['System', 'Type', 'Checkpoint', 'Scoring'], ['Sistem', 'Tür', 'Denetim noktası', 'Puanlama'])
    q = lambda k, name: F(f'pub.{k}.set.{name}', lang)
    scoring = {'colbertv2': ('sum of max cosine over query vectors; q ' + q('colbertv2', 'query_maxlen') + ' ([MASK] expansion), d ' + q('colbertv2', 'doc_maxlen') + ', dim ' + q('colbertv2', 'dim') + ', punctuation masked',
                             'sorgu vektörleri üzerinde en büyük kosinüs toplamı; s ' + q('colbertv2', 'query_maxlen') + ' ([MASK] genişletme), b ' + q('colbertv2', 'doc_maxlen') + ', ' + q('colbertv2', 'dim') + ' boyut, noktalama maskeli'),
               'answerai': ('as ColBERTv2; q ' + q('answerai', 'query_maxlen') + ', d ' + q('answerai', 'doc_maxlen') + ', dim ' + q('answerai', 'dim'), 'ColBERTv2 gibi; s ' + q('answerai', 'query_maxlen') + ', b ' + q('answerai', 'doc_maxlen') + ', ' + q('answerai', 'dim') + ' boyut'),
               'gtemc': ('PyLate encoding replicated; q ' + q('gtemc', 'query_length') + ', d ' + q('gtemc', 'document_length') + ', dim ' + q('gtemc', 'dim'), 'PyLate kodlaması yeniden üretildi; s ' + q('gtemc', 'query_length') + ', b ' + q('gtemc', 'document_length') + ', ' + q('gtemc', 'dim') + ' boyut'),
               'bgem3c': ('mean of max cosine over query vectors; max length ' + q('bgem3c', 'max_length'), 'sorgu vektörleri üzerinde en büyük kosinüs ortalaması (referans başlar); en çok ' + q('bgem3c', 'max_length') + ' token'),
               'bgem3d': ('cosine of [CLS] states', '[CLS] durumlarının kosinüsü'), 'bgem3s': ('sum of products of relu token weights over shared ids', 'ortak kimliklerde relu token ağırlıklarının çarpım toplamı'),
               'bgebase': ('cosine; query instruction prefix; max length ' + q('bgebase', 'max_seq_length'), 'kosinüs; sorgu yönerge öneki; en çok ' + q('bgebase', 'max_seq_length') + ' token'),
               'e5': ('cosine; "query:" / "passage:" prefixes', 'kosinüs; "query:" / "passage:" önekleri'),
               'mpnet': ('cosine, mean pooling', 'kosinüs, ortalama havuzlama'), 'gtemb': ('cosine, [CLS] pooling', 'kosinüs, [CLS] havuzlama'),
               'splade': ('dot product of max-pooled log(1 + relu) vocabulary vectors; max ' + q('splade', 'max_length'), 'en büyük havuzlu log(1 + relu) sözcük dağarcığı vektörlerinin iç çarpımı; en çok ' + q('splade', 'max_length') + ' token'),
               'minilm': ('relevance logit; max length ' + q('minilm', 'max_length'), 'ilgililik logiti; en çok ' + q('minilm', 'max_length') + ' token'), 'bgerr': ('relevance logit', 'ilgililik logiti'), 'bgerr2': ('relevance logit', 'ilgililik logiti'), 'mxbai': ('relevance logit', 'ilgililik logiti'),
               'monot5b': ('log P(true) at the first decoder step; max length ' + q('monot5b', 'max_length'), 'ilk çözücü adımında log P(true); en çok ' + q('monot5b', 'max_length') + ' token'), 'monot5l': ('as monoT5-base; bfloat16 weights, float32 decisive logits', 'monoT5-base gibi; bfloat16 ağırlık, float32 belirleyici logitler'),
               'olmo': ('log P(Yes) − log P(No), pointwise prompt; documents cut at ' + q('olmo', 'max_doc_tokens') + ' tokens', 'log P(Yes) − log P(No), noktasal istem; belgeler ' + q('olmo', 'max_doc_tokens') + ' token\'da kesilir'), 'phi': ('as OLMo-2', 'OLMo-2 gibi'), 'qwen3': ('as OLMo-2; thinking off; family of the second verifier', 'OLMo-2 gibi; düşünme kapalı; ikinci doğrulayıcının ailesi')}
    rows = []
    for k in PUB_ORDER:
        rows.append([PUB_SHORT[k].replace(' (judge)', ''), T(lang, *PUB_TYPE[k]), '`' + PUB_REPO[k] + '`', T(lang, *scoring[k])])
    revs = ', '.join(f'{PUB_SHORT[k].replace(" (judge)", "")} `' + F(f'pub.{k}.revision', lang) + '`' for k in PUB_ORDER if not (k.startswith('bgem3') and k != 'bgem3c'))
    cap = T(lang, 'Released systems scored on the diagnostics (Section 5): checkpoint identifiers on the Hugging Face Hub and the scoring applied; q and d are the query and document token limits. '
                  'Every system was used with its released weights and documented settings; the three bge-m3 rows are three heads of one checkpoint. Revisions resolved at scoring time (first eight characters of the commit): ' + revs.replace('bge-m3 multi-vector', 'bge-m3') + '.',
            'Teşhislerde puanlanan yayımlanmış sistemler (Bölüm 5): Hugging Face Hub kimlikleri ve uygulanan puanlama; s ve b sorgu ve belge token sınırlarıdır. '
            'Her sistem yayımlanmış ağırlıkları ve belgelenmiş ayarlarıyla kullanıldı; üç bge-m3 satırı tek bir denetim noktasının üç başıdır. Puanlama anında çözülen sürümler (commit\'in ilk sekiz karakteri): ' + revs.replace('bge-m3 multi-vector', 'bge-m3') + '.')
    return cap, pipe(head, rows, 'llll', [4, 3, 6, 8])


PUB_SHORT = {'colbertv2': 'ColBERTv2', 'answerai': 'answerai-colbert-small', 'gtemc': 'GTE-ModernColBERT', 'bgem3c': 'bge-m3 multi-vector', 'bgem3d': 'bge-m3 dense', 'bgem3s': 'bge-m3 sparse',
             'bgebase': 'bge-base-en-v1.5', 'e5': 'e5-base-v2', 'mpnet': 'all-mpnet-base-v2', 'gtemb': 'gte-modernbert-base', 'splade': 'SPLADE', 'minilm': 'ms-marco-MiniLM-L6',
             'bgerr': 'bge-reranker-base', 'bgerr2': 'bge-reranker-v2-m3', 'mxbai': 'mxbai-rerank-base', 'monot5b': 'monoT5-base', 'monot5l': 'monoT5-3B', 'olmo': 'OLMo-2-7B (judge)', 'phi': 'Phi-4-mini (judge)', 'qwen3': 'Qwen3-8B (judge)'}


def t_pubperm(lang):
    head = T(lang, ['System', 'acc N', 'flip O→N', 'flip O→S', 'SS_bal', '95% interval', 'NevIR all', '95% interval', 'nested', 'equal'],
             ['Sistem', 'doğr. N', 'flip O→N', 'flip O→S', 'SS_bal', '%95 aralık', 'NevIR tümü', '%95 aralık', 'iç içe', 'eşit'])
    rows = []
    for k in PUB_ORDER:
        rows.append([PUB_SHORT[k], F(f'pub.{k}.perm.acc_N', lang), F(f'pub.{k}.perm.flip_O_N', lang), F(f'pub.{k}.perm.flip_O_S', lang),
                     F(f'pub.{k}.perm.SS_bal', lang), ci(f'pub.{k}.perm.SS_bal', lang),
                     F(f'pub.{k}.nevir.all', lang), ci(f'pub.{k}.nevir.all', lang), F(f'pub.{k}.nevir.nested', lang), F(f'pub.{k}.nevir.equal', lang)])
    cap = T(lang, 'Released systems on NegPairs-Perm (' + F('data.perm.scenarios', lang) + ' scenarios: N-arm accuracy and decision changes in percent of scenarios; order-balanced scope sensitivity in points with its scenario-cluster bootstrap interval) '
                  'and on NevIR (paired accuracy in percent on all ' + F('nevir.instances', lang) + ' instances with its bootstrap interval, on the ' + F('nevir.nested', lang) + ' nested and the ' + F('nevir.equal_bags', lang) + ' equal-bag instances). '
                  'Every unmasked scorer over static token states has SS_bal = 0 and fails every nested instance (Corollaries A.1 and B.1). Complete values: Tables ' + '{{ref:tab:pubpermx}}' + ' and ' + '{{ref:tab:pubnevirx}}' + '.',
            'NegPairs-Perm\'de (' + F('data.perm.scenarios', lang) + ' senaryo: N kolunda doğruluk ve karar değişiklikleri senaryoların yüzdesi; senaryo kümeli bootstrap aralığıyla sıra dengeli kapsam duyarlılığı, puan) '
            've NevIR\'de (tüm ' + F('nevir.instances', lang) + ' örnekte bootstrap aralığıyla, ' + F('nevir.nested', lang) + ' iç içe ve ' + F('nevir.equal_bags', lang) + ' eşit torbalı örnekte eşli doğruluk, yüzde) yayımlanmış sistemler. '
            'Statik token durumları üzerindeki maskesiz her puanlayıcının SS_bal değeri 0\'dır ve her iç içe örnekte başarısızdır (Sonuç A.1 ve B.1). Tam değerler: Tablo ' + '{{ref:tab:pubpermx}}' + ' ve ' + '{{ref:tab:pubnevirx}}' + '.')
    return cap, pipe(head, rows, 'lrrrrlrlrr', [4, 1, 1, 1, 1, 3, 1, 3, 1, 1])


def t_pubpermx(lang):
    head = T(lang, ['System', 'acc O', 'acc P', 'acc N', 'acc S', 'flip O→P', 'flip O→N', 'flip O→S', 'joint (N, S)', 'SS first', 'SS second', 'SS_bal', '95% interval'],
             ['Sistem', 'doğr. O', 'doğr. P', 'doğr. N', 'doğr. S', 'flip O→P', 'flip O→N', 'flip O→S', 'ortak (N, S)', 'SS önce', 'SS sonra', 'SS_bal', '%95 aralık'])
    rows = []
    for k in PUB_ORDER:
        rows.append([PUB_SHORT[k], F(f'pub.{k}.perm.acc_O', lang), F(f'pub.{k}.perm.acc_P', lang), F(f'pub.{k}.perm.acc_N', lang), F(f'pub.{k}.perm.acc_S', lang),
                     F(f'pub.{k}.perm.flip_O_P', lang), F(f'pub.{k}.perm.flip_O_N', lang), F(f'pub.{k}.perm.flip_O_S', lang), F(f'pub.{k}.perm.joint_NS_acc', lang),
                     F(f'pub.{k}.perm.SS_first', lang), F(f'pub.{k}.perm.SS_second', lang), F(f'pub.{k}.perm.SS_bal', lang), ci(f'pub.{k}.perm.SS_bal', lang)])
    cap = T(lang, 'Released systems on NegPairs-Perm, complete values (' + F('data.perm.scenarios', lang) + ' scenarios; ' + F('data.perm.order.first', lang) + ' with the decisive clause first): accuracy per arm, decision changes and joint (N, S) accuracy in percent of scenarios; scope sensitivity with the decisive clause first and second and its order-balanced value in points, with the scenario-cluster bootstrap interval by clause order.',
            'NegPairs-Perm\'de yayımlanmış sistemler, tam değerler (' + F('data.perm.scenarios', lang) + ' senaryo; ' + F('data.perm.order.first', lang) + '\'inde belirleyici yan tümce önce): kol başına doğruluk, karar değişiklikleri ve ortak (N, S) doğruluğu senaryoların yüzdesi; belirleyici yan tümce önce ve sonra iken kapsam duyarlılığı ve sıra dengeli değeri puan olarak, yan tümce sırasına göre senaryo kümeli bootstrap aralığıyla.')
    return cap, pipe(head, rows, 'lrrrrrrrrrrrl', [4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3])


def t_pubfresh(lang):
    head = T(lang, ['System', 'acc O', 'acc N', 'acc S', 'flip O→N', 'flip O→S', 'SS', 'SS interval', 'SS seen', 'SS unseen', 'flip O→P'],
             ['Sistem', 'doğr. O', 'doğr. N', 'doğr. S', 'flip O→N', 'flip O→S', 'SS', 'SS aralığı', 'SS görülen', 'SS görülmeyen', 'flip O→P'])
    rows = []
    for k in PUB_ORDER:
        rows.append([PUB_SHORT[k], F(f'pub.{k}.fresh.acc_O', lang), F(f'pub.{k}.fresh.acc_N', lang), F(f'pub.{k}.fresh.acc_S', lang), F(f'pub.{k}.fresh.flip_O_N', lang), F(f'pub.{k}.fresh.flip_O_S', lang),
                     F(f'pub.{k}.fresh.SS', lang), ci(f'pub.{k}.fresh.SS', lang), F(f'pub.{k}.fresh.SS_seen', lang), F(f'pub.{k}.fresh.SS_unseen', lang), F(f'pub.{k}.fresh.flip_O_P', lang)])
    cap = T(lang, 'Released systems on NegPairs-Fresh (' + F('data.fresh.scenarios', lang) + ' standard minimal-pair scenarios, whose N and S arms differ in their words): accuracy per arm and decision changes (percent of scenarios), scope sensitivity SS (points) with its scenario-cluster bootstrap interval, overall and by family group ("seen" and "unseen" refer to the training families of Section 7; the released systems saw none of this data).',
            'NegPairs-Fresh\'te (' + F('data.fresh.scenarios', lang) + ' standart minimal çift senaryosu; N ve S kolları sözcükleriyle ayrılır) yayımlanmış sistemler: kol başına doğruluk ve karar değişiklikleri (senaryoların yüzdesi), senaryo kümeli bootstrap aralığıyla kapsam duyarlılığı SS (puan), genel ve aile grubuna göre ("görülen" ve "görülmeyen", Bölüm 7\'nin eğitim ailelerine göredir; yayımlanmış sistemler bu verinin hiçbirini görmedi).')
    return cap, pipe(head, rows, 'lrrrrrrlrrr', [5, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1])


def t_pubnevirx(lang):
    head = T(lang, ['System', 'All', 'interval', 'Nested', 'interval', 'Complement', 'interval', 'Equal bags', 'Per-query', 'Ties'],
             ['Sistem', 'Tümü', 'aralık', 'İç içe', 'aralık', 'Tümleyen', 'aralık', 'Eşit torba', 'Sorgu başına', 'Beraberlik'])
    rows = []
    for k in PUB_ORDER:
        rows.append([PUB_SHORT[k], F(f'pub.{k}.nevir.all', lang), ci(f'pub.{k}.nevir.all', lang), F(f'pub.{k}.nevir.nested', lang), ci(f'pub.{k}.nevir.nested', lang),
                     F(f'pub.{k}.nevir.complement', lang), ci(f'pub.{k}.nevir.complement', lang), F(f'pub.{k}.nevir.equal', lang), F(f'pub.{k}.nevir.all.perq', lang), F(f'pub.{k}.nevir.ties', lang)])
    rows.append([T(lang, 'Random ranking', 'Rastgele sıralama'), F('nevir.published.random', lang), '–', F('nevir.published.random', lang), '–', F('nevir.published.random', lang), '–', F('nevir.published.random', lang), '–', '–'])
    cap = T(lang, 'Released systems on NevIR: paired accuracy (percent) with instance-level bootstrap intervals on all instances, the nested subset and its complement; the equal-bag subset; per-query accuracy; and the number of instances with a tied comparison (a tie is an error). '
                  'ColBERTv2\'s ties come from documents that are identical after truncation at ' + F('pub.colbertv2.set.doc_maxlen', lang) + ' tokens.',
            'NevIR\'de yayımlanmış sistemler: tüm örneklerde, iç içe alt kümede ve tümleyeninde örnek düzeyinde bootstrap aralıklarıyla eşli doğruluk (yüzde); eşit torbalı alt küme; sorgu başına doğruluk; ve karşılaştırması berabere biten örnek sayısı (beraberlik hatadır). '
            'ColBERTv2\'nin beraberlikleri ' + F('pub.colbertv2.set.doc_maxlen', lang) + ' token\'da kesildikten sonra özdeşleşen belgelerden gelir.')
    return cap, pipe(head, rows, 'lrlrlrlrrr', [5, 1, 3, 1, 3, 1, 3, 1, 1, 1])


def t_pubprec(lang):
    head = T(lang, ['System', 'Decisive logits', 'NevIR all', 'NevIR nested', 'NevIR ties', 'Perm SS_bal', 'Perm ties', 'Fresh SS', 'Fresh ties'],
             ['Sistem', 'Belirleyici logitler', 'NevIR tümü', 'NevIR iç içe', 'NevIR beraberlik', 'Perm SS_bal', 'Perm beraberlik', 'Fresh SS', 'Fresh beraberlik'])
    rows = []
    for k in ('monot5b', 'monot5l', 'phi', 'olmo', 'qwen3'):
        for tag, lab in (('prev', T(lang, 'bfloat16 (first pass)', 'bfloat16 (ilk geçiş)')), ('new', T(lang, 'float32 (reported)', 'float32 (bildirilen)'))):
            rows.append([PUB_SHORT[k] if tag == 'prev' else '', lab, F(f'pub.prec.{k}.{tag}.nevir_all', lang), F(f'pub.prec.{k}.{tag}.nevir_nested', lang), F(f'pub.prec.{k}.{tag}.nevir_ties', lang),
                         F(f'pub.prec.{k}.{tag}.perm_SS_bal', lang), F(f'pub.prec.{k}.{tag}.perm_ties', lang), F(f'pub.prec.{k}.{tag}.fresh_SS', lang), F(f'pub.prec.{k}.{tag}.fresh_ties', lang)])
    cap = T(lang, 'Precision control for the systems run in bfloat16: the same models scored with the decisive logits taken from the bfloat16 output (first pass) and computed in float32 from the state entering the output head (reported values). '
                  'A tie is a comparison with identical scores; it counts as an error on NevIR and as the first option on NegPairs. Both passes used identical prompts, weights and data; values are percent (NevIR) and points (SS).',
            'bfloat16 ile çalıştırılan sistemler için kesinlik denetimi: aynı modeller, belirleyici logitleri bfloat16 çıktısından alınarak (ilk geçiş) ve çıkış katmanına giren durumdan float32 ile hesaplanarak (bildirilen değerler) puanlandı. '
            'Beraberlik özdeş puanlı karşılaştırmadır; NevIR\'de hata, NegPairs\'te ilk seçenek sayılır. İki geçiş de aynı istemleri, ağırlıkları ve veriyi kullandı; değerler yüzde (NevIR) ve puan (SS).')
    return cap, pipe(head, rows, 'llrrrrrrr', [4, 3, 1, 1, 1, 1, 1, 1, 1])


# ------------------------------------------------------------------------------------------- fine-tuning (revision step 3)
FT_ROWS = [('colbertv2', 'base', 'ColBERTv2', ('released', 'yayımlanmış')), ('colbertv2', 'np', 'ColBERTv2', ('NegPairs v1 train', 'NegPairs v1 train')), ('colbertv2', 'nv', 'ColBERTv2', ('NevIR train', 'NevIR train')),
           ('answerai', 'base', 'answerai-colbert-small-v1', ('released', 'yayımlanmış')), ('answerai', 'np', 'answerai-colbert-small-v1', ('NegPairs v1 train', 'NegPairs v1 train')), ('answerai', 'nv', 'answerai-colbert-small-v1', ('NevIR train', 'NevIR train'))]


def t_ft(lang):
    head = T(lang, ['Model', 'Training', 'Perm SS_bal', 'seen', 'unseen', 'flip O→N', 'flip O→S', 'Fresh SS', 'NevIR all', 'nested', 'equal', 'SciFact nDCG@10'],
             ['Model', 'Eğitim', 'Perm SS_bal', 'görülen', 'görülmeyen', 'flip O→N', 'flip O→S', 'Fresh SS', 'NevIR tümü', 'iç içe', 'eşit', 'SciFact nDCG@10'])
    rows = []
    for m, c, label, tr in FT_ROWS:
        rows.append([label if c == 'base' else '', T(lang, *tr), F(f'ft.{m}.{c}.perm.SS_bal', lang), F(f'ft.{m}.{c}.perm.SS_bal_seen', lang), F(f'ft.{m}.{c}.perm.SS_bal_unseen', lang),
                     F(f'ft.{m}.{c}.perm.flip_O_N', lang), F(f'ft.{m}.{c}.perm.flip_O_S', lang), F(f'ft.{m}.{c}.fresh.SS', lang), F(f'ft.{m}.{c}.nevir.all', lang), F(f'ft.{m}.{c}.nevir.nested', lang),
                     F(f'ft.{m}.{c}.nevir.equal', lang), F(f'ft.{m}.{c}.scifact', lang)])
    cap = T(lang, 'Two late-interaction checkpoints before and after end-to-end fine-tuning on NegPairs v1 train or NevIR train: means over ' + F('ft.cfg.n_seeds', lang) + ' seeds (the released checkpoint scored once). '
                  'Perm: order-balanced scope sensitivity (points) overall and by family group, and decision changes (percent of scenarios); Fresh: scope sensitivity; NevIR: paired accuracy (percent); SciFact: nDCG@10 (percent) by brute-force MaxSim. Per-seed values: Table ' + '{{ref:tab:ftseeds}}' + '.',
            'İki geç etkileşim denetim noktası, NegPairs v1 train veya NevIR train üzerinde uçtan uca ince ayardan önce ve sonra: ' + F('ft.cfg.n_seeds', lang) + ' tohum üzerinden ortalamalar (yayımlanmış denetim noktası bir kez puanlandı). '
            'Perm: genel ve aile grubuna göre sıra dengeli kapsam duyarlılığı (puan) ve karar değişiklikleri (senaryoların yüzdesi); Fresh: kapsam duyarlılığı; NevIR: eşli doğruluk (yüzde); SciFact: kaba kuvvet MaxSim ile nDCG@10 (yüzde). Tohum başına değerler: Tablo ' + '{{ref:tab:ftseeds}}' + '.')
    return cap, pipe(head, rows, 'llrrrrrrrrrr', [4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])


def t_ftseeds(lang):
    head = T(lang, ['Model', 'Training', 'Seed', 'Epoch', 'Dev before', 'Dev after', 'Perm SS_bal', 'Fresh SS', 'NevIR all', 'NevIR nested', 'SciFact nDCG@10'],
             ['Model', 'Eğitim', 'Tohum', 'Dönem', 'Dev önce', 'Dev sonra', 'Perm SS_bal', 'Fresh SS', 'NevIR tümü', 'NevIR iç içe', 'SciFact nDCG@10'])
    rows = []
    seeds = [s.strip() for s in val('ft.cfg.seeds').split(',')]
    for m, c, label, tr in FT_ROWS:
        if c == 'base':
            continue
        epochs = [e.strip() for e in val(f'ft.{m}.{c}.epochs').split(',')]
        for i, sd in enumerate(seeds, 1):
            rows.append([label if i == 1 else '', T(lang, *tr) if i == 1 else '', sd, epochs[i - 1], F(f'ft.{m}.{c}.dev_before', lang) if i == 1 else '', F(f'ft.{m}.{c}.dev_after.s{i}', lang) if f'ft.{m}.{c}.dev_after.s{i}' in REG else '–',
                         F(f'ft.{m}.{c}.perm.SS_bal.s{i}', lang), F(f'ft.{m}.{c}.fresh.SS.s{i}', lang), F(f'ft.{m}.{c}.nevir.all.s{i}', lang), F(f'ft.{m}.{c}.nevir.nested.s{i}', lang), F(f'ft.{m}.{c}.scifact.s{i}', lang)])
    cap = T(lang, 'Fine-tuning runs by seed: selected epoch (0 = the released weights were kept), the selection metric before and after training (v1 dev accuracy or NevIR validation paired accuracy, percent), and the test values of the selected state.',
            'Tohuma göre ince ayar koşuları: seçilen dönem (0 = yayımlanmış ağırlıklar korundu), eğitimden önce ve sonra seçim ölçütü (v1 dev doğruluğu veya NevIR validation eşli doğruluğu, yüzde) ve seçilen durumun test değerleri.')
    return cap, pipe(head, rows, 'lllrrrrrrrr', [4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2])


TABLES.update({'pubsys': t_pubsys, 'pubperm': t_pubperm, 'pubpermx': t_pubpermx, 'pubfresh': t_pubfresh, 'pubnevirx': t_pubnevirx, 'pubprec': t_pubprec, 'ft': t_ft, 'ftseeds': t_ftseeds})
