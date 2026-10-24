"""NegPairs-Perm builder and the item writer of the v3.0 sets (manifest v3.0, sections 4.2, 4.3 and 4.7).

A Perm scenario is one sentence of two coordinated clauses with one edit site each (the same word W, or an insertion
point after an auxiliary or copula). N applies the family's negation edit at the site in the decisive clause and S at
the site in the other clause, so N and S contain the same words and differ only in where the edit sits. The generator
returns the context with both sites marked as [[...]]; O, N and S are assembled here deterministically, and the eleven
filters of manifest v3.0, section 4.3, are applied in the listed order (a candidate is discarded at the first failure).
"""
from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .negpairs_build import (BANNED_TOPICS, CORE_NEG, FAMILIES, cue_in_family, extract_json, has_word, new_entities,
                             schedule as _v1_schedule, token_edit_distance, verify_arms_v2, words)
from .perm_prompts import PERM_GEN_PROMPT, PERM_PATTERNS

PATTERN_NAME = {"syntactic": "not-insertion", "quantifier": "determiner", "affixal": "prefix", "lexical": "lexical-pair"}
PERM_AUX = ("is", "are", "was", "were", "can", "will", "should", "has", "have")
QUANT_DETS = ("a", "an", "some")
PREFIXES = ("un", "in", "im", "il", "ir", "dis", "non")
LEXICAL_PAIRS = {"has": "lacks", "have": "lack", "had": "lacked", "with": "without", "present": "absent"}
# prefixed words that exist but do not negate their base (they would mislabel the affixal family)
AFFIX_STOPLIST = frozenset({
    "invaluable", "infamous", "inflammable", "inhabitable", "discovered", "disclosed", "dispatched", "dismissed",
    "discharged", "disposed", "discounted", "displayed", "dissolved", "disgusted", "informed", "invested", "installed",
    "impressed", "imposed", "improved", "imported", "intended", "inset", "indented"})
AFFIX_MIN_ZIPF = 2.0
UN_BASE_MIN_ZIPF = 3.0
# the exceptions listed in negpairs_build.cue_in_family (kept identical; checked by a unit test)
NOT_NEGATIONS = frozenset({
    "unless", "until", "under", "unit", "unique", "union", "universe", "university", "insist", "inside", "income",
    "index", "into", "image", "impact", "issue", "display", "discuss", "dish", "distance", "interest", "include",
    "increase", "instead", "illustrate", "irrigate", "nonetheless", "regardless"})
E0_RE = re.compile(r"\w+|[^\w\s]")
CASE_RE = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']")
MARK_CHUNK = re.compile(r"^\[\[([^\[\]]*)\]\]$")
REQUIRED = ("question", "options", "answer", "context_marked", "connector", "decisive", "negated_form", "context_P")
MAX_WORDS = 30
ARMS = ("O", "P", "N", "S")


def zipf(word: str) -> float:
    from wordfreq import zipf_frequency       # pinned to wordfreq 3.1.1 by the notebooks
    return zipf_frequency(word, "en")


def e0_bag(text: str) -> Counter:
    return Counter(t.lower() for t in E0_RE.findall(text))


def case_bag(text: str) -> Counter:
    return Counter(CASE_RE.findall(text))


# ----------------------------------------------------------------------------------------------- schedules
@dataclass(frozen=True)
class PermCandidate:
    uid: str
    family: str
    domain: str
    situation: str
    decisive_first: bool
    index: int


@dataclass(frozen=True)
class FreshCandidate:
    uid: str
    family: str
    domain: str
    situation: str
    index: int


def perm_schedule(per_family: int, seed: int, order_seed: int) -> List[PermCandidate]:
    """v1 schedule (balanced family x domain x situation, families interleaved) plus a clause order drawn by a seeded
    shuffle of a balanced list per family, independent of domain and situation."""
    base = _v1_schedule({f: per_family for f in FAMILIES}, 1.0, seed)
    rng = random.Random(order_seed)
    orders = {}
    for f in FAMILIES:
        n = sum(1 for c in base if c.family == f)
        lst = [True, False] * (n // 2) + ([True] if n % 2 else [])
        rng.shuffle(lst)
        orders[f] = iter(lst)
    return [PermCandidate(uid="perm-" + c.uid, family=c.family, domain=c.domain, situation=c.situation,
                          decisive_first=next(orders[c.family]), index=k) for k, c in enumerate(base)]


def fresh_schedule(per_family: int, seed: int) -> List[FreshCandidate]:
    base = _v1_schedule({f: per_family for f in FAMILIES}, 1.0, seed)
    return [FreshCandidate(uid="fresh-" + c.uid, family=c.family, domain=c.domain, situation=c.situation, index=k)
            for k, c in enumerate(base)]


def perm_prompt(c: PermCandidate) -> str:
    p = PERM_PATTERNS[c.family]
    return PERM_GEN_PROMPT.format(domain=c.domain, situation=c.situation, banned=BANNED_TOPICS,
                                  first="DECISIVE" if c.decisive_first else "OTHER", family=c.family,
                                  pattern_desc=p["pattern_desc"], negated_form_desc=p["negated_form_desc"])


# ----------------------------------------------------------------------------------------------- assembly
def parse_marks(marked: str) -> Tuple[Optional[str], Optional[List[str]], Optional[List[int]], Optional[List[str]]]:
    chunks = (marked or "").split()
    if marked.count("[[") != 2 or marked.count("]]") != 2:
        return "marks: exactly two marks required", None, None, None
    idx = [i for i, ch in enumerate(chunks) if MARK_CHUNK.match(ch)]
    if len(idx) != 2:
        return "marks: each mark must stand alone between spaces", None, None, None
    if idx[0] == 0:
        return "marks: the first word is marked", None, None, None
    if idx[1] == idx[0] + 1:
        return "marks: no text between the marks", None, None, None
    contents = [MARK_CHUNK.match(chunks[i]).group(1) for i in idx]
    return None, chunks, idx, contents


def _connector_regex(connector: str) -> re.Pattern:
    """Whole-word match: a word boundary is required on a side only if the connector ends there in a letter or digit
    (", and" may follow "today"; "and" must not match inside "standard")."""
    c = connector.strip()
    left = r"(?<![A-Za-z0-9'])" if re.match(r"[A-Za-z0-9]", c) else ""
    right = r"(?![A-Za-z0-9'])" if re.search(r"[A-Za-z0-9]$", c) else ""
    return re.compile(left + re.escape(c) + right, re.I)


def assemble(chunks: List[str], idx: List[int], contents: List[str], family: str, negated_form: str,
             decisive_first: bool, connector: str) -> Tuple[Optional[str], Optional[dict]]:
    """Build O, N and S from the marked chunks; locate the connector between the marks; return character spans."""
    dec = 0 if decisive_first else 1
    cue = "not" if family == "syntactic" else (negated_form or "").strip()
    if not connector or not connector.strip():
        return "connector: empty", None
    rx = _connector_regex(connector)
    arms = {}
    for arm, edit in (("O", None), ("N", dec), ("S", 1 - dec)):
        out, src = [], []
        for i, ch in enumerate(chunks):
            if i in idx:
                k = idx.index(i)
                if family == "syntactic":
                    if k == edit:
                        out.append("not")
                        src.append(("mark", k))
                else:
                    out.append(cue if k == edit else contents[k])
                    src.append(("mark", k))
            else:
                out.append(ch)
                src.append(i)
        starts, pos = [], 0
        for j, ch in enumerate(out):
            if j:
                pos += 1
            starts.append(pos)
            pos += len(ch)
        text = " ".join(out)
        between = [j for j, s in enumerate(src) if isinstance(s, int) and idx[0] < s < idx[1]]
        r0, r1 = starts[between[0]], starts[between[-1]] + len(out[between[-1]])
        hits = list(rx.finditer(text, r0, r1))
        if len(hits) != 1:
            return f"connector: found {len(hits)} times between the marks (must be 1)", None
        cs, ce = hits[0].start(), hits[0].end()
        c1 = (0, len(text[:cs].rstrip()))
        c2 = (ce + (len(text[ce:]) - len(text[ce:].lstrip())), len(text))
        dec_clause, oth_clause = (c1, c2) if decisive_first else (c2, c1)
        def span_of(k):
            for j, s in enumerate(src):
                if s == ("mark", k):
                    return [starts[j], starts[j] + len(out[j])]
            return None
        spans = {"clause_decisive": list(dec_clause), "clause_other": list(oth_clause), "connector": [cs, ce]}
        if arm == "O":
            spans["cue"] = None
            spans["unedited_W"] = None
            spans["sites"] = None if family == "syntactic" else [span_of(0), span_of(1)]
        else:
            spans["cue"] = span_of(edit)
            spans["unedited_W"] = None if family == "syntactic" else span_of(1 - edit)
            spans["sites"] = None
        arms[arm] = {"text": text, "spans": spans}
    if (arms["N"]["text"][arms["N"]["spans"]["connector"][0]:arms["N"]["spans"]["connector"][1]] !=
            arms["S"]["text"][arms["S"]["spans"]["connector"][0]:arms["S"]["spans"]["connector"][1]]):
        return "connector: differs between N and S", None
    site_words = [chunks[i - 1] for i in idx] if family == "syntactic" else list(contents)
    rec = {"O": arms["O"]["text"], "N": arms["N"]["text"], "S": arms["S"]["text"],
           "spans": {a: arms[a]["spans"] for a in ("O", "N", "S")}, "cue": cue, "decisive_mark": dec,
           "site_words": site_words, "connector": connector.strip()}
    return None, rec


def pattern_check(family: str, chunks: List[str], idx: List[int], contents: List[str], negated_form: str) -> Optional[str]:
    nf = (negated_form or "").strip()
    if family == "syntactic":
        if contents != ["^", "^"]:
            return "pattern: syntactic sites must be insertion marks [[^]]"
        for i in idx:
            if chunks[i - 1].lower() not in PERM_AUX:
                return f"pattern: word before an insertion mark is {chunks[i - 1]!r}, not an auxiliary of the list"
        if nf.lower() != "not":
            return "pattern: negated form must be 'not'"
        return None
    if contents[0] != contents[1]:
        return "pattern: the two marked words differ"
    w = contents[0]
    if family == "quantifier":
        if w not in QUANT_DETS:
            return f"pattern: marked word {w!r} is not a, an or some"
        if nf != "no":
            return "pattern: negated form must be 'no'"
        return None
    if family == "affixal":
        if not re.fullmatch(r"[a-z]+", w):
            return "pattern: affixal W must be one lower-case word"
        if not any(nf == p + w for p in PREFIXES):
            return f"pattern: negated form {nf!r} is not a prefix of the list followed by {w!r}"
        if nf in AFFIX_STOPLIST:
            return f"pattern: {nf!r} is on the stoplist (prefixed word that does not negate its base)"
        if zipf(nf) < AFFIX_MIN_ZIPF:
            return f"pattern: {nf!r} has Zipf frequency below {AFFIX_MIN_ZIPF}"
        for i in idx:
            if chunks[i - 1].lower() in ("a", "an"):
                return "pattern: affixal site preceded by a or an"
        return None
    if family == "lexical":
        if w not in LEXICAL_PAIRS:
            return f"pattern: marked word {w!r} is not has, have, had, with or present"
        if nf != LEXICAL_PAIRS[w]:
            return f"pattern: negated form must be {LEXICAL_PAIRS[w]!r}"
        return None
    return f"pattern: unknown family {family!r}"


def un_negation(text: str) -> Optional[str]:
    for w in words(text):
        if len(w) > 4 and w.startswith("un") and w not in NOT_NEGATIONS and zipf(w[2:]) >= UN_BASE_MIN_ZIPF:
            return w
    return None


def perm_filter(g: dict, c: PermCandidate) -> Tuple[Optional[str], Optional[dict]]:
    """The eleven deterministic checks of manifest v3.0, section 4.3, in order. Returns (reason, None) or (None, rec)."""
    # 1
    miss = [k for k in REQUIRED if not g.get(k)]
    if miss:
        return f"1 missing fields {miss}", None
    if g["answer"] not in ("A", "B"):
        return "1 answer malformed", None
    # 2
    opts, q = g["options"], g["question"]
    if not (isinstance(opts, list) and len(opts) == 2 and all(isinstance(o, str) for o in opts)):
        return "2 options malformed", None
    if opts[0].strip().lower() == opts[1].strip().lower():
        return "2 identical options", None
    if any(not (1 <= len(words(o)) <= 6) for o in opts):
        return "2 option length", None
    if any(CORE_NEG.search(o) for o in opts):
        return "2 negation marker in an option", None
    if e0_bag(opts[0]) == e0_bag(opts[1]):
        return "2 options have the same bag of tokens", None
    if not isinstance(q, str) or not q.strip().endswith("?"):
        return "2 question without question mark", None
    if CORE_NEG.search(q):
        return "2 negation marker in the question", None
    # 3
    reason, chunks, idx, contents = parse_marks(g["context_marked"])
    if reason:
        return "3 " + reason, None
    reason, rec = assemble(chunks, idx, contents, c.family, g["negated_form"], c.decisive_first, g["connector"])
    if reason:
        return "3 " + reason, None
    # 4
    reason = pattern_check(c.family, chunks, idx, contents, g["negated_form"])
    if reason:
        return "4 " + reason, None
    # 5
    if not cue_in_family(rec["cue"], c.family):
        return f"5 cue {rec['cue']!r} is not a {c.family} cue", None
    O, N, S, P = rec["O"], rec["N"], rec["S"], g["context_P"]
    # 6
    for name, text in (("O", O), ("P", P)):
        if CORE_NEG.search(text):
            return f"6 negation marker in context {name}", None
        un = un_negation(text)
        if un:
            return f"6 affixal negation {un!r} in context {name}", None
    # 7
    for name, text in (("O", O), ("P", P), ("N", N), ("S", S)):
        if len(words(text)) > MAX_WORDS:
            return f"7 context {name} longer than {MAX_WORDS} words", None
    # 8 (marks that the generator copied into `decisive` are removed first: the prompt asks for the words without marks)
    decisive = " ".join(g["decisive"].replace("[[^]]", " ").replace("[[", "").replace("]]", "").split())
    a, b = rec["spans"]["O"]["clause_decisive"]
    if not decisive or not has_word(O[a:b], decisive):
        return "8 decisive words not inside the decisive clause of O", None
    rec["decisive"] = decisive
    # 9
    for name, text in (("N", N), ("S", S)):
        d = token_edit_distance(words(O), words(text))
        if d != 1:
            return f"9 word edit distance O to {name} is {d} (must be 1)", None
    # 10
    if e0_bag(N) != e0_bag(S) or case_bag(N) != case_bag(S):
        return "10 N and S have different bags of tokens", None
    if N == S:
        return "10 N equals S", None
    # 11
    if token_edit_distance(words(O), words(P)) < 2:
        return "11 context P is not a paraphrase (too close to O)", None
    if new_entities(P, O):
        return "11 context P introduces a new name or number", None
    rec["P"] = P
    return None, rec


def perm_as_v2(g: dict, rec: dict) -> dict:
    """The fields that negpairs_build.verify_arms_v2 reads."""
    return {"answer_O": g["answer"], "options": g["options"], "question": g["question"],
            "context_O": rec["O"], "context_P": rec["P"], "context_N": rec["N"], "context_S": rec["S"]}


def scenario_arms(r: dict, kind: str) -> Tuple[Dict[str, str], str, List[str], str]:
    """(arms, question, options in generator order, answer letter) of a record of either set."""
    g = r["generated"]
    if kind == "perm":
        rec = r["assembled"]
        return {"O": rec["O"], "P": rec["P"], "N": rec["N"], "S": rec["S"]}, g["question"], g["options"], g["answer"]
    return {a: g["context_" + a] for a in ARMS}, g["question"], g["options"], g["answer_O"]


# ----------------------------------------------------------------------------------------------- one candidate
def process_perm(c: PermCandidate, call_gen: Callable[[str, str, int], str], call_ver: Callable[[str, str, int], str],
                 gen_model: str, ver_model: str, is_duplicate: Optional[Callable[[str, str, Sequence[str]], bool]] = None,
                 gen_max_tokens: int = 4000) -> dict:
    meta = {"uid": c.uid, "family": c.family, "domain": c.domain, "situation": c.situation,
            "decisive_first": c.decisive_first, "index": c.index}
    try:
        text = call_gen(gen_model, perm_prompt(c), gen_max_tokens)
    except ValueError as e:              # empty or cut-off response (raised by the call wrapper)
        return {"stage": "generator", "reason": str(e)[:80], "meta": meta}
    g = extract_json(text)
    if not g:
        return {"stage": "generator", "reason": "unparseable output", "meta": meta}
    if not g.get("usable"):
        return {"stage": "generator", "reason": str(g.get("reason", "declined"))[:80], "meta": meta, "generated": g}
    reason, rec = perm_filter(g, c)
    if reason:
        return {"stage": "filter", "reason": reason, "generated": g, "meta": meta}
    if is_duplicate is not None and is_duplicate(c.domain, rec["O"], g["options"]):
        return {"stage": "duplicate", "reason": "near-duplicate (before verification)", "generated": g,
                "assembled": rec, "meta": meta}
    reason, verdicts = verify_arms_v2(perm_as_v2(g, rec), call_ver, ver_model)
    if reason:
        return {"stage": "verifier", "reason": reason, "generated": g, "assembled": rec, "verdicts": verdicts, "meta": meta}
    return {"stage": "haiku_passed", "generated": g, "assembled": rec, "verdicts": verdicts, "meta": meta}


# ----------------------------------------------------------------------------------------------- items
def to_items_v3(accepted: List[dict], kind: str, split: str, seed: int) -> List[dict]:
    """One row per arm (v1 format). Option order randomized per scenario with Random(seed + 1), scenarios sorted by
    identifier, as in v1; Perm rows carry the field `perm` (spans refer to the row's own context)."""
    rng = random.Random(seed + 1)
    rows = []
    for r in sorted(accepted, key=lambda x: x["meta"]["uid"]):
        g, m = r["generated"], r["meta"]
        arms, question, options, answer = scenario_arms(r, kind)
        swap = rng.random() < 0.5
        opts = [options[1], options[0]] if swap else list(options)
        y = 0 if answer == "A" else 1
        if swap:
            y = 1 - y
        labels = {"O": y, "P": y, "N": 1 - y, "S": y}
        if kind == "perm":
            cue = r["assembled"]["cue"]
            cues = {"O": "", "P": "", "N": cue, "S": cue}
        else:
            cues = {"O": "", "P": "", "N": g["cue_N"], "S": g["cue_S"]}
        decisive = r["assembled"]["decisive"] if kind == "perm" else g["decisive"]
        for arm in ARMS:
            row = {"scenario_id": m["uid"], "arm": arm, "item_id": f"{m['uid']}-{arm}", "family": m["family"],
                   "seen_family": FAMILIES[m["family"]]["seen"], "domain": m["domain"], "situation": m["situation"],
                   "split": split, "context": arms[arm], "question": question, "options": opts, "label": labels[arm],
                   "cue": cues[arm], "decisive": decisive}
            if kind == "perm":
                rec = r["assembled"]
                row["perm"] = {"pattern": PATTERN_NAME[m["family"]], "site_words": rec["site_words"],
                               "negated_form": rec["cue"], "decisive_first": m["decisive_first"],
                               "connector": rec["connector"], "spans": rec["spans"].get(arm)}
            rows.append(row)
    return rows
