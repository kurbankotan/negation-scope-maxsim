"""NegPairs builder: controlled minimal-pair scenarios in the CACSA format (context, question, two content options).

Each scenario has four arms that share the question and the two options:
  O  affirmative context; answer y
  P  paraphrase of O (same meaning, no negation); answer y                      (edit-cost control)
  N  O with ONE negation cue applied to the decisive words; answer flips to 1-y  (polarity probe)
  S  O with ONE negation cue on an irrelevant detail; answer stays y              (scope control)

Pipeline per candidate: generate (one call) -> deterministic filters (free) -> blind verification of every arm by a
different model (four calls) -> keep. Families F1 syntactic and F2 quantifier are "seen" (train/dev/test); F3 affixal
and F4 lexical are "unseen" (test only). This file is imported by the Colab notebook; `call` is injected so that the
pipeline can be tested offline with a stub.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

FAMILIES = {
    "syntactic": {"desc": "a negative particle or adverb on the decisive verb or adjective: not, n't, never, cannot",
                  "examples": "is not; isn't; never; cannot", "seen": True},
    "quantifier": {"desc": "a negative determiner or pronoun in the decisive phrase: no, none, nobody, nothing, neither, nowhere",
                   "examples": "no; none of; nobody; nothing", "seen": True},
    "affixal": {"desc": "a negative prefix or suffix on the decisive word: un-, in-, im-, il-, ir-, dis-, non-, -less "
                        "(replace one word by its negated form, e.g. available -> unavailable, harmful -> harmless)",
                "examples": "unavailable; incomplete; dishonest; harmless", "seen": False},
    "lexical": {"desc": "a word or phrase that expresses negation lexically: lacks, fails to, absent, without, free of, "
                        "out of, refuses to, missing",
                "examples": "lacks; failed to; is out of; is free of", "seen": False},
}
DOMAINS = ["daily routine", "food and dining", "health and medicine", "travel and transport", "weather and outdoors",
           "home and household", "work and office", "technology and devices", "money and shopping", "school and study",
           "sports and fitness", "events and appointments", "pets and animals", "gardening and farming"]
SITUATIONS = ["availability of a resource", "the state or condition of an object", "a person's state or need",
              "a rule, permission or requirement", "distance or location", "time or schedule",
              "capacity or quantity", "safety or risk"]
BANNED_TOPICS = ("a car at a car wash, nut allergies, umbrellas and rain, a forgotten wallet and lunch, walking to a park, "
                 "printer ink, phone batteries, a drained pool, hotel check-in, a closed bridge, a library closing, "
                 "an elevator under maintenance, server memory, disk space, warranties, spoiled milk, cancelled meetings, homework")

GEN_PROMPT = """You are writing ONE test item for a study of how reading models handle negation.

Write a short, realistic everyday scenario in the domain "{domain}" about {situation}.
Do NOT use any of these topics: {banned}.

Return these fields:
- question: one short question asking what someone should do. It must not reveal the answer and must contain no negation.
- options: exactly two short action phrases (2 to 5 words each). Both must be sensible actions in general; neither may contain a negation word.
- context_O: one or two plain sentences (at most 25 words in total) stating facts that make exactly one option the correct choice. context_O must contain NO negation of any kind (no not, n't, no, never, none, nothing, nobody, without, lacks, fails, un-/in-/dis-/non- words meaning "not", -less words, and so on).
- answer_O: "A" if the first option is correct for context_O, "B" if the second.
- decisive: the exact words of context_O that decide the answer.
- context_P: a paraphrase of context_O: same meaning and same answer, clearly different wording, also with no negation of any kind.
- context_N: context_O changed MINIMALLY by applying ONE negation cue of the family "{family}" ({family_desc}) to the decisive words, so that the OTHER option becomes the correct choice. Change at most three words. Keep every name, number, object and the sentence structure.
- cue_N: the exact negation cue as it appears in context_N (examples for this family: {examples}).
- context_S: context_O with ONE negation cue of the same family added on some OTHER, irrelevant detail, so that the answer stays exactly the same as for context_O. The decisive words must stay unchanged and un-negated.
- cue_S: the exact negation cue as it appears in context_S.

If you cannot satisfy every constraint naturally, return {{"usable": false, "reason": "<= 8 words"}}.

Output strict JSON only:
{{"usable": true, "question": "...", "options": ["...", "..."], "answer_O": "A", "context_O": "...", "decisive": "...",
  "context_P": "...", "context_N": "...", "cue_N": "...", "context_S": "...", "cue_S": "..."}}"""

VERIFY_PROMPT = """Read the context and answer the question by choosing option A or option B. Use only the context.

CONTEXT: {context}
QUESTION: {question}
A) {option_a}
B) {option_b}

Return strict JSON only:
{{"answer": "A" | "B" | "unclear", "natural": true | false, "negation_present": true | false}}
- "answer": the option the context supports; "unclear" if the context does not settle it.
- "natural": false if the context is ungrammatical, odd, or contradicts itself.
- "negation_present": true if the context contains negation of any kind (not, n't, no, never, none, nobody, nothing, without, lacks, fails to, free of, out of, un-/in-/dis-/non- words meaning "not", -less words meaning "without", and similar)."""

# Core negation markers (closed-class and common lexical items). Used only as a cheap pre-filter on O, P, the question
# and the options; the independent model judgement ("negation_present") is the decisive check. Affixal negation is
# deliberately not pattern-matched here (too many false positives) and is left to the verifier.
CORE_NEG = re.compile(r"\b(not|no|never|none|nothing|nobody|nowhere|neither|nor|without|cannot|lacks?|lacking|lacked|"
                      r"fails?|failed|absent|missing|refuses?|refused)\b|n['’]t\b", re.I)
_WORD = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']")


def words(s: str) -> List[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z0-9']+", s)]


def token_edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def entities(s: str) -> set:
    out, toks = set(), s.split()
    for i, t in enumerate(toks):
        w = t.strip('.,;:()"\'`!?')
        if not w:
            continue
        if re.fullmatch(r"\d[\d,.\-]*", w):
            out.add(w)
        elif i > 0 and w[:1].isupper() and not toks[i - 1].endswith(('.', '!', '?')):
            out.add(w.lower())
    return out


def extract_json(text: str) -> Optional[dict]:
    t = re.sub(r"^```[a-z]*\n", "", (text or "").strip())
    t = re.sub(r"\n```$", "", t)
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def content_set(s: str) -> set:
    stop = {"the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "in", "on", "at", "for", "and", "or", "it",
            "this", "that", "with", "has", "have", "had", "will", "can", "should", "his", "her", "their", "its", "by"}
    return {w for w in words(s) if w not in stop}


@dataclass
class Candidate:
    uid: str
    family: str
    domain: str
    situation: str


def schedule(targets: Dict[str, int], oversample: float, seed: int) -> List[Candidate]:
    """Balanced (family, domain, situation) work list; families interleaved so every family progresses together."""
    rng = random.Random(seed)
    per_family = {f: int(round(n * oversample)) for f, n in targets.items()}
    lists = {}
    for f, n in per_family.items():
        combos = [(d, s) for d in DOMAINS for s in SITUATIONS]
        rng.shuffle(combos)
        lists[f] = [Candidate(uid=f"{f[:3]}-{k:05d}", family=f, domain=combos[k % len(combos)][0],
                              situation=combos[k % len(combos)][1]) for k in range(n)]
    out, k = [], 0
    while any(k < len(v) for v in lists.values()):
        for f in targets:
            if k < len(lists[f]):
                out.append(lists[f][k])
        k += 1
    return out


FAMILY_CUE = {
    "syntactic": re.compile(r"^(?:.*\b)?(not|never|cannot|can['’]t|n['’]t|\w+n['’]t)(?:\b.*)?$", re.I),
    "quantifier": re.compile(r"^(?:.*\b)?(no|none|nobody|nothing|neither|nowhere|no one)(?:\b.*)?$", re.I),
    "affixal": re.compile(r"^(?:un|in|im|il|ir|dis|non)-?[a-z]{3,}$|^[a-z]{3,}less$", re.I),
    "lexical": re.compile(r"\b(lack|lacks|lacked|lacking|fail|fails|failed|absent|without|free of|free from|out of|"
                          r"refuse|refuses|refused|missing|devoid of)\b", re.I),
}


def cue_in_family(cue: str, family: str) -> bool:
    """Dataset-construction check that the declared cue belongs to the assigned family (keeps families disjoint)."""
    c = cue.strip().strip('.,;:!?"\'')
    if family == "affixal":
        not_negations = {"unless", "until", "under", "unit", "unique", "union", "universe", "university", "insist", "inside",
                         "income", "index", "into", "image", "impact", "issue", "display", "discuss", "dish", "distance",
                         "interest", "include", "increase", "instead", "illustrate", "irrigate", "nonetheless", "regardless"}
        return bool(FAMILY_CUE["affixal"].match(c)) and not CORE_NEG.search(c) and c.lower() not in not_negations
    if family == "lexical":
        return bool(FAMILY_CUE["lexical"].search(c))
    return bool(FAMILY_CUE[family].match(c)) and not FAMILY_CUE["lexical"].search(c)


def structural_filter(g: dict, family: Optional[str] = None) -> Optional[str]:
    """Free, deterministic checks. Returns a reason string if the scenario must be discarded, else None."""
    need = ["question", "options", "answer_O", "context_O", "decisive", "context_P", "context_N", "cue_N", "context_S", "cue_S"]
    miss = [k for k in need if not g.get(k)]
    if miss:
        return f"missing fields {miss}"
    opts = g["options"]
    if not (isinstance(opts, list) and len(opts) == 2 and all(isinstance(o, str) for o in opts)):
        return "options malformed"
    if opts[0].strip().lower() == opts[1].strip().lower():
        return "identical options"
    if any(not (1 <= len(words(o)) <= 6) for o in opts):
        return "option length"
    if g["answer_O"] not in ("A", "B"):
        return "answer_O malformed"
    for name in ("question", "context_O", "context_P"):
        if CORE_NEG.search(g[name]):
            return f"negation marker in {name}"
    if any(CORE_NEG.search(o) for o in opts):
        return "negation marker in an option"
    if not g["question"].strip().endswith("?"):
        return "question without question mark"
    O, P, N, S = (g[k] for k in ("context_O", "context_P", "context_N", "context_S"))
    if any(len(words(x)) > 30 for x in (O, P, N, S)):
        return "context longer than 30 words"
    if g["decisive"].lower() not in O.lower():
        return "decisive words not in context_O"
    if g["decisive"].lower() not in S.lower():
        return "decisive words changed in context_S"
    for name, ctx, cue in (("N", N, g["cue_N"]), ("S", S, g["cue_S"])):
        if cue.lower() not in ctx.lower():
            return f"cue_{name} not found in context_{name}"
        if cue.lower() in O.lower():
            return f"cue_{name} already present in context_O"
    if family is not None:
        for name in ("N", "S"):
            if not cue_in_family(g[f"cue_{name}"], family):
                return f"cue_{name} {g[f'cue_{name}']!r} is not a {family} cue"
    dn = token_edit_distance(words(O), words(N))
    if not (1 <= dn <= 3):
        return f"context_N edit distance {dn} (must be 1-3 words)"
    if token_edit_distance(words(O), words(S)) > 8:
        return "context_S edit too large"
    if token_edit_distance(words(O), words(P)) < 2:
        return "context_P is not a paraphrase (too close to O)"
    for name, ctx in (("P", P), ("N", N)):
        if entities(ctx) - entities(O):
            return f"context_{name} introduces a new name or number"
    return None


def verify_arms(g: dict, call: Callable[[str, str, int], str], model: str) -> Tuple[Optional[str], dict]:
    """Blind verification: each arm is shown alone. Returns (reason or None, raw verdicts)."""
    y = g["answer_O"]
    flip = "B" if y == "A" else "A"
    expect = {"O": (y, False), "P": (y, False), "N": (flip, True), "S": (y, True)}
    verdicts = {}
    for arm, (ans, neg) in expect.items():
        ctx = g[f"context_{arm}"]
        v = extract_json(call(model, VERIFY_PROMPT.format(context=ctx, question=g["question"], option_a=g["options"][0],
                                                          option_b=g["options"][1]), 200)) or {}
        verdicts[arm] = v
        if v.get("answer") != ans:
            return f"arm {arm} read as {v.get('answer')!r}", verdicts
        if v.get("natural") is False:
            return f"arm {arm} judged unnatural", verdicts
        if bool(v.get("negation_present")) != neg:
            return f"arm {arm} negation_present={v.get('negation_present')!r}", verdicts
    return None, verdicts


class NearDuplicateIndex:
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.by_domain: Dict[str, List[set]] = defaultdict(list)
        self.exact: set = set()

    def is_duplicate(self, domain: str, context_o: str, options: Sequence[str]) -> bool:
        key = " ".join(words(context_o))
        if key in self.exact:
            return True
        cs = content_set(context_o) | content_set(" ".join(options))
        for other in self.by_domain[domain]:
            inter = len(cs & other)
            union = len(cs | other) or 1
            if inter / union >= self.threshold:
                return True
        return False

    def add(self, domain: str, context_o: str, options: Sequence[str]) -> None:
        self.exact.add(" ".join(words(context_o)))
        self.by_domain[domain].append(content_set(context_o) | content_set(" ".join(options)))


def process(c: Candidate, call: Callable[[str, str, int], str], gen_model: str, ver_model: str,
            is_duplicate: Optional[Callable[[str, str, Sequence[str]], bool]] = None) -> dict:
    fam = FAMILIES[c.family]
    prompt = GEN_PROMPT.format(domain=c.domain, situation=c.situation, banned=BANNED_TOPICS, family=c.family,
                               family_desc=fam["desc"], examples=fam["examples"])
    g = extract_json(call(gen_model, prompt, 1200))
    if not g:
        return {"stage": "generator", "reason": "unparseable output"}
    if not g.get("usable"):
        return {"stage": "generator", "reason": str(g.get("reason", "declined"))[:80]}
    bad = structural_filter(g, c.family)
    if bad:
        return {"stage": "filter", "reason": bad, "generated": g}
    if is_duplicate is not None and is_duplicate(c.domain, g["context_O"], g["options"]):
        return {"stage": "duplicate", "reason": "near-duplicate of a kept scenario", "generated": g}
    reason, verdicts = verify_arms(g, call, ver_model)
    if reason:
        return {"stage": "verifier", "reason": reason, "generated": g, "verdicts": verdicts}
    return {"stage": "kept", "generated": g, "verdicts": verdicts,
            "meta": {"uid": c.uid, "family": c.family, "domain": c.domain, "situation": c.situation}}


def assign_splits(kept: List[dict], seed: int, seen_fracs=(0.60, 0.15, 0.25)) -> Dict[str, str]:
    """Scenario-level split. Seen families are split train/dev/test within family; unseen families go to test."""
    rng = random.Random(seed)
    by_fam = defaultdict(list)
    for k in kept:
        by_fam[k["meta"]["family"]].append(k["meta"]["uid"])
    out = {}
    for fam, uids in by_fam.items():
        uids = sorted(uids)
        rng.shuffle(uids)
        if FAMILIES[fam]["seen"]:
            n = len(uids)
            n_tr, n_dv = int(round(seen_fracs[0] * n)), int(round(seen_fracs[1] * n))
            for i, u in enumerate(uids):
                out[u] = "train" if i < n_tr else ("dev" if i < n_tr + n_dv else "test")
        else:
            for u in uids:
                out[u] = "test"
    return out


def to_items(kept: List[dict], splits: Dict[str, str], seed: int) -> List[dict]:
    """One row per arm. The option order is randomized per scenario (seeded) so that labels are balanced."""
    rng = random.Random(seed + 1)
    rows = []
    for k in sorted(kept, key=lambda x: x["meta"]["uid"]):
        g, m = k["generated"], k["meta"]
        swap = rng.random() < 0.5
        options = [g["options"][1], g["options"][0]] if swap else list(g["options"])
        y = 0 if g["answer_O"] == "A" else 1
        if swap:
            y = 1 - y
        labels = {"O": y, "P": y, "N": 1 - y, "S": y}
        cues = {"O": "", "P": "", "N": g["cue_N"], "S": g["cue_S"]}
        for arm in ("O", "P", "N", "S"):
            rows.append({"scenario_id": m["uid"], "arm": arm, "item_id": f"{m['uid']}-{arm}", "family": m["family"],
                         "seen_family": FAMILIES[m["family"]]["seen"], "domain": m["domain"], "situation": m["situation"],
                         "split": splits[m["uid"]], "context": g[f"context_{arm}"], "question": g["question"],
                         "options": options, "label": labels[arm], "cue": cues[arm], "decisive": g["decisive"]})
    return rows


def prompt_hashes() -> Dict[str, str]:
    return {"GEN_PROMPT": hashlib.sha256(GEN_PROMPT.encode()).hexdigest(),
            "VERIFY_PROMPT": hashlib.sha256(VERIFY_PROMPT.encode()).hexdigest()}


# ================================================================================================ construction v2
# Introduced 3 October 2026 after the v1 run had examined 1,416 candidates and kept 98 (7%). The v1 record showed that
# most losses did not come from bad scenarios but from the pipeline: (a) 228 generator responses were empty or cut off
# because thinking consumed the 1,200-token limit; (b) two filter defects: the "cue already present in context_O" test
# used substring matching ("no" matched "notebook" and "afternoon"), and a name counted as new in P or N when it merely
# moved out of sentence-initial position or took a possessive ("Lena's"); (c) the N edit limit of three words excluded
# most natural quantifier and lexical negations ("has plenty of ink" -> "has no ink left"); (d) the verifier rejected
# clear items: its "does not settle it" wording made it answer "unclear" (for example "The tomatoes are ripe and deep
# red" -> pick or leave?), and its negation flag fired on words such as "only" and "empty" while missing explicit "not"
# on irrelevant details. v2 keeps GEN_PROMPT unchanged, fixes (b), widens the edit limits (N at most 5 words, S at most
# 10), replaces the verifier prompt (better-choice wording; no negation flag, since the deterministic filters already
# guarantee the declared cue in N and S and no core negation marker in O and P), and verifies every arm of every
# candidate under this single protocol, including the candidates generated in v1 (their v1 verdicts are archived,
# not reused).

VERIFY_PROMPT_V2 = """Read the context and decide which of the two options is the better thing to do. Use only the context.

CONTEXT: {context}
QUESTION: {question}
A) {option_a}
B) {option_b}

Return strict JSON only:
{{"answer": "A" | "B" | "unclear", "natural": true | false}}
- "answer": the option that the context gives a reason to prefer. Answer "unclear" only if the context gives no reason to prefer one option over the other, or supports both equally.
- "natural": false if the context is ungrammatical, odd, or contradicts itself."""

V2_SETTINGS = {"n_max_edit": 5, "s_max_edit": 10}


def has_word(text: str, phrase: str) -> bool:
    """Case-insensitive whole-word (or whole-phrase) occurrence; "no" does not match "notebook"."""
    p = (phrase or "").strip().lower()
    if not p:
        return False
    return re.search(r"(?<![a-z0-9'])" + re.escape(p) + r"(?![a-z0-9])", text.lower()) is not None


def _word_forms(s: str) -> set:
    out = set()
    for w in re.findall(r"[A-Za-z0-9'’]+", s):
        w = w.lower().replace("’", "'")
        out.add(w)
        out.add(re.sub(r"'s$", "", w))
    return out


def new_entities(ctx: str, ref: str) -> set:
    """Names (capitalized, not sentence-initial) and numbers in `ctx` whose lower-cased form, possessive stripped,
    occurs nowhere in `ref` (in any position)."""
    known = _word_forms(ref)
    norm_num = lambda x: x.replace(",", "").rstrip(".-")
    known_numbers = {norm_num(m) for m in re.findall(r"\d[\d,.\-]*", ref)}
    out, toks = set(), ctx.split()
    for i, t in enumerate(toks):
        w = t.strip('.,;:()"\'`!?').replace("’", "'")
        if not w:
            continue
        base = re.sub(r"'s$", "", w).lower()
        if re.fullmatch(r"\d[\d,.\-]*", w):
            if norm_num(w) not in known_numbers:
                out.add(w)
        elif i > 0 and w[:1].isupper() and not toks[i - 1].endswith(('.', '!', '?')) and base not in known:
            out.add(base)
    return out


def structural_filter_v2(g: dict, family: Optional[str] = None, n_max_edit: int = 5, s_max_edit: int = 10) -> Optional[str]:
    """v1 checks with the two defects fixed and the edit limits widened. Returns a reason string or None."""
    need = ["question", "options", "answer_O", "context_O", "decisive", "context_P", "context_N", "cue_N", "context_S", "cue_S"]
    miss = [k for k in need if not g.get(k)]
    if miss:
        return f"missing fields {miss}"
    opts = g["options"]
    if not (isinstance(opts, list) and len(opts) == 2 and all(isinstance(o, str) for o in opts)):
        return "options malformed"
    if opts[0].strip().lower() == opts[1].strip().lower():
        return "identical options"
    if any(not (1 <= len(words(o)) <= 6) for o in opts):
        return "option length"
    if g["answer_O"] not in ("A", "B"):
        return "answer_O malformed"
    for name in ("question", "context_O", "context_P"):
        if CORE_NEG.search(g[name]):
            return f"negation marker in {name}"
    if any(CORE_NEG.search(o) for o in opts):
        return "negation marker in an option"
    if not g["question"].strip().endswith("?"):
        return "question without question mark"
    O, P, N, S = (g[k] for k in ("context_O", "context_P", "context_N", "context_S"))
    if any(len(words(x)) > 30 for x in (O, P, N, S)):
        return "context longer than 30 words"
    if g["decisive"].lower() not in O.lower():
        return "decisive words not in context_O"
    if g["decisive"].lower() not in S.lower():
        return "decisive words changed in context_S"
    for name, ctx, cue in (("N", N, g["cue_N"]), ("S", S, g["cue_S"])):
        if cue.lower() not in ctx.lower():
            return f"cue_{name} not found in context_{name}"
        if has_word(O, cue):
            return f"cue_{name} already present in context_O"
    if family is not None:
        for name in ("N", "S"):
            if not cue_in_family(g[f"cue_{name}"], family):
                return f"cue_{name} {g[f'cue_{name}']!r} is not a {family} cue"
    dn = token_edit_distance(words(O), words(N))
    if not (1 <= dn <= n_max_edit):
        return f"context_N edit distance {dn} (must be 1-{n_max_edit} words)"
    if token_edit_distance(words(O), words(S)) > s_max_edit:
        return "context_S edit too large"
    if token_edit_distance(words(O), words(P)) < 2:
        return "context_P is not a paraphrase (too close to O)"
    for name, ctx in (("P", P), ("N", N)):
        if new_entities(ctx, O):
            return f"context_{name} introduces a new name or number"
    return None


def verify_arms_v2(g: dict, call: Callable[[str, str, int], str], model: str, max_tokens: int = 200) -> Tuple[Optional[str], dict]:
    """Blind v2 verification: each arm alone; expected answer and naturalness. Stops at the first failing arm."""
    y = g["answer_O"]
    flip = "B" if y == "A" else "A"
    expect = {"O": y, "P": y, "N": flip, "S": y}
    verdicts = {}
    for arm, ans in expect.items():
        v = extract_json(call(model, VERIFY_PROMPT_V2.format(context=g[f"context_{arm}"], question=g["question"],
                                                             option_a=g["options"][0], option_b=g["options"][1]), max_tokens)) or {}
        verdicts[arm] = v
        if v.get("answer") != ans:
            return f"arm {arm} read as {v.get('answer')!r}", verdicts
        if v.get("natural") is False:
            return f"arm {arm} judged unnatural", verdicts
    return None, verdicts


def check_and_verify_v2(g: dict, c: "Candidate", call: Callable[[str, str, int], str], ver_model: str,
                        is_duplicate: Optional[Callable[[str, str, Sequence[str]], bool]] = None, source: str = "v2") -> dict:
    meta = {"uid": c.uid, "family": c.family, "domain": c.domain, "situation": c.situation}
    bad = structural_filter_v2(g, c.family, V2_SETTINGS["n_max_edit"], V2_SETTINGS["s_max_edit"])
    if bad:
        return {"stage": "filter", "reason": bad, "generated": g, "meta": meta, "source": source}
    if is_duplicate is not None and is_duplicate(c.domain, g["context_O"], g["options"]):
        return {"stage": "duplicate", "reason": "near-duplicate of a kept scenario", "generated": g, "meta": meta, "source": source}
    reason, verdicts = verify_arms_v2(g, call, ver_model)
    if reason:
        return {"stage": "verifier", "reason": reason, "generated": g, "verdicts": verdicts, "meta": meta, "source": source}
    return {"stage": "kept", "generated": g, "verdicts": verdicts, "meta": meta, "source": source}


def process_v2(c: "Candidate", call_gen: Callable[[str, str, int], str], call_ver: Callable[[str, str, int], str],
               gen_model: str, ver_model: str, is_duplicate=None, gen_max_tokens: int = 4000) -> dict:
    """Generate one candidate with the unchanged GEN_PROMPT, then filter and verify under v2."""
    fam = FAMILIES[c.family]
    prompt = GEN_PROMPT.format(domain=c.domain, situation=c.situation, banned=BANNED_TOPICS, family=c.family,
                               family_desc=fam["desc"], examples=fam["examples"])
    meta = {"uid": c.uid, "family": c.family, "domain": c.domain, "situation": c.situation}
    try:
        text = call_gen(gen_model, prompt, gen_max_tokens)
    except ValueError as e:                 # the call wrapper raises ValueError for an empty or cut-off response
        return {"stage": "generator", "reason": str(e)[:80], "meta": meta, "source": "v2-generated"}
    g = extract_json(text)
    if not g:
        return {"stage": "generator", "reason": "unparseable output", "meta": meta, "source": "v2-generated"}
    if not g.get("usable"):
        return {"stage": "generator", "reason": str(g.get("reason", "declined"))[:80], "meta": meta, "source": "v2-generated"}
    return check_and_verify_v2(g, c, call_ver, ver_model, is_duplicate, source="v2-generated")


def prompt_hashes() -> Dict[str, str]:   # noqa: F811  (v2 adds the second verifier prompt)
    return {"GEN_PROMPT": hashlib.sha256(GEN_PROMPT.encode()).hexdigest(),
            "VERIFY_PROMPT": hashlib.sha256(VERIFY_PROMPT.encode()).hexdigest(),
            "VERIFY_PROMPT_V2": hashlib.sha256(VERIFY_PROMPT_V2.encode()).hexdigest()}
