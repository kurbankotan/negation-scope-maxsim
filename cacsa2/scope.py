"""Negation-cue lexicon and a deterministic dependency-based scope resolver (used by aggregator A4).

The resolver is deliberately simple and fully specified here so that it can be released and
re-run. It is a heuristic resource, not a trained negation-scope model; its cue recall against the
cue annotated by CondaQA is measured and reported (see `cue_recall`).

Cue sources (all three are documented in the paper):
  1. a core list of syntactic and lexical negators (below),
  2. optionally, the cue inventory of the CondaQA *training* split (never dev/test),
  3. an affixal rule: prefixes un-/in-/im-/il-/ir-/dis-/non- and suffix -less, accepted only when the
     stripped base is a frequent English word (wordfreq Zipf frequency >= `base_zipf_min`) and the
     word is not on the false-positive stoplist.

Scope rules operate on a spaCy dependency parse (model and version pinned in the run environment):
  * adverbial/syntactic negator attached to a predicate (dep `neg`, or ADV cue such as never/rarely):
        scope = subtree(predicate) minus the subject subtree minus the cue;
  * determiner / pronoun cue (no, none, nobody, nothing, neither, few ...):
        scope = subtree(nearest verbal ancestor) minus the cue, minus the subject subtree unless the
        cue sits inside the subject;
  * prepositional cue (without, except ...): scope = subtree(cue) minus the cue (the PP object);
  * lexical / affixal cue:
        ADJ attributive (amod): scope = head-noun subtree, cue included;
        ADJ predicative (acomp/attr/oprd): scope = subject subtree of the copula plus the cue;
        VERB: scope = subtree(cue) minus the subject subtree, cue included;
        NOUN: scope = subtree(cue), cue included.
  * tokens inside an odd number of scopes are marked negated (parity rule).
Multiword exceptions that are not negation ("not only", "no doubt", "not just") are skipped.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

CORE_SYNTACTIC = {
    "not", "n't", "no", "never", "nor", "neither", "none", "nobody", "nothing", "nowhere", "cannot",
    "without", "except", "nope", "nay",
}
CORE_LEXICAL = {
    # adverbs of (near) negation
    "rarely", "seldom", "hardly", "barely", "scarcely",
    # verbs
    "lack", "lacks", "lacked", "lacking", "fail", "fails", "failed", "failing", "refuse", "refused",
    "refuses", "deny", "denies", "denied", "prevent", "prevents", "prevented", "preventing", "cease",
    "ceased", "stop", "stopped", "avoid", "avoided", "avoids", "exclude", "excluded", "excludes",
    "forbid", "forbade", "forbidden", "prohibit", "prohibited", "prohibits",
    # nouns
    "absence", "lack", "failure", "refusal", "denial", "prohibition",
    # adjectives
    "absent", "unable", "unknown", "unclear", "unlike", "unlikely", "unsuccessful", "impossible",
    "insufficient", "incapable", "unaware", "unavailable",
    # determiners / quantifiers used as negation cues
    "few", "little",
}
AFFIX_PREFIXES = ("un", "in", "im", "il", "ir", "dis", "non")
AFFIX_SUFFIXES = ("less",)
# frequent words that match the affix pattern but are not negations
AFFIX_STOPLIST = {
    "display", "discover", "disease", "distance", "discuss", "dish", "indeed", "inform", "install",
    "insist", "invite", "involve", "inside", "instead", "instance", "income", "increase", "industry",
    "input", "insert", "insure", "intend", "interest", "invest", "island", "issue", "image", "impact",
    "import", "improve", "impress", "imply", "illustrate", "irony", "union", "unit", "unite", "united",
    "universe", "university", "unless", "until", "under", "undergo", "uniform", "unique", "nonetheless",
    "discount", "district", "distinct", "disturb", "disaster", "discipline", "dismiss", "dispute",
    "disk", "disc", "disco", "disguise", "dispatch", "dispose", "distress", "distribute", "disposal",
    "information", "inflation", "influence", "injury", "innocent", "insect", "inspire", "inspect",
    "instrument", "insurance", "intense", "interview", "introduce", "invent", "investigate", "invoke",
    "unlock", "unfold", "unveil", "unleash", "unite", "unity", "universal", "regardless", "nevertheless",
    "wireless", "homeless", "harmless", "endless", "countless", "careless", "needless",  # see note below
    "indian", "india", "indiana", "indicate", "indication", "individual", "indoor", "inner", "innate",
    "imperial", "immune", "immigrant", "image", "imagine", "impose", "impulse", "inherit", "initial",
    "irish", "iron", "ironic", "irrigate", "illinois", "illustrious", "illuminate", "nonprofit", "nonsense",
    "disorder", "dissolve", "distill", "disturbance", "dismal",
}
# Note on -less words: the stoplist removes "regardless" and "nevertheless" (not negations) and, as a conservative choice
# made for CondaQA, also seven frequent -less words (wireless, homeless, harmless, endless, countless, careless,
# needless). These seven are therefore never detected as cues. The list is frozen as released (0.1.x) and was not
# revised after the NegPairs data were generated; its detection rate is reported per negation family.
NON_NEGATION_BIGRAMS = {("not", "only"), ("no", "doubt"), ("not", "just"), ("no", "wonder"), ("not", "merely")}
# multiword cues: the first token carries the cue; scope is taken from the last token's attachment
MULTIWORD_CUES = (("with", "the", "exception", "of"), ("free", "of"), ("free", "from"), ("no", "longer"),
                  ("far", "from"), ("by", "no", "means"), ("not", "at", "all"), ("devoid", "of"))

SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
# dependents whose subtrees are never inside the scope of a clause-level negator
CLAUSE_EXCLUDED_DEPS = {"conj", "cc", "advcl", "punct", "parataxis", "dep"}
PREDICATIVE_DEPS = {"acomp", "attr", "oprd"}


@dataclass
class CueLexicon:
    core_syntactic: Set[str] = field(default_factory=lambda: set(CORE_SYNTACTIC))
    core_lexical: Set[str] = field(default_factory=lambda: set(CORE_LEXICAL))
    extra_cues: Set[str] = field(default_factory=set)       # e.g. CondaQA training cues (lower-cased)
    affixal: bool = True
    base_zipf_min: float = 3.5
    stoplist: Set[str] = field(default_factory=lambda: set(AFFIX_STOPLIST))

    def __post_init__(self):
        self._zipf = None
        if self.affixal:
            try:
                from wordfreq import zipf_frequency
                self._zipf = zipf_frequency
            except Exception:  # pragma: no cover
                self._zipf = None

    def affixal_match(self, word: str) -> bool:
        w = word.lower()
        if not w.isalpha() or len(w) < 5 or w in self.stoplist or self._zipf is None:
            return False
        for p in AFFIX_PREFIXES:
            if w.startswith(p) and len(w) - len(p) >= 3:
                base = w[len(p):]
                if self._zipf(base, "en") >= self.base_zipf_min and self._zipf(w, "en") > 0:
                    return True
        for s in AFFIX_SUFFIXES:
            if w.endswith(s) and len(w) - len(s) >= 3:
                base = w[: -len(s)]
                if self._zipf(base, "en") >= self.base_zipf_min:
                    return True
        return False

    def kind(self, token_text: str, lemma: str) -> Optional[str]:
        """Return 'syntactic', 'lexical', 'extra', 'affixal' or None."""
        w = token_text.lower()
        l = lemma.lower()
        if w in self.core_syntactic or l in self.core_syntactic:
            return "syntactic"
        if w in self.core_lexical or l in self.core_lexical:
            return "lexical"
        if w in self.extra_cues or l in self.extra_cues:
            return "extra"
        if self.affixal and (self.affixal_match(w) or self.affixal_match(l)):
            return "affixal"
        return None


@dataclass
class ScopeResult:
    cues: List[Tuple[int, int, str, str]]        # (start_char, end_char, text, kind)
    negated_spans: List[Tuple[int, int]]         # character spans of words inside a negation scope
    n_words: int
    cue_sentences: List[Tuple[int, int]] = field(default_factory=list)   # character spans of sentences containing a cue
    sentences: List[Tuple[int, int]] = field(default_factory=list)       # character spans of all sentences (spaCy)

    def word_in_scope(self, start: int, end: int) -> bool:
        return any(s < end and start < e for s, e in self.negated_spans)


class ScopeResolver:
    def __init__(self, lexicon: Optional[CueLexicon] = None, spacy_model: str = "en_core_web_sm", nlp=None,
                 combine: str = "union"):
        """combine: 'union' (a token is negated if inside any scope; default) or 'parity' (odd number of scopes)."""
        if combine not in ("union", "parity"):
            raise ValueError(combine)
        self.combine = combine
        self.lexicon = lexicon or CueLexicon()
        if nlp is None:
            import spacy
            nlp = spacy.load(spacy_model, disable=["ner"])
        self.nlp = nlp
        self.spacy_model = spacy_model
        self.spacy_version = getattr(nlp, "meta", {}).get("version", "?")

    # -- helpers -------------------------------------------------------------------------------
    @staticmethod
    def _subtree(tok) -> Set[int]:
        return {t.i for t in tok.subtree}

    @staticmethod
    def _subject_subtree(pred) -> Set[int]:
        out: Set[int] = set()
        for ch in pred.children:
            if ch.dep_ in SUBJECT_DEPS:
                out |= {t.i for t in ch.subtree}
        return out

    @classmethod
    def _clause_scope(cls, pred, keep_subject: bool = False) -> Set[int]:
        """Predicate subtree without coordinated/adverbial clauses and punctuation, optionally without the subject."""
        scope = cls._subtree(pred)
        for ch in pred.children:
            if ch.dep_ in CLAUSE_EXCLUDED_DEPS:
                scope -= cls._subtree(ch)
            if not keep_subject and ch.dep_ in SUBJECT_DEPS:
                scope -= cls._subtree(ch)
        return {i for i in scope if not pred.doc[i].is_punct}

    @staticmethod
    def _verbal_ancestor(tok):
        """Nearest VERB/AUX on the path from `tok` to its sentence root (inclusive), or None.

        spaCy creates a new Token object on every access, so the root must be detected by index
        (`head.i == i`), never by identity. The loop is additionally capped at the document length.
        """
        cur = tok
        for _ in range(len(tok.doc) + 1):
            if cur.pos_ in ("VERB", "AUX"):
                return cur
            head = cur.head
            if head.i == cur.i:          # sentence root reached
                return None
            cur = head
        return None

    def _scope_for(self, tok, kind: str) -> Set[int]:
        i = tok.i
        # syntactic negator attached as `neg`, or adverbial cue (never, rarely, seldom ...)
        if tok.dep_ == "neg" or (tok.pos_ in ("ADV", "PART") and kind in ("syntactic", "lexical", "extra")):
            pred = tok.head
            if pred.pos_ not in ("VERB", "AUX", "ADJ", "NOUN"):
                anc = self._verbal_ancestor(tok)
                pred = anc or pred
            return self._clause_scope(pred) - {i}
        if tok.pos_ == "ADP" or tok.dep_ == "prep":                      # without / except
            return {j for j in self._subtree(tok) - {i} if not tok.doc[j].is_punct}
        if tok.pos_ in ("DET", "PRON") or (tok.dep_ in ("det", "nsubj", "nsubjpass", "dobj", "pobj") and kind == "syntactic"):
            anc = self._verbal_ancestor(tok)
            if anc is None:
                return {j for j in self._subtree(tok.head) - {i} if not tok.doc[j].is_punct}
            in_subject = i in self._subject_subtree(anc)
            return self._clause_scope(anc, keep_subject=in_subject) - {i}
        if tok.pos_ == "ADJ":
            if tok.dep_ == "amod":
                return {j for j in self._subtree(tok.head) if not tok.doc[j].is_punct}      # cue included
            if tok.dep_ in PREDICATIVE_DEPS:
                return {j for j in (self._subject_subtree(tok.head) | self._subtree(tok)) if not tok.doc[j].is_punct}
            return {j for j in self._subtree(tok) if not tok.doc[j].is_punct}
        if tok.pos_ == "VERB":
            return self._clause_scope(tok)                                  # cue included, subject excluded
        if tok.pos_ in ("NOUN", "PROPN"):
            return {j for j in self._subtree(tok) if not tok.doc[j].is_punct}
        anc = self._verbal_ancestor(tok)
        if anc is not None:
            return self._clause_scope(anc) - {i}
        return {j for j in self._subtree(tok.head) - {i} if not tok.doc[j].is_punct}

    # -- public --------------------------------------------------------------------------------
    def _resolve_doc(self, doc) -> ScopeResult:
        counts = [0] * len(doc)
        cues: List[Tuple[int, int, str, str]] = []
        skip: Set[int] = set()
        lower = [t.text.lower() for t in doc]
        for k in range(len(doc) - 1):
            if (lower[k], lower[k + 1]) in NON_NEGATION_BIGRAMS:
                skip.add(k)
        # multiword cues
        for mw in MULTIWORD_CUES:
            L = len(mw)
            for start in range(len(doc) - L + 1):
                if tuple(lower[start:start + L]) == mw:
                    first, last = doc[start], doc[start + L - 1]
                    cues.append((first.idx, last.idx + len(last.text), doc[start:start + L].text, "multiword"))
                    scope = {j for j in self._subtree(last) if not doc[j].is_punct} - set(range(start, start + L))
                    if not scope:
                        anc = self._verbal_ancestor(last)
                        scope = (self._clause_scope(anc) if anc is not None else set()) - set(range(start, start + L))
                    for j in scope:
                        counts[j] += 1
                    skip.update(range(start, start + L))
        for tok in doc:
            if tok.i in skip:
                continue
            kind = self.lexicon.kind(tok.text, tok.lemma_)
            if kind is None:
                continue
            cues.append((tok.idx, tok.idx + len(tok.text), tok.text, kind))
            for j in self._scope_for(tok, kind):
                counts[j] += 1
        if self.combine == "parity":
            marked = [j for j in range(len(doc)) if counts[j] % 2 == 1]
        else:
            marked = [j for j in range(len(doc)) if counts[j] > 0]
        negated = [(doc[j].idx, doc[j].idx + len(doc[j].text)) for j in marked]
        cue_sents, sents = [], []
        for sent in doc.sents:
            sents.append((sent.start_char, sent.end_char))
            if any(sent.start_char <= c[0] < sent.end_char for c in cues):
                cue_sents.append((sent.start_char, sent.end_char))
        return ScopeResult(cues=cues, negated_spans=negated, n_words=len(doc), cue_sentences=cue_sents, sentences=sents)

    def resolve(self, text: str) -> ScopeResult:
        """Scope analysis of one text; results are cached by text."""
        cache = self.__dict__.setdefault("_cache", {})
        res = cache.get(text)
        if res is None:
            res = self._resolve_doc(self.nlp(text))
            cache[text] = res
        return res

    def prefetch(self, texts: Sequence[str], batch_size: int = 128, progress: bool = False) -> int:
        """Parse all not-yet-cached texts with spaCy's batched `nlp.pipe` (much faster than one call per text)."""
        cache = self.__dict__.setdefault("_cache", {})
        todo = [t for t in dict.fromkeys(texts) if t not in cache]
        t0 = time.time()
        for k, (text, doc) in enumerate(zip(todo, self.nlp.pipe(todo, batch_size=batch_size)), start=1):
            cache[text] = self._resolve_doc(doc)
            if progress and (k % 1000 == 0 or k == len(todo)):
                print(f"  scope: {k}/{len(todo)} texts parsed ({time.time() - t0:.0f}s)", flush=True)
        return len(todo)

    def polarity_mask(self, text: str, token_offsets: Sequence[Tuple[int, int]], mode: str = "scope") -> List[int]:
        """1 for encoder tokens whose character span overlaps a negated region, else 0.

        mode="scope": the dependency-based scope (aggregator A4);
        mode="sentence": every token of any sentence that contains a detected cue (scope-blind control A6)."""
        res = self.resolve(text)
        spans = res.negated_spans if mode == "scope" else res.cue_sentences if mode == "sentence" else None
        if spans is None:
            raise ValueError(mode)
        if not spans:
            return [0] * len(token_offsets)
        mark = bytearray(len(text) + 1)
        for a, b in spans:
            mark[a:b] = b"\x01" * (b - a)
        return [1 if (e > s and any(mark[s:e])) else 0 for (s, e) in token_offsets]


def cue_recall(resolver: ScopeResolver, passages_with_cues: Iterable[Tuple[str, str]], progress: bool = False) -> dict:
    """Fraction of passages in which the annotated cue string overlaps a detected cue.

    `passages_with_cues` yields (passage_text, cue_string). Only the first occurrence of the
    cue string in the passage is used, matched case-insensitively on word boundaries.
    """
    pairs = []
    for passage, cue in passages_with_cues:
        cue = cue.strip()
        if not cue:
            continue
        m = re.search(r"(?<!\w)" + re.escape(cue) + r"(?!\w)", passage, re.IGNORECASE)
        if m:
            pairs.append((passage, cue, m.start(), m.end()))
    resolver.prefetch([p for p, _, _, _ in pairs], progress=progress)
    n = hit = 0
    missed: Dict[str, int] = {}
    for passage, cue, a, b in pairs:
        n += 1
        res = resolver.resolve(passage)
        if any(s < b and a < e for (s, e, _, _) in res.cues):
            hit += 1
        else:
            missed[cue.lower()] = missed.get(cue.lower(), 0) + 1
    return {"n": n, "detected": hit, "recall": (hit / n if n else None),
            "missed_cues": dict(sorted(missed.items(), key=lambda kv: -kv[1])[:40])}
