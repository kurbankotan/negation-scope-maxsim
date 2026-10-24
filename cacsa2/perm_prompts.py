"""Prompts of the v3.0 data build (frozen with manifest v3.0, Appendices A and B).

PERM_GEN_PROMPT      generator prompt of NegPairs-Perm (placeholders: domain, situation, banned, first, family,
                     pattern_desc, negated_form_desc; doubled braces are literal braces after formatting)
PERM_PATTERNS        the four pattern descriptions inserted into PERM_GEN_PROMPT
QWEN_VERIFY_PROMPT   cross-family verifier prompt (placeholders: context, question, option_a, option_b)

The SHA-256 values printed by `prompt_hashes()` are listed in the manifest; the build notebook recomputes them and
stops if any differs.
"""
from __future__ import annotations

import hashlib
from typing import Dict

PERM_GEN_PROMPT = """You are writing ONE test item for a study of how reading models handle the scope of negation.

Write a short, realistic everyday scenario in the domain "{domain}" about {situation}.
Do NOT use any of these topics: {banned}.

The context is ONE sentence (at most 25 words) made of two clauses joined by a connector such as "and", ", and", "while" or "but":
- the DECISIVE clause states a fact that makes exactly one of two options the better choice;
- the OTHER clause states a detail that does not matter for the choice.
Put the {first} clause first.

Negation pattern for this item ({family}): {pattern_desc}
Mark the two edit sites, one in each clause, exactly as described. The first word of the sentence must not be an edit site. Write a marked word in lower case and in exactly the same form at both sites, with no punctuation attached to it.
Applying the pattern at the site in the DECISIVE clause must make the OTHER option the better choice.
Applying the pattern at the site in the OTHER clause must leave the better choice unchanged.
Both edited sentences must be natural English.

Return these fields:
- question: one short question asking what someone should do. It must not reveal the answer and must contain no negation.
- options: exactly two short action phrases (2 to 5 words each). Both must be sensible actions in general; neither may contain a negation word.
- answer: "A" if the first option is the better choice for the unmarked context, "B" if the second.
- context_marked: the sentence with the two edit sites marked.
- connector: the exact word or words joining the two clauses, as written in context_marked.
- decisive: the exact words of the decisive clause that decide the answer, as written in context_marked (without marks).
- negated_form: {negated_form_desc}
- context_P: a paraphrase of the unmarked context: same meaning and same answer, clearly different wording.

The unmarked context, context_P, the question and the options must contain NO negation of any kind (no not, n't, no, never, none, nothing, nobody, without, lacks, fails, un-/in-/dis-/non- words meaning "not", -less words, and so on).

If you cannot satisfy every constraint naturally, return {{"usable": false, "reason": "<= 8 words"}}.

Output strict JSON only:
{{"usable": true, "question": "...", "options": ["...", "..."], "answer": "A", "context_marked": "...", "connector": "...", "decisive": "...", "negated_form": "...", "context_P": "..."}}"""

PERM_PATTERNS: Dict[str, Dict[str, str]] = {
    "syntactic": {
        "pattern_desc": ('the word "not" is inserted directly after an auxiliary or copula verb (is, are, was, were, can, '
                         'will, should, has, have). Each clause must contain such a verb; mark the insertion point after '
                         'it with [[^]]. Format example: "the X is [[^]] ready, and the Y is [[^]] warm".'),
        "negated_form_desc": 'the word "not".',
    },
    "quantifier": {
        "pattern_desc": ('the determiner "a", "an" or "some" is replaced by "no". Both clauses must use the SAME '
                         'determiner; mark it in each clause with double brackets. Format example: "there is [[a]] X at '
                         'the door, and there is [[a]] Y in the hall".'),
        "negated_form_desc": 'the word "no".',
    },
    "affixal": {
        "pattern_desc": ('the SAME adjective or participle W occurs in both clauses and is replaced by its negated form '
                         'with a prefix un-, in-, im-, il-, ir-, dis- or non- written without a hyphen (for example '
                         'available becomes unavailable, accessible becomes inaccessible, connected becomes '
                         'disconnected). W must not follow "a" or "an". Mark W in each clause with double brackets. '
                         'Format example: "the X is [[available]] today, and the Y is [[available]] too".'),
        "negated_form_desc": 'the prefixed word that replaces W (for example "unavailable").',
    },
    "lexical": {
        "pattern_desc": ('the SAME word occurs in both clauses and is replaced by its lexical negation: has becomes '
                         'lacks, have becomes lack, had becomes lacked, with becomes without, present becomes absent. '
                         'Mark the word in each clause with double brackets. Format example: "the X [[has]] a key, and '
                         'the Y [[has]] a lamp".'),
        "negated_form_desc": "the replacing word (lacks, lack, lacked, without or absent).",
    },
}

QWEN_VERIFY_PROMPT = """Read the context and decide which of the two options is the better thing to do. Use only the context.

CONTEXT: {context}
QUESTION: {question}
A) {option_a}
B) {option_b}

Answer with one letter: A or B for the option that the context gives a reason to prefer, or U if the context gives no reason to prefer one option over the other or supports both equally."""


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def prompt_hashes() -> Dict[str, str]:
    out = {"PERM_GEN_PROMPT": _h(PERM_GEN_PROMPT), "QWEN_VERIFY_PROMPT": _h(QWEN_VERIFY_PROMPT)}
    for fam, d in PERM_PATTERNS.items():
        for k, v in d.items():
            out[f"PERM_PATTERNS.{fam}.{k}"] = _h(v)
    return out


if __name__ == "__main__":
    for k, v in prompt_hashes().items():
        print(k, v)
