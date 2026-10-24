"""CondaQA loading and the derived structures used by the study.

Source files are the publisher's JSON files (`condaqa_train.json`, `condaqa_dev.json`,
`condaqa_test.json`, GitHub repository AbhilashaRavichander/CondaQA, Apache-2.0).
The same rows are available from the Hugging Face Hub (lasha-nlp/CONDAQA); both are hashed
in the provenance record. Nothing here re-splits the data.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

LABELS: Tuple[str, ...] = ("YES", "NO", "DON'T KNOW")
LABEL_TO_INDEX = {l: i for i, l in enumerate(LABELS)}
EDIT_NAMES = {0: "original", 1: "paraphrase", 2: "scope", 3: "affirmative"}
SPLIT_FILES = {"train": "condaqa_train.json", "dev": "condaqa_dev.json", "test": "condaqa_test.json"}
ANSWER_SURFACE = {"YES": "yes", "NO": "no", "DON'T KNOW": "don't know"}


@dataclass(frozen=True)
class Row:
    split: str
    sample_id: str
    passage_id: str
    question_id: str
    edit_id: int
    passage: str
    question: str
    label: str
    cue: str
    original_sentence: str

    @property
    def group_key(self) -> Tuple[str, str]:
        return (self.passage_id, self.question_id)

    @property
    def eligible(self) -> bool:
        return self.label in LABEL_TO_INDEX

    @property
    def label_index(self) -> int:
        return LABEL_TO_INDEX[self.label]

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(text) -> str:
    return re.sub(r"\s+", " ", str(text if text is not None else "")).strip()


def _read_json_rows(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = list(data.values())
        return list(data)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_condaqa(data_dir: str | Path, splits: Iterable[str] = ("train", "dev", "test")) -> Dict[str, List[Row]]:
    """Load the publisher JSON files into normalized rows, keyed by split name."""
    data_dir = Path(data_dir)
    out: Dict[str, List[Row]] = {}
    for split in splits:
        rows = []
        for r in _read_json_rows(data_dir / SPLIT_FILES[split]):
            rows.append(Row(
                split=split,
                sample_id=str(r["SampleID"]),
                passage_id=str(r["PassageID"]),
                question_id=str(r["QuestionID"]),
                edit_id=int(r["PassageEditID"]),
                passage=_norm(r["sentence1"]),
                question=_norm(r["sentence2"]),
                label=_norm(r["label"]),
                cue=_norm(r.get("original cue", "")),
                original_sentence=_norm(r.get("original sentence", "")),
            ))
        out[split] = rows
    return out


def file_hashes(data_dir: str | Path) -> Dict[str, str]:
    data_dir = Path(data_dir)
    return {name: sha256_file(data_dir / name) for name in SPLIT_FILES.values()}


def eligible_rows(rows: Iterable[Row]) -> List[Row]:
    return [r for r in rows if r.eligible]


def groups(rows: Iterable[Row]) -> Dict[Tuple[str, str], Dict[int, Row]]:
    """(passage_id, question_id) -> {edit_id: row}. Duplicates (same key and edit) keep the first."""
    g: Dict[Tuple[str, str], Dict[int, Row]] = defaultdict(dict)
    for r in rows:
        g[r.group_key].setdefault(r.edit_id, r)
    return dict(g)


def complete_groups(rows: Iterable[Row]) -> Dict[Tuple[str, str], Dict[int, Row]]:
    return {k: v for k, v in groups(rows).items() if set(v) == {0, 1, 2, 3} and all(x.eligible for x in v.values())}


def contrast_pairs(rows: Iterable[Row], edit_id: int, label_flipping: Optional[bool] = None) -> List[Tuple[Row, Row]]:
    """Pairs (original, edited) with both rows eligible. `label_flipping` filters by whether labels differ."""
    pairs = []
    for g in groups(rows).values():
        if 0 in g and edit_id in g and g[0].eligible and g[edit_id].eligible:
            flips = g[0].label != g[edit_id].label
            if label_flipping is None or flips == label_flipping:
                pairs.append((g[0], g[edit_id]))
    return pairs


def cue_inventory(rows: Iterable[Row]) -> Dict[str, int]:
    inv: Dict[str, int] = defaultdict(int)
    for r in rows:
        if r.cue:
            inv[r.cue.lower()] += 1
    return dict(sorted(inv.items(), key=lambda kv: -kv[1]))


def unseen_cue_flags(train_rows: Iterable[Row], rows: Iterable[Row]) -> Dict[str, bool]:
    """sample_id -> True when the row's cue never occurs in the training split."""
    seen = {c for c in cue_inventory(train_rows)}
    return {r.sample_id: (r.cue.lower() not in seen) for r in rows}


def render_candidates(question: str = "", labels: Tuple[str, ...] = LABELS) -> List[str]:
    """Candidate texts are the bare answers (the question is part of the jointly encoded context, as in CACSA v2).
    The `question` argument is accepted for interface stability and ignored."""
    return [ANSWER_SURFACE[l] for l in labels]


def context_text(question: str, passage: str) -> str:
    """Jointly encoded context: question first (so that truncation never removes it), then the passage.
    CACSA v2 encoded context and question together; the order here protects the question under truncation."""
    return f"{question.strip()} {passage.strip()}"


_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def delete_cue(passage: str, cue: str) -> Tuple[str, int]:
    """Remove whole-word occurrences of `cue` (possibly multi-word) from `passage`.

    Used only for the exact Proposition-1 test under the context-independent encoder.
    Returns the edited passage and the number of removed occurrences.
    """
    cue = cue.strip()
    if not cue:
        return passage, 0
    pattern = re.compile(r"(?<!\w)" + re.escape(cue) + r"(?!\w)", re.IGNORECASE)
    edited, n = pattern.subn("", passage)
    edited = re.sub(r"\s+", " ", edited).strip()
    edited = re.sub(r"\s+([,.;:!?])", r"\1", edited)
    return edited, n


def summary(rows_by_split: Dict[str, List[Row]]) -> dict:
    out = {}
    for split, rows in rows_by_split.items():
        el = eligible_rows(rows)
        g = groups(rows)
        cg = complete_groups(rows)
        out[split] = {
            "rows": len(rows),
            "eligible_rows": len(el),
            "excluded_free_text_rows": len(rows) - len(el),
            "passages": len({r.passage_id for r in rows}),
            "groups": len(g),
            "complete_eligible_groups": len(cg),
            "labels": {l: sum(r.label == l for r in rows) for l in LABELS},
            "pairs_original_vs_paraphrase": len(contrast_pairs(rows, 1)),
            "pairs_original_vs_scope": len(contrast_pairs(rows, 2)),
            "pairs_original_vs_affirmative": len(contrast_pairs(rows, 3)),
            "label_flipping_affirmative_pairs": len(contrast_pairs(rows, 3, label_flipping=True)),
            "label_flipping_scope_pairs": len(contrast_pairs(rows, 2, label_flipping=True)),
            "distinct_cues": len(cue_inventory(rows)),
        }
    return out
