"""Encoder packs: everything a training worker needs for one encoder, written once by the main process.

A pack is the encoder's token-state store (fp16) plus a record file with the rows of every split and the polarity
masks: dependency scope and sentence-wide masks per text (keyed by the store's text hash), the dataset-supplied
oracle mask per item. Workers rebuild `Item`s from it without spaCy or any encoder model.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .encoders import EncodedStore, text_key
from .negpairs import LABELS2, NegRow, oracle_mask
from .scope import ScopeResolver
from .train import Item

MASK_MODES = ("scope", "sentence", "oracle")


def row_to_dict(r: NegRow) -> dict:
    d = asdict(r)
    d["options"] = list(r.options)
    return d


def row_from_dict(d: dict) -> NegRow:
    d = dict(d)
    d["options"] = tuple(d["options"])
    return NegRow(**d)


def build_masks(rows: Sequence[NegRow], store: EncodedStore, resolver: Optional[ScopeResolver]) -> dict:
    masks = {"scope": {}, "sentence": {}, "oracle": {}}
    texts = sorted({r.text for r in rows})
    if resolver is not None:
        resolver.prefetch(texts)
    for t in texts:
        offs = store.get(t).offsets
        if resolver is not None:
            masks["scope"][text_key(t)] = resolver.polarity_mask(t, offs, mode="scope")
            masks["sentence"][text_key(t)] = resolver.polarity_mask(t, offs, mode="sentence")
        else:
            masks["scope"][text_key(t)] = [0] * len(offs)
            masks["sentence"][text_key(t)] = [0] * len(offs)
    for r in rows:
        masks["oracle"][r.sample_id] = oracle_mask(r, store.get(r.text).offsets)
    return masks


def save_pack(path_prefix: str | Path, store: EncodedStore, rows_by_split: Dict[str, Sequence[NegRow]],
              resolver: Optional[ScopeResolver]) -> dict:
    """Writes <prefix>_store.pt and <prefix>_records.json; returns the record (also kept in memory by the caller)."""
    all_rows = [r for rows in rows_by_split.values() for r in rows]
    store.save(f"{path_prefix}_store.pt")
    rec = {"dim": store.encoder.dim, "rows": {s: [row_to_dict(r) for r in rows] for s, rows in rows_by_split.items()},
           "masks": build_masks(all_rows, store, resolver)}
    Path(f"{path_prefix}_records.json").write_text(json.dumps(rec))
    return rec


class Pack:
    def __init__(self, path_prefix: str | Path):
        self.prefix = str(path_prefix)
        self.store = EncodedStore(encoder=None)
        self.store.load(f"{self.prefix}_store.pt")
        rec = json.loads(Path(f"{self.prefix}_records.json").read_text())
        self.dim = int(rec["dim"])
        self.rows = {s: [row_from_dict(d) for d in rows] for s, rows in rec["rows"].items()}
        self.masks = rec["masks"]

    def items(self, split: str, mode: str = "scope") -> List[Item]:
        if mode not in MASK_MODES:
            raise ValueError(mode)
        out = []
        for r in self.rows[split]:
            ctx = self.store.get(r.text)
            if mode == "oracle":
                pol = self.masks["oracle"][r.sample_id]
            else:
                pol = self.masks[mode][text_key(r.text)]
            out.append(Item(row=r, ctx=ctx.states, cands=[self.store.get(o).states for o in r.options],
                            polarity=torch.tensor(pol, dtype=torch.uint8), label=LABELS2.index(r.label),
                            unseen_cue=not r.seen_family))
        return out

    def zero_mask_items(self, split: str) -> List[Item]:
        """Items with an all-zero polarity mask (unmasked rules; exact checks)."""
        out = []
        for it in self.items(split, "scope"):
            out.append(Item(row=it.row, ctx=it.ctx, cands=it.cands, polarity=torch.zeros_like(it.polarity), label=it.label,
                            unseen_cue=it.unseen_cue))
        return out
