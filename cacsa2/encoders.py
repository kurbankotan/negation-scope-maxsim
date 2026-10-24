"""Frozen token encoders.

E0 `HashEncoder`     context-independent: every token type maps to a fixed unit vector. This is the
                     regime of Propositions 1-2 and exists to test them exactly.
E1/E2 `HFFrozenEncoder` frozen Hugging Face encoders (MiniLM, DeBERTa-v3-base); token states are the
                     last hidden layer without special tokens; computed once and cached.

Every encoder returns character offsets so that scope masks computed on words can be aligned to
encoder tokens.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class Encoded:
    states: torch.Tensor                    # (n, d)
    tokens: List[str]
    offsets: List[Tuple[int, int]]          # character spans in the source text
    truncated: bool = False

    def __len__(self) -> int:
        return self.states.shape[0]


def word_tokenize(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    toks, offs = [], []
    for m in _WORD_RE.finditer(text):
        toks.append(m.group(0))
        offs.append((m.start(), m.end()))
    return toks, offs


class HashEncoder:
    """Deterministic context-independent embeddings (lower-cased word types)."""

    name = "E0-hash"
    context_independent = True

    def __init__(self, dim: int = 384, salt: str = "cacsa2-e0", max_tokens: Optional[int] = 512):
        self.dim = dim
        self.salt = salt
        self.max_tokens = max_tokens
        self._table: Dict[str, torch.Tensor] = {}

    def _vector(self, token: str) -> torch.Tensor:
        key = token.lower()
        v = self._table.get(key)
        if v is None:
            seed = int(hashlib.sha256((self.salt + "\x00" + key).encode("utf-8")).hexdigest()[:16], 16) % (2 ** 63)
            g = torch.Generator().manual_seed(seed)
            v = torch.randn(self.dim, generator=g)
            v = v / v.norm()
            self._table[key] = v
        return v

    def encode(self, text: str) -> Encoded:
        toks, offs = word_tokenize(text)
        truncated = self.max_tokens is not None and len(toks) > self.max_tokens
        if self.max_tokens is not None:
            toks, offs = toks[: self.max_tokens], offs[: self.max_tokens]
        if not toks:
            toks, offs = ["<empty>"], [(0, 0)]
        states = torch.stack([self._vector(t) for t in toks])
        return Encoded(states=states, tokens=toks, offsets=offs, truncated=truncated)

    def encode_batch(self, texts: Sequence[str], **kwargs) -> List[Encoded]:
        return [self.encode(t) for t in texts]


class HFFrozenEncoder:
    """Frozen transformer encoder; token states of the last layer, special tokens removed."""

    context_independent = False

    def __init__(self, model_name: str, max_length: int = 512, device: Optional[str] = None,
                 dtype: torch.dtype = torch.float32, name: Optional[str] = None, revision: Optional[str] = None,
                 normalize: bool = False):
        """normalize=True L2-normalizes every token state, as the archived CACSA v2 notebook did."""
        from transformers import AutoModel, AutoTokenizer  # imported lazily: not needed for E0 runs

        self.model_name = model_name
        self.normalize = normalize
        self.revision = revision
        self.name = name or model_name.split("/")[-1]
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
        try:
            model = AutoModel.from_pretrained(model_name, revision=revision, dtype=dtype)
        except TypeError:  # older transformers versions only know `torch_dtype`
            model = AutoModel.from_pretrained(model_name, revision=revision, torch_dtype=dtype)
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode_batch(self, texts: Sequence[str], max_length: Optional[int] = None) -> List[Encoded]:
        max_length = max_length or self.max_length
        enc = self.tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True,
                             max_length=max_length, return_offsets_mapping=True,
                             return_special_tokens_mask=True)
        offsets = enc.pop("offset_mapping")
        special = enc.pop("special_tokens_mask").bool()
        attn = enc["attention_mask"].bool()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        hidden = self.model(**enc).last_hidden_state.float().cpu()
        out = []
        for b in range(hidden.shape[0]):
            keep = attn[b] & ~special[b]
            ids = enc["input_ids"][b].cpu()
            toks = self.tokenizer.convert_ids_to_tokens(ids[keep].tolist())
            offs = [tuple(int(x) for x in o) for o in offsets[b][keep].tolist()]
            n_real = int(attn[b].sum())
            truncated = n_real >= max_length
            states = hidden[b][keep]
            if self.normalize and states.shape[0] > 0:
                states = torch.nn.functional.normalize(states, dim=-1)
            if states.shape[0] == 0:  # defensive: empty text
                states = torch.zeros(1, self.dim)
                toks, offs = ["<empty>"], [(0, 0)]
            out.append(Encoded(states=states, tokens=toks, offsets=offs, truncated=truncated))
        return out

    def encode(self, text: str) -> Encoded:
        return self.encode_batch([text])[0]


class GloveEncoder:
    """E3: context-independent GloVe vectors (glove.6B.300d; Pennington et al., 2014), looked up by the lower-cased
    tokens of the E0 tokenizer and L2-normalized. A token outside the GloVe vocabulary gets an E0-style hash vector
    in `dim` dimensions with salt `cacsa2-e3-oov` (counted in `oov`). `table` maps lower-cased token -> list of floats;
    `load_subset` reads only the needed tokens from the 822 MB zip of the Hugging Face repository `stanfordnlp/glove`."""

    name = "E3-glove"
    context_independent = True

    def __init__(self, table: Optional[Dict[str, Sequence[float]]] = None, dim: int = 300, max_tokens: Optional[int] = 512,
                 oov_salt: str = "cacsa2-e3-oov", vocab: Optional[Sequence[str]] = None, vectors: Optional[torch.Tensor] = None):
        """Either `table` (token -> vector) or `vocab` with the (V, dim) matrix `vectors`."""
        self.dim = dim
        self.max_tokens = max_tokens
        self._oov = HashEncoder(dim=dim, salt=oov_salt, max_tokens=None)
        if vectors is None:
            table = table or {}
            vocab = list(table)
            vectors = (torch.tensor([list(table[w]) for w in vocab], dtype=torch.float32) if vocab
                       else torch.zeros(0, dim))
        self._index: Dict[str, int] = {w.lower(): i for i, w in enumerate(vocab)}
        self._matrix = torch.nn.functional.normalize(vectors.float(), dim=-1)
        self.vocab_size = len(self._index)
        self.oov: Dict[str, int] = {}
        self.n_lookups = 0

    @staticmethod
    def load_subset(zip_path: str | Path, tokens: Iterable[str], member: str = "glove.6B.300d.txt") -> Dict[str, List[float]]:
        """Vectors of the given lower-cased tokens from `member` inside glove.6B.zip (streamed; nothing else is kept)."""
        import zipfile
        need = {t.lower() for t in tokens}
        out: Dict[str, List[float]] = {}
        with zipfile.ZipFile(zip_path) as z, z.open(member) as fh:
            for raw in fh:
                k = raw.find(b" ")
                if k <= 0:
                    continue
                w = raw[:k].decode("utf-8", "replace")
                if w in need:
                    out[w] = [float(x) for x in raw[k + 1:].split()]
        return out

    def _vector(self, token: str) -> torch.Tensor:
        key = token.lower()
        self.n_lookups += 1
        i = self._index.get(key)
        if i is None:
            self.oov[key] = self.oov.get(key, 0) + 1
            return self._oov._vector(key)
        return self._matrix[i]

    def with_max_tokens(self, max_tokens: Optional[int]) -> "GloveEncoder":
        """A shallow copy sharing the table (for example without a token limit, for NevIR documents)."""
        import copy
        c = copy.copy(self)
        c.max_tokens = max_tokens
        return c

    def encode(self, text: str) -> Encoded:
        toks, offs = word_tokenize(text)
        truncated = self.max_tokens is not None and len(toks) > self.max_tokens
        if self.max_tokens is not None:
            toks, offs = toks[: self.max_tokens], offs[: self.max_tokens]
        if not toks:
            toks, offs = ["<empty>"], [(0, 0)]
        return Encoded(states=torch.stack([self._vector(t) for t in toks]), tokens=toks, offsets=offs, truncated=truncated)

    def encode_batch(self, texts: Sequence[str], **kwargs) -> List[Encoded]:
        return [self.encode(t) for t in texts]

    def oov_rate(self) -> Optional[float]:
        n = sum(self.oov.values())
        return (n / self.n_lookups) if self.n_lookups else None


class ColBERTDocEncoder(HFFrozenEncoder):
    """E6: ColBERTv2 token embeddings in document mode (Santhanam et al., 2022; checkpoint colbert-ir/colbertv2.0).
    Input `[CLS] [D] tokens [SEP]` with the document marker `[unused1]` after `[CLS]`, as the checkpoint's document
    tokenizer builds it; the last-layer states pass through the checkpoint's linear projection (hidden -> 128) and are
    L2-normalized; `[CLS]`, `[D]` and `[SEP]` are removed afterwards; punctuation is kept. `projection` is the
    (128, hidden) weight read from the checkpoint file (see `load_projection`)."""

    def __init__(self, model_name: str, projection: torch.Tensor, max_length: int = 128, device: Optional[str] = None,
                 dtype: torch.dtype = torch.float32, name: Optional[str] = None, revision: Optional[str] = None,
                 doc_marker: str = "[unused1]"):
        super().__init__(model_name, max_length=max_length, device=device, dtype=dtype, name=name or "E6-colbertv2",
                         revision=revision, normalize=True)
        self.projection = projection.to(self.device, dtype=dtype)
        self.dim = int(projection.shape[0])
        self.marker_id = int(self.tokenizer.convert_tokens_to_ids(doc_marker))
        unk = self.tokenizer.unk_token_id
        if self.marker_id is None or self.marker_id == unk:
            raise RuntimeError(f"document marker {doc_marker} is not in the tokenizer vocabulary")

    @staticmethod
    def load_projection(model_name: str, revision: Optional[str] = None, filename: str = "model.safetensors") -> torch.Tensor:
        """The checkpoint's projection weight: the unique tensor whose key ends with 'linear.weight' and has two
        dimensions with the smaller one equal to 128."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        path = hf_hub_download(model_name, filename, revision=revision)
        sd = load_file(path)
        cands = [k for k, v in sd.items() if k.endswith("linear.weight") and v.ndim == 2 and min(v.shape) == 128]
        if len(cands) != 1:
            raise RuntimeError(f"projection weight not identified in {filename}: candidates {cands}")
        w = sd[cands[0]].float()
        return w if w.shape[0] == 128 else w.T

    @torch.no_grad()
    def encode_batch(self, texts: Sequence[str], max_length: Optional[int] = None) -> List[Encoded]:
        max_length = max_length or self.max_length
        tok = self.tokenizer
        enc = tok(list(texts), add_special_tokens=False, truncation=True, max_length=max_length - 3,
                  return_offsets_mapping=True)
        ids_list = [[tok.cls_token_id, self.marker_id] + ids + [tok.sep_token_id] for ids in enc["input_ids"]]
        n = max(len(x) for x in ids_list)
        pad = tok.pad_token_id
        input_ids = torch.full((len(texts), n), pad, dtype=torch.long)
        attn = torch.zeros(len(texts), n, dtype=torch.long)
        for b, ids in enumerate(ids_list):
            input_ids[b, : len(ids)] = torch.tensor(ids)
            attn[b, : len(ids)] = 1
        hidden = self.model(input_ids=input_ids.to(self.device), attention_mask=attn.to(self.device)).last_hidden_state
        proj = torch.nn.functional.normalize(hidden @ self.projection.T, dim=-1).float().cpu()
        out = []
        for b, ids in enumerate(ids_list):
            m = len(ids) - 3
            states = proj[b, 2:2 + m]
            toks = tok.convert_ids_to_tokens(ids[2:2 + m])
            offs = [tuple(int(x) for x in o) for o in enc["offset_mapping"][b]]
            truncated = len(tok(texts[b], add_special_tokens=False)["input_ids"]) > max_length - 3
            if m == 0:
                states, toks, offs = torch.zeros(1, self.dim), ["<empty>"], [(0, 0)]
            out.append(Encoded(states=states, tokens=toks, offsets=offs, truncated=truncated))
        return out


def hf_load_report(model_name: str, revision: Optional[str] = None, texts: Sequence[str] = (), head: str = "base",
                   example: str = "The bakery on Elm Street is not open today.") -> dict:
    """Sanity report for a Hugging Face checkpoint as this package loads it: tokenizer class, unknown-token rate on
    `texts`, an example tokenization, and the missing / unexpected / mismatched weight keys reported by from_pretrained
    (head: "base" = AutoModel as used by the frozen encoders, "mc" = AutoModelForMultipleChoice as used by C1,
    "seq" = AutoModelForSequenceClassification). Newly initialised task heads are expected to appear as missing."""
    from transformers import (AutoModel, AutoModelForMultipleChoice, AutoModelForSequenceClassification,
                              AutoTokenizer)

    tok = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
    unk = tok.unk_token_id
    n_tok = n_unk = 0
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        n_tok += len(ids)
        n_unk += sum(1 for i in ids if unk is not None and i == unk)
    cls = {"base": AutoModel, "mc": AutoModelForMultipleChoice, "seq": AutoModelForSequenceClassification}[head]
    rep = {"model": model_name, "revision": revision, "head": head, "tokenizer_class": type(tok).__name__,
           "texts": len(texts), "tokens": n_tok, "unk_tokens": n_unk, "unk_rate": (n_unk / n_tok) if n_tok else None,
           "example_tokens": tok.convert_ids_to_tokens(tok(example)["input_ids"])}
    try:
        try:
            model, info = cls.from_pretrained(model_name, revision=revision, dtype=torch.float32, output_loading_info=True)
        except TypeError:
            model, info = cls.from_pretrained(model_name, revision=revision, torch_dtype=torch.float32, output_loading_info=True)
        info = info or {}
        rep.update({"missing_keys": sorted(info.get("missing_keys", []) or []),
                    "unexpected_keys_n": len(info.get("unexpected_keys", []) or []),
                    "unexpected_keys_head": sorted(info.get("unexpected_keys", []) or [])[:12],
                    "mismatched_keys": [str(x) for x in (info.get("mismatched_keys", []) or [])],
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "param_dtypes": sorted({str(p.dtype) for p in model.parameters()})})
        del model
    except Exception as e:                                  # report, do not fail the notebook
        rep["load_error"] = f"{type(e).__name__}: {e}"[:500]
    return rep


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EncodedStore:
    """Cache of Encoded objects keyed by text hash; persisted as fp16 tensors.

    One store per encoder. `get_or_encode` encodes only the texts that are missing.
    """

    def __init__(self, encoder, path: Optional[str | Path] = None, store_dtype: torch.dtype = torch.float16):
        self.encoder = encoder
        self.path = Path(path) if path else None
        self.store_dtype = store_dtype
        self._items: Dict[str, Encoded] = {}
        if self.path and self.path.exists():
            self.load()

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, text: str) -> bool:
        return text_key(text) in self._items

    def get(self, text: str) -> Encoded:
        return self._items[text_key(text)]

    def get_or_encode(self, texts: Sequence[str], batch_size: int = 32, progress: bool = False, **encode_kwargs) -> List[Encoded]:
        missing = [t for t in dict.fromkeys(texts) if text_key(t) not in self._items]
        for i in range(0, len(missing), batch_size):
            chunk = missing[i:i + batch_size]
            for t, e in zip(chunk, self.encoder.encode_batch(chunk, **encode_kwargs)):
                e.states = e.states.to(self.store_dtype)
                self._items[text_key(t)] = e
            if progress and (i // batch_size) % 20 == 0:
                print(f"  encoded {min(i + batch_size, len(missing))}/{len(missing)}", end="\r")
        if progress and missing:
            print()
        return [self._items[text_key(t)] for t in texts]

    def save(self, path: Optional[str | Path] = None) -> Path:
        path = Path(path or self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: {"states": v.states, "tokens": v.tokens, "offsets": v.offsets, "truncated": v.truncated}
                   for k, v in self._items.items()}
        torch.save({"encoder": getattr(self.encoder, "name", "?"), "dim": self.encoder.dim, "items": payload}, path)
        return path

    def load(self, path: Optional[str | Path] = None) -> None:
        path = Path(path or self.path)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        for k, v in blob["items"].items():
            self._items[k] = Encoded(states=v["states"], tokens=v["tokens"], offsets=v["offsets"], truncated=v["truncated"])
