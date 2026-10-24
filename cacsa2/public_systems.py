"""Public retrieval systems scored on NegPairs and NevIR through their own interfaces (revision step 1, October 2026).

Every scorer is a function s(query, document) -> float with the system's own tokenization, length limits and
similarity. On NegPairs the query is the answer option and the document is "<context> <question>", the formulation of
the article's trained scorers and of M0-native (negpairs.NegRow.text); the pointwise LLM scorers see the context, the
question and the option as separate fields of one prompt instead. On NevIR the four query-document combinations of
every instance are scored (nevir.four_combos). Decisions and metrics are then computed by the article's own
functions (negpairs.report, np_analysis_v3.perm_report / cluster_bootstrap_flip, nevir.evaluate_scorer), so the numbers
are directly comparable with the article's tables.

Scorer interface
    name, kind, settings (JSON-serializable record of what was done), heads (None or a list of head names)
    score_pairs(queries, docs, batch_size, head=None) -> list of floats
    score_negpairs(items, batch_size, head=None)       items: (option, context, question)
    score_nevir(pairs, batch_size, head=None)          pairs: (query, document)
    close()
Kinds: late_interaction, dense, sparse, cross_encoder, monot5, llm_pointwise, api_pairwise (NevIR only).

Third-party libraries are imported lazily inside the constructors, so the metric code and the tests run without them.
"""
from __future__ import annotations

import json
import math
import re
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from . import colbert as colbert_native
from .negpairs import ARMS, LABELS2, NegRow, report
from .np_analysis_v3 import cluster_bootstrap_flip, flatten_perm, perm_report
from .nevir import evaluate_scorer, four_combos, to_score_dict

KINDS = ("late_interaction", "dense", "sparse", "cross_encoder", "monot5", "llm_pointwise", "api_pairwise")


def device_name(device: Optional[str] = None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def gpu_dtype(device: Optional[str] = None, half: bool = True) -> torch.dtype:
    """bfloat16 on a GPU that supports it (fp16 otherwise) when `half`, float32 on the CPU."""
    if not str(device_name(device)).startswith("cuda") or not half:
        return torch.float32
    try:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    except Exception:
        return torch.float16


def resolve_revision(repo: str, revision: Optional[str] = None, kind: str = "model") -> Optional[str]:
    """The commit sha the hub resolves for `repo` (the given revision or the current main); None when offline."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(repo, revision=revision) if kind == "model" else api.dataset_info(repo, revision=revision)
        return info.sha
    except Exception:
        return revision


def _from_pretrained(cls, repo: str, revision: Optional[str], dtype: Optional[torch.dtype] = None, **kw):
    """transformers >= 4.56 takes `dtype`, older versions `torch_dtype`."""
    if dtype is None:
        return cls.from_pretrained(repo, revision=revision, **kw)
    try:
        return cls.from_pretrained(repo, revision=revision, dtype=dtype, **kw)
    except TypeError:
        return cls.from_pretrained(repo, revision=revision, torch_dtype=dtype, **kw)


def purge_hub_cache(repo: str) -> bool:
    """Delete the local hub cache of one repository (large checkpoints are not needed after scoring)."""
    import shutil
    try:
        from huggingface_hub import constants
        root = Path(constants.HF_HUB_CACHE)
    except Exception:
        return False
    d = root / f"models--{repo.replace('/', '--')}"
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        return True
    return False


def negpairs_document(context: str, question: str) -> str:
    return f"{context.strip()} {question.strip()}"



def decisive_logits(model, forward_kwargs: dict, token_ids: Sequence[int], position: int = -1) -> Tuple[torch.Tensor, str]:
    """Logits of `token_ids` at `position`, computed in float32 from the state that enters the output head (captured
    with a forward pre-hook on the head) and the float32-cast head rows. A model run in bfloat16 or float16 produces
    logits with the resolution of that type (about 0.1 at a magnitude of 20), which turns small differences between
    two documents into exact ties; the score differences used here (log-odds, log-softmax over two tokens) do not
    depend on the other logits, so only these rows are needed. Falls back to the model's own logits when the head
    cannot be found. Returns (B, len(token_ids)) float32 and the precision label."""
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if head is None or not hasattr(head, "weight"):
        out = model(**forward_kwargs).logits[:, position, :].float()
        return out[:, list(token_ids)], "model logits (" + str(next(model.parameters()).dtype) + ")"
    captured = {}
    handle = head.register_forward_pre_hook(lambda module, args: captured.__setitem__("h", args[0]))
    try:
        try:
            model(**forward_kwargs, logits_to_keep=1 if position == -1 else 0)
        except TypeError:
            model(**forward_kwargs)
    finally:
        handle.remove()
    h = captured["h"][:, position, :].float()
    W = head.weight[list(token_ids)].float()
    out = h @ W.T
    if getattr(head, "bias", None) is not None:
        out = out + head.bias[list(token_ids)].float()
    return out, "float32 head on the float32-cast pre-head state"


# ===================================================================================================== base class
class Scorer:
    name: str = ""
    kind: str = ""
    heads: Optional[List[str]] = None

    def __init__(self):
        self.settings: dict = {}

    def score_pairs(self, queries: Sequence[str], docs: Sequence[str], batch_size: int = 32, head: Optional[str] = None) -> List[float]:
        raise NotImplementedError

    def score_negpairs(self, items: Sequence[Tuple[str, str, str]], batch_size: int = 32, head: Optional[str] = None) -> List[float]:
        return self.score_pairs([o for o, _, _ in items], [negpairs_document(c, q) for _, c, q in items], batch_size, head)

    def score_nevir(self, pairs: Sequence[Tuple[str, str]], batch_size: int = 32, head: Optional[str] = None) -> List[float]:
        return self.score_pairs([q for q, _ in pairs], [d for _, d in pairs], batch_size, head)

    def close(self):
        for attr in ("model", "native"):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _unique_index(texts: Sequence[str]) -> Tuple[List[str], Dict[str, int]]:
    uniq = list(dict.fromkeys(texts))
    return uniq, {t: i for i, t in enumerate(uniq)}


# ===================================================================================================== late interaction
class _NativeFlags(colbert_native.ColBERTNative):
    """ColBERTNative with the two checkpoint flags that colbertv2.0 fixes (attend_to_mask_tokens, mask_punctuation)."""

    def __init__(self, *args, attend_to_mask_tokens: bool = False, mask_punctuation: bool = True, **kw):
        super().__init__(*args, **kw)
        self.attend_to_mask_tokens = bool(attend_to_mask_tokens)
        if not mask_punctuation:
            self.skiplist = set()
        self.settings.update({"attend_to_mask_tokens": self.attend_to_mask_tokens, "mask_punctuation": bool(mask_punctuation)})

    @torch.no_grad()
    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        if not self.attend_to_mask_tokens:
            return super().encode_queries(texts)
        ids, attn, _ = self._ids(texts, self.q_id, self.query_maxlen, pad_with_mask=True)
        attn = torch.ones_like(attn)
        h = self.model(input_ids=ids.to(self.device), attention_mask=attn.to(self.device)).last_hidden_state
        return F.normalize(h @ self.W.T, dim=-1).float()


def load_colbert_projection(repo: str, revision: Optional[str] = None, filename: str = "model.safetensors") -> Tuple[torch.Tensor, str]:
    """The checkpoint's output projection: the unique two-dimensional tensor whose key ends with 'linear.weight' and
    whose first dimension (the output size) is smaller than its second (the hidden size)."""
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo, filename, revision=revision)
        from safetensors import safe_open
        with safe_open(path, "pt") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys() if k.endswith("linear.weight")}
    except Exception:
        if filename != "pytorch_model.bin":
            filename = "pytorch_model.bin"
            path = hf_hub_download(repo, filename, revision=revision)
            sd = torch.load(path, map_location="cpu", weights_only=True)
            tensors = {k: v for k, v in sd.items() if k.endswith("linear.weight")}
        else:
            raise
    cands = [(k, t) for k, t in tensors.items() if t.ndim == 2 and t.shape[0] < t.shape[1]]
    if len(cands) != 1:
        raise RuntimeError(f"projection weight not identified in {repo}/{filename}: candidates {[k for k, _ in cands]}")
    k, t = cands[0]
    return t.float(), k


def load_colbert_metadata(repo: str, revision: Optional[str] = None) -> dict:
    """artifact.metadata of a Stanford-format ColBERT checkpoint (empty when the file is absent)."""
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(repo, "artifact.metadata", revision=revision)
    except Exception:
        return {}
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return {}


class ColBERTCheckpoint(Scorer):
    """A Stanford-format ColBERT checkpoint (colbert-ir/colbertv2.0, answerdotai/answerai-colbert-small-v1, ...) scored
    with its own settings: query_maxlen with [MASK] augmentation, doc_maxlen, projection dim, cosine similarity,
    punctuation masking and the mask-attention flag read from artifact.metadata (overridable)."""
    kind = "late_interaction"

    def __init__(self, repo: str, revision: Optional[str] = None, device: Optional[str] = None, dtype: Optional[torch.dtype] = None,
                 name: Optional[str] = None, metadata_overrides: Optional[dict] = None):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.name = name or repo
        self.device = device_name(device)
        dtype = dtype or torch.float32
        meta = load_colbert_metadata(repo, revision)
        meta.update(metadata_overrides or {})
        sim = str(meta.get("similarity", "cosine")).lower()
        if sim != "cosine":
            raise NotImplementedError(f"{repo}: similarity {sim!r} not supported (cosine only)")
        W, key = load_colbert_projection(repo, revision)
        tok = AutoTokenizer.from_pretrained(repo, revision=revision, use_fast=True)
        model = _from_pretrained(AutoModel, repo, revision, dtype)
        self.native = _NativeFlags(repo, W, device=self.device, dtype=dtype, revision=revision,
                                   query_maxlen=int(meta.get("query_maxlen", 32)), doc_maxlen=int(meta.get("doc_maxlen", 180)),
                                   query_marker=str(meta.get("query_token_id", "[unused0]")), doc_marker=str(meta.get("doc_token_id", "[unused1]")),
                                   tokenizer=tok, model=model, attend_to_mask_tokens=bool(meta.get("attend_to_mask_tokens", False)),
                                   mask_punctuation=bool(meta.get("mask_punctuation", True)))
        if int(W.shape[0]) != int(meta.get("dim", W.shape[0])):
            raise RuntimeError(f"{repo}: projection dim {W.shape[0]} differs from artifact.metadata dim {meta.get('dim')}")
        self.settings = {"repo": repo, "revision": revision, "projection_key": key, "dtype": str(dtype), **self.native.settings,
                         "metadata_found": bool(meta), "hidden": int(W.shape[1])}
        self.truncated_docs = 0

    @torch.no_grad()
    def score_pairs(self, queries, docs, batch_size: int = 32, head=None) -> List[float]:
        out: List[float] = []
        for i in range(0, len(queries), batch_size):
            Q = self.native.encode_queries(list(queries[i:i + batch_size]))
            D, trunc = self.native.encode_docs(list(docs[i:i + batch_size]))
            self.truncated_docs += int(sum(trunc))
            out.extend(self.native.score(Q, D).cpu().tolist())
        return out


def _repo_file(repo: str, filename: str, revision: Optional[str] = None) -> Path:
    """A file of a checkpoint: from a local directory when `repo` is one, else from the hub cache."""
    local = Path(repo) / filename
    if Path(repo).is_dir():
        if not local.exists():
            raise FileNotFoundError(str(local))
        return local
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename, revision=revision))


def _load_state(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))
    return torch.load(str(path), map_location="cpu", weights_only=True)


class STColBERT(Scorer):
    """A PyLate-format ColBERT checkpoint (sentence-transformers layout: transformer at the root, a Dense projection
    module, `config_sentence_transformers.json` with the ColBERT settings), e.g. lightonai/GTE-ModernColBERT-v1, scored
    without the PyLate library by replicating its encoding: the prefix token ([Q] / [D]) inserted after the first
    token, queries truncated to query_length - 1 and expanded with [MASK] tokens to query_length (not attended unless
    attend_to_expansion_tokens; every query vector takes part in the score), documents truncated to document_length - 1
    with the skiplist (punctuation) and padding vectors removed, projection, L2 normalization, and the sum over query
    vectors of the maximum dot product over document vectors."""
    kind = "late_interaction"

    def __init__(self, repo: str, revision: Optional[str] = None, device: Optional[str] = None, dtype: Optional[torch.dtype] = None,
                 name: Optional[str] = None, document_length: Optional[int] = None, query_length: Optional[int] = None):
        super().__init__()
        import string
        from transformers import AutoModel, AutoTokenizer
        self.name = name or repo
        self.device = device_name(device)
        dtype = dtype or torch.float32
        cfg = json.loads(_repo_file(repo, "config_sentence_transformers.json", revision).read_text(encoding="utf-8"))
        modules = json.loads(_repo_file(repo, "modules.json", revision).read_text(encoding="utf-8"))
        dense = [m for m in modules if "Dense" in str(m.get("type", ""))]
        if len(dense) != 1:
            raise RuntimeError(f"{repo}: expected one Dense module in modules.json, found {len(dense)}")
        ddir = dense[0]["path"]
        dcfg = json.loads(_repo_file(repo, f"{ddir}/config.json", revision).read_text(encoding="utf-8"))
        if "Identity" not in str(dcfg.get("activation_function", "Identity")):
            raise NotImplementedError(f"{repo}: Dense activation {dcfg.get('activation_function')!r} not supported")
        if dcfg.get("use_residual"):
            raise NotImplementedError(f"{repo}: residual Dense not supported")
        try:
            sd = _load_state(_repo_file(repo, f"{ddir}/model.safetensors", revision))
        except Exception:
            sd = _load_state(_repo_file(repo, f"{ddir}/pytorch_model.bin", revision))
        self.W = sd["linear.weight"].to(self.device, dtype)
        self.b = sd["linear.bias"].to(self.device, dtype) if "linear.bias" in sd else None
        self.tok = AutoTokenizer.from_pretrained(repo, revision=revision)
        self.model = _from_pretrained(AutoModel, repo, revision, dtype).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.query_prefix = cfg.get("query_prefix", "[Q] ")
        self.document_prefix = cfg.get("document_prefix", "[D] ")
        prefixes = [p for p in (self.query_prefix, self.document_prefix) if p]
        if prefixes and any(self.tok.convert_tokens_to_ids(p) in (None, self.tok.unk_token_id) for p in prefixes):
            self.tok.add_tokens(prefixes)
            self.model.resize_token_embeddings(len(self.tok))
        self.q_prefix_id = self.tok.convert_tokens_to_ids(self.query_prefix) if self.query_prefix else None
        self.d_prefix_id = self.tok.convert_tokens_to_ids(self.document_prefix) if self.document_prefix else None
        if self.tok.mask_token_id is not None:
            self.tok.pad_token_id = self.tok.mask_token_id
        elif self.tok.eos_token_id is not None:
            self.tok.pad_token_id = self.tok.eos_token_id
        self.query_length = int(query_length or cfg.get("query_length") or 32)
        self.document_length = int(document_length or cfg.get("document_length") or 180)
        self.do_query_expansion = bool(cfg.get("do_query_expansion", True))
        self.attend_to_expansion_tokens = bool(cfg.get("attend_to_expansion_tokens", False))
        words = cfg.get("skiplist_words")
        self.skiplist_words = list(words) if words is not None else list(string.punctuation)
        self.skiplist = {int(self.tok.convert_tokens_to_ids(w)) for w in self.skiplist_words}
        self.settings = {"repo": repo, "revision": revision, "dtype": str(dtype), "query_prefix": self.query_prefix, "document_prefix": self.document_prefix,
                         "query_length": self.query_length, "document_length": self.document_length, "do_query_expansion": self.do_query_expansion,
                         "attend_to_expansion_tokens": self.attend_to_expansion_tokens, "skiplist_words": "".join(self.skiplist_words),
                         "dim": int(self.W.shape[0]), "dense_module": ddir, "pad_token_for_expansion": self.tok.convert_ids_to_tokens(self.tok.pad_token_id),
                         "implementation": "cacsa2.public_systems.STColBERT (PyLate encoding replicated without the library)"}
        self.truncated_docs = 0

    def _tokenize(self, texts: Sequence[str], is_query: bool):
        max_len = (self.query_length if is_query else self.document_length) - 1
        texts = [str(t).strip() for t in texts]
        padding = "max_length" if (is_query and self.do_query_expansion) else True
        enc = self.tok(texts, padding=padding, truncation=True, max_length=max_len, return_tensors="pt")
        if not is_query:
            lengths = [len(x) for x in self.tok(texts, truncation=False)["input_ids"]]
            self.truncated_docs += sum(n > max_len for n in lengths)
        ids, attn = enc["input_ids"], enc["attention_mask"]
        prefix = self.q_prefix_id if is_query else self.d_prefix_id
        if prefix is not None:
            n = ids.shape[0]
            ids = torch.cat([ids[:, :1], torch.full((n, 1), int(prefix), dtype=ids.dtype), ids[:, 1:]], dim=1)
            attn = torch.cat([attn[:, :1], torch.ones((n, 1), dtype=attn.dtype), attn[:, 1:]], dim=1)
        if is_query and self.attend_to_expansion_tokens:
            attn = torch.ones_like(attn)
        return ids, attn

    @torch.no_grad()
    def encode(self, texts: Sequence[str], is_query: bool) -> List[torch.Tensor]:
        ids, attn = self._tokenize(texts, is_query)
        h = self.model(input_ids=ids.to(self.device), attention_mask=attn.to(self.device)).last_hidden_state
        e = h @ self.W.T
        if self.b is not None:
            e = e + self.b
        e = F.normalize(e.float(), dim=-1)
        out = []
        for i in range(ids.shape[0]):
            if is_query:
                keep = torch.ones(ids.shape[1], dtype=torch.bool) if self.do_query_expansion else attn[i].bool()
            else:
                keep = attn[i].bool() & ~torch.isin(ids[i], torch.tensor(sorted(self.skiplist), dtype=ids.dtype))
            out.append(e[i][keep.to(e.device)])
        return out

    @torch.no_grad()
    def score_pairs(self, queries, docs, batch_size: int = 32, head=None) -> List[float]:
        uq, qi = _unique_index(queries)
        ud, di = _unique_index(docs)
        Q: List[torch.Tensor] = []
        for i in range(0, len(uq), batch_size):
            Q.extend(self.encode(uq[i:i + batch_size], True))
        D: List[torch.Tensor] = []
        for i in range(0, len(ud), batch_size):
            D.extend(self.encode(ud[i:i + batch_size], False))
        out = []
        for q, d in zip(queries, docs):
            a, b = Q[qi[q]], D[di[d]]
            out.append(float((a @ b.T).max(dim=1).values.sum()) if a.shape[0] and b.shape[0] else 0.0)
        return out


class BGEM3(Scorer):
    """BAAI/bge-m3 with its three heads (dense, sparse, colbert) computed in one pass from the released weights,
    following the reference implementation: dense = L2-normalized [CLS] state; sparse = relu(sparse_linear(h)) with
    the maximum weight per token id, unused tokens (<s>, </s>, <pad>, <unk>) removed, score = sum over shared token
    ids of the products; colbert = L2-normalized colbert_linear(h[1:]) under the attention mask, score = mean over
    query vectors of the maximum dot product over document vectors."""
    kind = "late_interaction"
    heads = ["dense", "sparse", "colbert"]

    def __init__(self, repo: str = "BAAI/bge-m3", revision: Optional[str] = None, device: Optional[str] = None,
                 dtype: Optional[torch.dtype] = None, max_length: int = 1024, name: Optional[str] = None):
        super().__init__()
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoTokenizer
        self.name = name or repo
        self.device = device_name(device)
        # the reference implementation runs the model in float16 (use_fp16=True); same here on a GPU
        self.dtype = dtype or (torch.float16 if str(self.device).startswith("cuda") else torch.float32)
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(repo, revision=revision)
        self.model = _from_pretrained(AutoModel, repo, revision, self.dtype).to(self.device).eval()
        hid = int(self.model.config.hidden_size)
        self.colbert_linear = torch.nn.Linear(hid, hid)
        self.sparse_linear = torch.nn.Linear(hid, 1)
        for lin, fn in ((self.colbert_linear, "colbert_linear.pt"), (self.sparse_linear, "sparse_linear.pt")):
            sd = torch.load(hf_hub_download(repo, fn, revision=revision), map_location="cpu", weights_only=True)
            lin.load_state_dict(sd)
            lin.to(self.device, self.dtype).eval()
        self.unused = {self.tok.cls_token_id, self.tok.eos_token_id, self.tok.pad_token_id, self.tok.unk_token_id}
        self._cache: Dict[str, tuple] = {}
        self.settings = {"repo": repo, "revision": revision, "dtype": str(self.dtype), "max_length": max_length, "heads": self.heads,
                         "implementation": "cacsa2.public_systems.BGEM3 (reference heads re-implemented from the released weights)"}

    @torch.no_grad()
    def encode(self, texts: Sequence[str], batch_size: int = 16):
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        todo.sort(key=len, reverse=True)
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            enc = self.tok(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            dense = F.normalize(h[:, 0].float(), dim=-1).cpu()
            tw = torch.relu(self.sparse_linear(h)).squeeze(-1).float().cpu()
            cv = self.colbert_linear(h[:, 1:]) * enc["attention_mask"][:, 1:, None].to(h.dtype)
            cv = F.normalize(cv.float(), dim=-1).cpu()
            ids = enc["input_ids"].cpu()
            am = enc["attention_mask"].cpu()
            for b, t in enumerate(batch):
                n = int(am[b].sum())
                sp: Dict[int, float] = {}
                for j in range(n):
                    tid, w = int(ids[b, j]), float(tw[b, j])
                    if w > 0 and tid not in self.unused:
                        if w > sp.get(tid, 0.0):
                            sp[tid] = w
                self._cache[t] = (dense[b].clone(), sp, cv[b, : n - 1].clone().half())

    def score_pairs(self, queries, docs, batch_size: int = 16, head: Optional[str] = "dense") -> List[float]:
        head = head or "dense"
        if head not in self.heads:
            raise ValueError(head)
        self.encode(list(queries) + list(docs), batch_size)
        out = []
        for q, d in zip(queries, docs):
            qd, qs, qc = self._cache[q]
            dd, ds, dc = self._cache[d]
            if head == "dense":
                out.append(float(torch.dot(qd, dd)))
            elif head == "sparse":
                out.append(float(sum(w * ds[t] for t, w in qs.items() if t in ds)))
            else:
                if qc.shape[0] == 0 or dc.shape[0] == 0:
                    out.append(0.0)
                else:
                    out.append(float((qc.float() @ dc.float().T).max(dim=1).values.sum() / qc.shape[0]))
        return out

    def close(self):
        self._cache.clear()
        super().close()


def bgem3_selfcheck(scorer: "BGEM3", pairs: Sequence[Tuple[str, str]], repo: str = "BAAI/bge-m3", install: bool = True) -> dict:
    """Checks the re-implemented bge-m3 heads on a few pairs: the dense head against sentence-transformers (CLS
    pooling, normalized) and, when FlagEmbedding can be imported (an install without dependencies is attempted),
    all three heads against the reference implementation. Records maximum absolute differences; never raises."""
    out: dict = {"pairs": len(pairs)}
    qs, ds = [q for q, _ in pairs], [d for _, d in pairs]
    mine = {h: scorer.score_pairs(qs, ds, head=h) for h in scorer.heads}
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(repo, device=scorer.device)
        Q = st.encode(qs, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False).float()
        D = st.encode(ds, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False).float()
        ref = [float((Q[i] * D[i]).sum()) for i in range(len(qs))]
        out["dense_vs_sentence_transformers_max_abs_diff"] = max(abs(a - b) for a, b in zip(mine["dense"], ref))
        del st, Q, D
    except Exception as e:
        out["dense_check"] = f"not run: {e!r}"[:300]
    try:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except Exception:
            if not install:
                raise
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "FlagEmbedding"], check=False, capture_output=True)
            from FlagEmbedding import BGEM3FlagModel
        m = BGEM3FlagModel(repo, use_fp16=scorer.dtype != torch.float32, device=scorer.device)
        oq = m.encode(qs, return_dense=True, return_sparse=True, return_colbert_vecs=True, max_length=scorer.max_length)
        od = m.encode(ds, return_dense=True, return_sparse=True, return_colbert_vecs=True, max_length=scorer.max_length)
        dense_ref = [float((oq["dense_vecs"][i] * od["dense_vecs"][i]).sum()) for i in range(len(qs))]
        sparse_ref = [float(m.compute_lexical_matching_score(oq["lexical_weights"][i], od["lexical_weights"][i])) for i in range(len(qs))]
        colbert_ref = [float(m.colbert_score(oq["colbert_vecs"][i], od["colbert_vecs"][i])) for i in range(len(qs))]
        for h, ref in (("dense", dense_ref), ("sparse", sparse_ref), ("colbert", colbert_ref)):
            out[f"{h}_vs_flagembedding_max_abs_diff"] = max(abs(a - b) for a, b in zip(mine[h], ref))
        del m
    except Exception as e:
        out["flagembedding_check"] = f"not run: {e!r}"[:300]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ===================================================================================================== dense and sparse
class SentenceDense(Scorer):
    """A sentence-transformers bi-encoder with optional query / document prefixes; cosine of the pooled vectors."""
    kind = "dense"

    def __init__(self, repo: str, revision: Optional[str] = None, device: Optional[str] = None, name: Optional[str] = None,
                 query_prefix: str = "", doc_prefix: str = "", max_seq_length: int = 512, trust_remote_code: bool = False,
                 model_kwargs: Optional[dict] = None):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.name = name or repo
        self.device = device_name(device)
        kw = {"device": self.device, "trust_remote_code": trust_remote_code}
        if revision:
            kw["revision"] = revision
        if model_kwargs:
            kw["model_kwargs"] = model_kwargs
        self.model = SentenceTransformer(repo, **kw)
        try:
            self.model.max_seq_length = max_seq_length
        except Exception:
            pass
        self.qp, self.dp = query_prefix, doc_prefix
        self.settings = {"repo": repo, "revision": revision, "query_prefix": query_prefix, "doc_prefix": doc_prefix,
                         "max_seq_length": getattr(self.model, "max_seq_length", None), "similarity": "cosine (normalized dot)"}

    @torch.no_grad()
    def score_pairs(self, queries, docs, batch_size: int = 64, head=None) -> List[float]:
        uq, qi = _unique_index(queries)
        ud, di = _unique_index(docs)
        Q = self.model.encode([self.qp + q for q in uq], batch_size=batch_size, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False).float()
        D = self.model.encode([self.dp + d for d in ud], batch_size=batch_size, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False).float()
        return [float((Q[qi[q]] * D[di[d]]).sum()) for q, d in zip(queries, docs)]


class SpladeScorer(Scorer):
    """SPLADE (masked-language-model head): rep = max over tokens of log(1 + relu(logits)) under the attention mask;
    score = dot product of the two vocabulary vectors (the benchmark's own evaluation uses the same aggregation)."""
    kind = "sparse"

    def __init__(self, repo: str = "naver/splade-cocondenser-ensembledistil", revision: Optional[str] = None, device: Optional[str] = None,
                 name: Optional[str] = None, max_length: int = 512):
        super().__init__()
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.name = name or repo
        self.device = device_name(device)
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(repo, revision=revision)
        self.model = _from_pretrained(AutoModelForMaskedLM, repo, revision, torch.float32).to(self.device).eval()
        self._cache: Dict[str, Dict[int, float]] = {}
        self.settings = {"repo": repo, "revision": revision, "max_length": max_length, "aggregation": "max", "activation": "log1p(relu)"}

    @torch.no_grad()
    def encode(self, texts: Sequence[str], batch_size: int = 32):
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        todo.sort(key=len, reverse=True)
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            enc = self.tok(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
            rep = torch.log1p(torch.relu(logits)) * enc["attention_mask"][:, :, None].to(logits.dtype)
            rep = rep.max(dim=1).values.float().cpu()
            for b, t in enumerate(batch):
                nz = torch.nonzero(rep[b], as_tuple=False).flatten()
                self._cache[t] = {int(i): float(rep[b, i]) for i in nz.tolist()}

    def score_pairs(self, queries, docs, batch_size: int = 32, head=None) -> List[float]:
        self.encode(list(queries) + list(docs), batch_size)
        out = []
        for q, d in zip(queries, docs):
            a, b = self._cache[q], self._cache[d]
            if len(b) < len(a):
                a, b = b, a
            out.append(float(sum(w * b[t] for t, w in a.items() if t in b)))
        return out

    def close(self):
        self._cache.clear()
        super().close()


# ===================================================================================================== cross-encoders
class CrossEncoderScorer(Scorer):
    """A sentence-transformers cross-encoder (ms-marco MiniLM, bge-reranker, mxbai-rerank, ...); raw logit of the
    relevance output (no activation), the positive class for two-way heads."""
    kind = "cross_encoder"

    def __init__(self, repo: str, revision: Optional[str] = None, device: Optional[str] = None, name: Optional[str] = None,
                 max_length: int = 512, trust_remote_code: bool = False, model_kwargs: Optional[dict] = None):
        super().__init__()
        from sentence_transformers import CrossEncoder
        self.name = name or repo
        self.device = device_name(device)
        kw = {"device": self.device, "max_length": max_length, "trust_remote_code": trust_remote_code}
        if revision:
            kw["revision"] = revision
        if model_kwargs:
            kw["model_kwargs"] = model_kwargs
        try:
            self.model = CrossEncoder(repo, **kw)
        except TypeError:
            kw.pop("model_kwargs", None)
            self.model = CrossEncoder(repo, **kw)
        self.num_labels = None
        try:
            self.num_labels = int(self.model.model.config.num_labels)
        except Exception:
            pass
        self.settings = {"repo": repo, "revision": revision, "max_length": max_length, "num_labels": self.num_labels, "output": "raw logit"}

    @torch.no_grad()
    def score_pairs(self, queries, docs, batch_size: int = 32, head=None) -> List[float]:
        import numpy as np
        pairs = [(q, d) for q, d in zip(queries, docs)]
        kw = dict(batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        try:
            out = self.model.predict(pairs, activation_fn=torch.nn.Identity(), **kw)
        except TypeError:
            out = self.model.predict(pairs, activation_fct=torch.nn.Identity(), **kw)
        out = np.asarray(out, dtype=float)
        if out.ndim == 2:
            out = out[:, 1] if out.shape[1] == 2 else out[:, -1]
        return [float(x) for x in out]


class MonoT5Scorer(Scorer):
    """monoT5 (castorini): input 'Query: q Document: d Relevant:'; score = log-probability of the token 'true' in the
    softmax over {'false', 'true'} at the first decoder step (as in pygaggle, the benchmark's reranker code)."""
    kind = "monot5"

    def __init__(self, repo: str, revision: Optional[str] = None, device: Optional[str] = None, dtype: Optional[torch.dtype] = None,
                 name: Optional[str] = None, max_length: int = 512):
        super().__init__()
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        self.name = name or repo
        self.device = device_name(device)
        self.dtype = dtype or gpu_dtype(self.device)
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(repo, revision=revision)
        self.model = _from_pretrained(T5ForConditionalGeneration, repo, revision, self.dtype).to(self.device).eval()
        vocab = self.tok.get_vocab()
        self.true_id = vocab.get("▁true", None)
        self.false_id = vocab.get("▁false", None)
        if self.true_id is None or self.false_id is None:
            self.true_id = self.tok.encode("true", add_special_tokens=False)[0]
            self.false_id = self.tok.encode("false", add_special_tokens=False)[0]
        self.settings = {"repo": repo, "revision": revision, "dtype": str(self.dtype), "max_length": max_length,
                         "true_id": int(self.true_id), "false_id": int(self.false_id), "template": "Query: {q} Document: {d} Relevant:"}

    @torch.no_grad()
    def score_pairs(self, queries, docs, batch_size: int = 16, head=None) -> List[float]:
        out: List[float] = []
        start = self.model.config.decoder_start_token_id
        if start is None:
            start = self.tok.pad_token_id
        for i in range(0, len(queries), batch_size):
            texts = [f"Query: {q} Document: {d} Relevant:" for q, d in zip(queries[i:i + batch_size], docs[i:i + batch_size])]
            enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            dec = torch.full((len(texts), 1), int(start), dtype=torch.long, device=self.device)
            logits, prec = decisive_logits(self.model, {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "decoder_input_ids": dec},
                                           [self.false_id, self.true_id], position=0)
            self.settings["head_precision"] = prec
            two = torch.log_softmax(logits, dim=-1)
            out.extend(two[:, 1].cpu().tolist())
        return out


# ===================================================================================================== pointwise LLMs
class LLMPointwise(Scorer):
    """An instruction-tuned causal LM used as a pointwise relevance judge: the prompt asks for Yes or No and the score
    is log P(Yes) - log P(No) at the first generated position (log-sum-exp over the single-token spellings of each
    word). Documents are truncated to `max_doc_tokens` before the prompt is built, so the instruction is never cut."""
    kind = "llm_pointwise"
    SYSTEM = "You are a careful reader. Answer with exactly one word: Yes or No."
    NEGPAIRS_USER = ("Context: {context}\nQuestion: {question}\nProposed answer: {option}\n\n"
                     "According to the context, is the proposed answer the correct answer to the question? Answer Yes or No.")
    NEVIR_USER = ("Query: {query}\nDocument: {document}\n\n"
                  "Is the document relevant to the query, that is, does the document describe what the query asks for? Answer Yes or No.")
    YES = ("Yes", " Yes", "yes", " yes", "YES")
    NO = ("No", " No", "no", " no", "NO")

    def __init__(self, repo: str, revision: Optional[str] = None, device: Optional[str] = None, dtype: Optional[torch.dtype] = None,
                 name: Optional[str] = None, max_doc_tokens: int = 700, chat_kwargs: Optional[dict] = None, use_system: bool = True,
                 trust_remote_code: bool = False, token: Optional[str] = None):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = name or repo
        self.device = device_name(device)
        self.dtype = dtype or gpu_dtype(self.device)
        self.max_doc_tokens = max_doc_tokens
        self.chat_kwargs = dict(chat_kwargs or {})
        self.use_system = use_system
        tk = {"trust_remote_code": trust_remote_code}
        if token:
            tk["token"] = token
        self.tok = AutoTokenizer.from_pretrained(repo, revision=revision, **tk)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = _from_pretrained(AutoModelForCausalLM, repo, revision, self.dtype, **tk).to(self.device).eval()
        self.yes_ids = self._single_tokens(self.YES)
        self.no_ids = self._single_tokens(self.NO)
        if not self.yes_ids or not self.no_ids:
            raise RuntimeError(f"{repo}: no single-token spelling of Yes/No in the vocabulary")
        self._system_ok = None
        self.settings = {"repo": repo, "revision": revision, "dtype": str(self.dtype), "max_doc_tokens": max_doc_tokens,
                         "chat_kwargs": self.chat_kwargs, "yes_ids": self.yes_ids, "no_ids": self.no_ids,
                         "system": self.SYSTEM if use_system else None, "negpairs_prompt": self.NEGPAIRS_USER, "nevir_prompt": self.NEVIR_USER,
                         "score": "log P(Yes) - log P(No) at the first generated token (decisive logits in float32)"}

    def _single_tokens(self, words: Sequence[str]) -> List[int]:
        ids: List[int] = []
        for w in words:
            t = self.tok.encode(w, add_special_tokens=False)
            if len(t) == 1 and t[0] not in ids:
                ids.append(int(t[0]))
        return ids

    def truncate(self, text: str, max_tokens: int) -> str:
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        return self.tok.decode(ids[:max_tokens], skip_special_tokens=True)

    def prompt(self, user: str) -> str:
        msgs = [{"role": "user", "content": user}]
        if self.use_system and self._system_ok is not False:
            try:
                s = self.tok.apply_chat_template([{"role": "system", "content": self.SYSTEM}] + msgs, tokenize=False,
                                                 add_generation_prompt=True, **self.chat_kwargs)
                self._system_ok = True
                return s
            except Exception:
                self._system_ok = False
                self.settings["system"] = None
                self.settings["system_note"] = "chat template rejected the system role; instruction kept in the user turn"
        if not self.use_system or self._system_ok is False:
            msgs = [{"role": "user", "content": f"{self.SYSTEM}\n\n{user}"}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **self.chat_kwargs)

    @torch.no_grad()
    def score_prompts(self, prompts: Sequence[str], batch_size: int = 8) -> List[float]:
        order = sorted(range(len(prompts)), key=lambda k: -len(prompts[k]))
        out = [0.0] * len(prompts)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            enc = self.tok([prompts[k] for k in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            attn = enc["attention_mask"]
            pos = (attn.cumsum(-1) - 1).clamp(min=0)                 # left padding: real tokens keep positions 0..n-1
            kw = {"input_ids": enc["input_ids"], "attention_mask": attn, "position_ids": pos}
            try:
                logits, prec = decisive_logits(self.model, kw, self.yes_ids + self.no_ids, position=-1)
            except TypeError:
                kw.pop("position_ids")
                logits, prec = decisive_logits(self.model, kw, self.yes_ids + self.no_ids, position=-1)
            self.settings["head_precision"] = prec
            ny = len(self.yes_ids)
            yes = torch.logsumexp(logits[:, :ny], dim=-1)          # log P(Yes) - log P(No) = logsumexp(l_yes) - logsumexp(l_no)
            no = torch.logsumexp(logits[:, ny:], dim=-1)
            for k, v in zip(idx, (yes - no).cpu().tolist()):
                out[k] = float(v)
        return out

    def score_pairs(self, queries, docs, batch_size: int = 8, head=None) -> List[float]:
        return self.score_nevir(list(zip(queries, docs)), batch_size)

    def score_negpairs(self, items, batch_size: int = 8, head=None) -> List[float]:
        prompts = [self.prompt(self.NEGPAIRS_USER.format(context=self.truncate(c, self.max_doc_tokens), question=q, option=o)) for o, c, q in items]
        return self.score_prompts(prompts, batch_size)

    def score_nevir(self, pairs, batch_size: int = 8, head=None) -> List[float]:
        prompts = [self.prompt(self.NEVIR_USER.format(query=q, document=self.truncate(d, self.max_doc_tokens))) for q, d in pairs]
        return self.score_prompts(prompts, batch_size)


# ===================================================================================================== API pairwise (NevIR)
class ClaudePairwiseNevIR(Scorer):
    """An API model as a pairwise judge on NevIR: for each query both documents are shown in both orders and the
    model names the more relevant one (A or B). The score of a document is the number of orders in which it was
    chosen (0, 1 or 2), so an order-dependent answer is a tie (an error under the benchmark's strict rule).
    `call(model, system, user) -> (text, in_tokens, out_tokens)` is injected; the cost is metered with `price`
    (USD per million input and output tokens) and the run stops at `cap_usd`."""
    kind = "api_pairwise"
    SYSTEM = "You compare two documents for a search query. Answer with exactly one letter: A or B."
    USER = ("Query: {query}\n\nDocument A: {a}\n\nDocument B: {b}\n\n"
            "Which document is more relevant to the query, that is, which one describes what the query asks for? Answer A or B.")

    def __init__(self, model: str, call: Callable, price: Tuple[float, float], cap_usd: float, name: Optional[str] = None):
        super().__init__()
        self.model, self.call, self.price, self.cap = model, call, price, cap_usd
        self.name = name or model
        self.usd, self.calls, self.in_tokens, self.out_tokens = 0.0, 0, 0, 0
        self.settings = {"model": model, "price_usd_per_million": list(price), "cap_usd": cap_usd, "system": self.SYSTEM, "user": self.USER,
                         "orders": 2, "score": "number of orders in which the document was chosen"}

    def judge(self, query: str, a: str, b: str) -> Optional[str]:
        text, i, o = self.call(self.model, self.SYSTEM, self.USER.format(query=query, a=a, b=b))
        self.calls += 1
        self.in_tokens += int(i)
        self.out_tokens += int(o)
        self.usd += (int(i) * self.price[0] + int(o) * self.price[1]) / 1e6
        m = re.search(r"\b([AB])\b", (text or "").strip().upper())
        return m.group(1) if m else None

    def score_instances(self, instances: Sequence[dict], done: Optional[Dict[str, Dict[str, float]]] = None,
                        on_progress: Optional[Callable] = None) -> Dict[str, Dict[str, float]]:
        """Judges the instances not yet in `done`, which is updated in place (so partial progress can be saved from
        `on_progress(n_done, usd)`), and returns it."""
        out: Dict[str, Dict[str, float]] = done if done is not None else {}
        for ins in instances:
            if ins["id"] in out:
                continue
            if self.usd >= self.cap:
                raise RuntimeError(f"API cost cap reached: {self.usd:.2f} USD >= {self.cap} USD after {self.calls} calls")
            rec = {}
            for qk in ("q1", "q2"):
                first = self.judge(ins[qk], ins["doc1"], ins["doc2"])      # A = doc1
                second = self.judge(ins[qk], ins["doc2"], ins["doc1"])     # A = doc2
                d1 = int(first == "A") + int(second == "B")
                d2 = int(first == "B") + int(second == "A")
                rec[f"{qk}d1"], rec[f"{qk}d2"] = float(d1), float(d2)
                rec[f"{qk}_answers"] = [first, second]
            out[ins["id"]] = rec
            if on_progress:
                on_progress(len(out), self.usd)
        return out


# ===================================================================================================== factory
def build_scorer(spec: dict, device: Optional[str] = None, hf_token: Optional[str] = None) -> Scorer:
    """Constructs a scorer from a notebook specification: {'name', 'type', 'repo', 'revision', ...type-specific keys}."""
    t = spec["type"]
    repo, rev, name = spec.get("repo"), spec.get("revision"), spec["name"]
    if t == "colbert_checkpoint":
        return ColBERTCheckpoint(repo, rev, device, name=name, metadata_overrides=spec.get("metadata_overrides"))
    if t in ("pylate", "st_colbert"):
        return STColBERT(repo, rev, device, name=name, document_length=spec.get("document_length"), query_length=spec.get("query_length"))
    if t == "bgem3":
        return BGEM3(repo or "BAAI/bge-m3", rev, device, name=name, max_length=spec.get("max_length", 1024))
    if t == "dense":
        return SentenceDense(repo, rev, device, name=name, query_prefix=spec.get("query_prefix", ""), doc_prefix=spec.get("doc_prefix", ""),
                             max_seq_length=spec.get("max_seq_length", 512), trust_remote_code=spec.get("trust_remote_code", False))
    if t == "splade":
        return SpladeScorer(repo, rev, device, name=name, max_length=spec.get("max_length", 512))
    if t == "cross_encoder":
        return CrossEncoderScorer(repo, rev, device, name=name, max_length=spec.get("max_length", 512), trust_remote_code=spec.get("trust_remote_code", False))
    if t == "monot5":
        return MonoT5Scorer(repo, rev, device, name=name, max_length=spec.get("max_length", 512))
    if t == "llm_pointwise":
        return LLMPointwise(repo, rev, device, name=name, max_doc_tokens=spec.get("max_doc_tokens", 700), chat_kwargs=spec.get("chat_kwargs"),
                            use_system=spec.get("use_system", True), trust_remote_code=spec.get("trust_remote_code", False), token=hf_token)
    raise ValueError(f"unknown scorer type {t!r}")


# ===================================================================================================== metrics glue
def negpairs_records(rows: Sequence[NegRow], scores: Dict[str, Sequence[float]]) -> List[dict]:
    """Prediction records in the format of the training code: argmax of the two option scores (the first option on a
    tie, as for M0-native), softmax of the two scores as pseudo-probabilities."""
    recs = []
    for r in rows:
        s0, s1 = float(scores[r.sample_id][0]), float(scores[r.sample_id][1])
        z = torch.softmax(torch.tensor([s0, s1], dtype=torch.float64), dim=0).tolist()
        recs.append({"sample_id": r.sample_id, "passage_id": r.passage_id, "question_id": r.question_id, "edit_id": r.edit_id,
                     "label": r.label, "pred": LABELS2[0 if s0 >= s1 else 1], "probs": [round(x, 6) for x in z],
                     "scores": [s0, s1], "cue": r.cue, "unseen_cue": not r.seen_family, "tie": s0 == s1})
    return recs


def perm_metrics(records: Sequence[dict], rows_by_id: Dict[str, NegRow], n_boot: int = 10000, seed: int = 20261003) -> dict:
    """The article's permutation estimands for one system (one run): flatten_perm(perm_report) plus scenario-cluster
    bootstrap intervals of SS_perm_bal (all scenarios, resampled within each clause order) and of pooled SS."""
    runs = {("system", "max", 0): list(records)}
    rep = perm_report(records, rows_by_id)
    flat = flatten_perm(rep)
    out = {"flat": flat, "report": rep,
           "SS_bal_all_ci": cluster_bootstrap_flip(runs, rows_by_id, "system", "max", None, "SS", "all", None, n_boot, seed=seed, by_order=True),
           "SS_pooled_all_ci": cluster_bootstrap_flip(runs, rows_by_id, "system", "max", None, "SS", "all", None, n_boot, seed=seed, by_order=False),
           "ties": int(sum(bool(r.get("tie")) for r in records)), "n_items": len(records)}
    return out


def fresh_metrics(records: Sequence[dict], rows_by_id: Dict[str, NegRow], n_boot: int = 10000, seed: int = 20261003) -> dict:
    runs = {("system", "max", 0): list(records)}
    return {"report": report(records, rows_by_id),
            "SS_seen_ci": cluster_bootstrap_flip(runs, rows_by_id, "system", "max", None, "SS", "seen", None, n_boot, seed=seed),
            "SS_unseen_ci": cluster_bootstrap_flip(runs, rows_by_id, "system", "max", None, "SS", "unseen", None, n_boot, seed=seed),
            "SS_all_ci": cluster_bootstrap_flip(runs, rows_by_id, "system", "max", None, "SS", "all", None, n_boot, seed=seed),
            "ties": int(sum(bool(r.get("tie")) for r in records)), "n_items": len(records)}


def nevir_metrics(scores: Dict[str, Dict[str, float]], instances: Sequence[dict], nested_ids: Sequence[str],
                  equal_ids: Sequence[str] = (), n_boot: int = 10000) -> dict:
    """evaluate_scorer (all / nested / complement, strict comparisons) plus the equal-bag subset, the number of
    instances with a tied comparison and the per-query accuracy split by subset."""
    out = evaluate_scorer(scores, instances, nested_ids, 0.0, n_boot)
    eq = set(equal_ids)
    if eq:
        sub = [i for i in instances if i["id"] in eq]
        out["equal_bags"] = {k: v for k, v in evaluate_scorer(scores, sub, [i["id"] for i in sub], 0.0, n_boot)["all"].items()}
    ties = 0
    for ins in instances:
        s = scores[ins["id"]]
        if s["q1d1"] == s["q1d2"] or s["q2d2"] == s["q2d1"]:
            ties += 1
    out["instances_with_tie"] = ties
    return out


def anchor_against_archived(new_perm: Dict[str, Sequence[float]], old_perm: Dict[str, Sequence[float]],
                            new_nevir: Dict[str, Dict[str, float]], old_nevir: Dict[str, Dict[str, float]]) -> dict:
    """Agreement of a re-scored system with an archived score file (M0-native of the main run): decision agreement on
    the NegPairs items, maximum absolute score difference, and agreement of the NevIR comparisons."""
    ids = sorted(set(new_perm) & set(old_perm))
    agree = sum((new_perm[i][0] >= new_perm[i][1]) == (old_perm[i][0] >= old_perm[i][1]) for i in ids)
    maxdiff = max((abs(float(new_perm[i][k]) - float(old_perm[i][k])) for i in ids for k in (0, 1)), default=None)
    nids = sorted(set(new_nevir) & set(old_nevir))
    same = 0
    for i in nids:
        a, b = new_nevir[i], old_nevir[i]
        same += int(((a["q1d1"] > a["q1d2"]) == (b["q1d1"] > b["q1d2"])) and ((a["q2d2"] > a["q2d1"]) == (b["q2d2"] > b["q2d1"])))
    nmax = max((abs(float(new_nevir[i][k]) - float(old_nevir[i][k])) for i in nids for k in ("q1d1", "q1d2", "q2d1", "q2d2")), default=None)
    return {"negpairs_items": len(ids), "negpairs_decision_agreement_pct": (100.0 * agree / len(ids)) if ids else None,
            "negpairs_max_abs_diff": maxdiff, "nevir_instances": len(nids),
            "nevir_comparison_agreement_pct": (100.0 * same / len(nids)) if nids else None, "nevir_max_abs_diff": nmax}


# ===================================================================================================== tables
def _f(x, nd=1):
    return "" if x is None else f"{x:.{nd}f}"


def perm_table(rows: Sequence[dict]) -> str:
    """rows: {'name', 'kind', 'perm': perm_metrics output}."""
    head = ["system", "kind", "acc_O", "acc_P", "acc_N", "acc_S", "flip_O_N", "flip_O_S", "SS", "SS_first", "SS_second", "SS_bal", "SS_bal 95% CI", "flip_O_P", "joint_NS_acc", "ties"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        f = r["perm"]["flat"]
        ci = r["perm"]["SS_bal_all_ci"]
        lines.append("| " + " | ".join([r["name"], r["kind"], _f(f["all.pooled.acc_O"]), _f(f["all.pooled.acc_P"]), _f(f["all.pooled.acc_N"]), _f(f["all.pooled.acc_S"]),
                                        _f(f["all.pooled.flip_O_N"]), _f(f["all.pooled.flip_O_S"]), _f(f["all.pooled.SS"]), _f(f["all.first.SS"]), _f(f["all.second.SS"]),
                                        _f(f["all.SS_bal"]), f"[{_f(ci.get('low'))}, {_f(ci.get('high'))}]", _f(f["all.pooled.flip_O_P"]), _f(f["all.pooled.joint_NS_acc"]),
                                        str(r["perm"]["ties"])]) + " |")
    return "\n".join(lines)


def fresh_table(rows: Sequence[dict]) -> str:
    head = ["system", "kind", "acc_O", "acc_N", "acc_S", "flip_O_N", "flip_O_S", "SS_all", "SS_all 95% CI", "SS_seen", "SS_unseen", "flip_O_P", "ties"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        rep = r["fresh"]["report"]
        a = rep["all"]
        ci = r["fresh"]["SS_all_ci"]
        lines.append("| " + " | ".join([r["name"], r["kind"], _f(a["acc_O"]), _f(a["acc_N"]), _f(a["acc_S"]), _f(a["flip_O_N"]), _f(a["flip_O_S"]), _f(a["SS"]),
                                        f"[{_f(ci.get('low'))}, {_f(ci.get('high'))}]", _f(rep["seen"]["SS"] if rep["seen"] else None),
                                        _f(rep["unseen"]["SS"] if rep["unseen"] else None), _f(a["flip_O_P"]), str(r["fresh"]["ties"])]) + " |")
    return "\n".join(lines)


def nevir_table(rows: Sequence[dict]) -> str:
    head = ["system", "kind", "paired all", "95% CI", "paired nested", "95% CI", "paired complement", "95% CI", "equal bags", "per-query all", "instances with a tie"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        n = r["nevir"]
        cell = lambda k: (_f(n[k]["paired_accuracy"]), f"[{_f(n[k]['bootstrap']['low'])}, {_f(n[k]['bootstrap']['high'])}]")
        eq = n.get("equal_bags", {}).get("paired_accuracy")
        lines.append("| " + " | ".join([r["name"], r["kind"], *cell("all"), *cell("nested"), *cell("complement"), _f(eq), _f(n["all"]["per_query_accuracy"]),
                                        str(n["instances_with_tie"])]) + " |")
    return "\n".join(lines)
