"""M0-native: ColBERTv2's own late-interaction scoring with the checkpoint's settings (manifest v3.0, section 5.4).

Settings of colbert-ir/colbertv2.0 (artifact.metadata): query_maxlen 32 with [MASK] augmentation and no attention to
the mask tokens, doc_maxlen 180, dim 128, cosine similarity, punctuation masked in documents. As in the ColBERT code:
  query:    "[CLS] [Q] tokens [SEP]" padded to 32 with [MASK]; attention mask 1 for real tokens only; every one of the
            32 output vectors (projected and L2-normalized) takes part in the score;
  document: "[CLS] [D] tokens [SEP]" truncated to 180; projected; vectors of punctuation tokens zeroed; L2-normalized
            (a zero vector stays zero), special tokens kept;
  score:    sum over the 32 query vectors of the maximum over document vectors of the dot product.
"""
from __future__ import annotations

import string
from typing import List, Optional, Sequence

import torch


class ColBERTNative:
    def __init__(self, model_name: str, projection: torch.Tensor, device: Optional[str] = None,
                 dtype: torch.dtype = torch.float32, revision: Optional[str] = None, query_maxlen: int = 32,
                 doc_maxlen: int = 180, query_marker: str = "[unused0]", doc_marker: str = "[unused1]",
                 tokenizer=None, model=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if tokenizer is None or model is None:
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
            try:
                model = AutoModel.from_pretrained(model_name, revision=revision, dtype=dtype)
            except TypeError:
                model = AutoModel.from_pretrained(model_name, revision=revision, torch_dtype=dtype)
        self.tok = tokenizer
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.W = projection.to(self.device, dtype=dtype)
        self.query_maxlen, self.doc_maxlen = query_maxlen, doc_maxlen
        self.q_id = int(self.tok.convert_tokens_to_ids(query_marker))
        self.d_id = int(self.tok.convert_tokens_to_ids(doc_marker))
        self.skiplist = {self.tok.convert_tokens_to_ids(s) for s in string.punctuation} | \
                        {self.tok.convert_tokens_to_ids("##" + s) for s in string.punctuation}
        self.skiplist = {i for i in self.skiplist if i is not None and i != self.tok.unk_token_id}
        self.settings = {"query_maxlen": query_maxlen, "doc_maxlen": doc_maxlen, "dim": int(projection.shape[0]),
                         "similarity": "cosine", "attend_to_mask_tokens": False, "mask_punctuation": True,
                         "query_marker": query_marker, "doc_marker": doc_marker}

    def _ids(self, texts: Sequence[str], marker: int, maxlen: int, pad_with_mask: bool):
        enc = self.tok(list(texts), add_special_tokens=False, truncation=True, max_length=maxlen - 3)
        ids_list = [[self.tok.cls_token_id, marker] + ids + [self.tok.sep_token_id] for ids in enc["input_ids"]]
        n = maxlen if pad_with_mask else max(len(x) for x in ids_list)
        fill = self.tok.mask_token_id if pad_with_mask else self.tok.pad_token_id
        ids = torch.full((len(texts), n), fill, dtype=torch.long)
        attn = torch.zeros(len(texts), n, dtype=torch.long)
        truncated = []
        for b, x in enumerate(ids_list):
            ids[b, : len(x)] = torch.tensor(x)
            attn[b, : len(x)] = 1
            truncated.append(len(self.tok(texts[b], add_special_tokens=False)["input_ids"]) > maxlen - 3)
        return ids, attn, truncated

    @torch.no_grad()
    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        """(B, query_maxlen, dim), every position used."""
        ids, attn, _ = self._ids(texts, self.q_id, self.query_maxlen, pad_with_mask=True)
        h = self.model(input_ids=ids.to(self.device), attention_mask=attn.to(self.device)).last_hidden_state
        return torch.nn.functional.normalize(h @ self.W.T, dim=-1).float()

    @torch.no_grad()
    def encode_docs(self, texts: Sequence[str]):
        """(B, n, dim) with zeroed punctuation and padding vectors, plus the truncation flags."""
        ids, attn, truncated = self._ids(texts, self.d_id, self.doc_maxlen, pad_with_mask=False)
        h = self.model(input_ids=ids.to(self.device), attention_mask=attn.to(self.device)).last_hidden_state
        d = h @ self.W.T
        keep = attn.bool().to(self.device)
        for s in self.skiplist:
            keep &= ids.to(self.device) != s
        d = d * keep.unsqueeze(-1).to(d.dtype)
        return torch.nn.functional.normalize(d, dim=-1).float(), truncated

    @staticmethod
    def score(Q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        """Q (B, q, dim), D (B, n, dim) -> (B,): sum over query vectors of the max over document vectors."""
        return torch.matmul(Q, D.transpose(1, 2)).max(dim=2).values.sum(dim=1)

    @torch.no_grad()
    def score_pairs(self, queries: Sequence[str], docs: Sequence[str], batch_size: int = 32) -> List[float]:
        out = []
        for i in range(0, len(queries), batch_size):
            Q = self.encode_queries(queries[i:i + batch_size])
            D, _ = self.encode_docs(docs[i:i + batch_size])
            out.extend(self.score(Q, D).cpu().tolist())
        return out
