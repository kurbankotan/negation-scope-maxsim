"""Cross-family verification with Qwen2.5-7B-Instruct (manifest v3.0, section 4.4).

Each arm is shown alone, twice (options in both orders). The verdict of one presentation is the highest of the
next-token logits of "A", "B" and "U", computed in float32 from the final hidden state and the matching rows of the
output layer (exact ties count as no verdict). The arm's verdict is the option chosen in both orders, otherwise
"inconsistent" (also when "U" wins in either order). A scenario passes if every arm gives the expected option.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .perm_prompts import QWEN_VERIFY_PROMPT

QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LETTERS = ("A", "B", "U")
ARMS = ("O", "P", "N", "S")
MIN_GPU_GIB = 22.0          # an L4 reports about 22.5 GiB; a T4 (15 GiB, no bfloat16) is refused


def _pick(lg: Sequence[float]) -> Optional[int]:
    m = max(lg)
    best = [i for i, v in enumerate(lg) if v == m]
    return best[0] if len(best) == 1 else None


def arm_verdict(lg_ab: Sequence[float], lg_ba: Sequence[float]):
    """lg_ab: logits (A, B, U) with option 0 as A; lg_ba: with option 1 as A. Returns 0, 1 or 'inconsistent'."""
    p1, p2 = _pick(lg_ab), _pick(lg_ba)
    if p1 is None or p2 is None or p1 == 2 or p2 == 2:
        return "inconsistent"
    o1, o2 = p1, 1 - p2
    return o1 if o1 == o2 else "inconsistent"


def scenario_check(logits8: Sequence[Sequence[float]], answer: str) -> Tuple[bool, Dict[str, dict]]:
    """logits8: for arms O, P, N, S in this order, the (A, B, U) logits of order ab then ba."""
    y = 0 if answer == "A" else 1
    expect = {"O": y, "P": y, "N": 1 - y, "S": y}
    out, ok = {}, True
    for k, arm in enumerate(ARMS):
        ab, ba = list(logits8[2 * k]), list(logits8[2 * k + 1])
        v = arm_verdict(ab, ba)
        out[arm] = {"ab": ab, "ba": ba, "verdict": v, "expected": expect[arm]}
        ok = ok and v == expect[arm]
    return ok, out


class QwenVerifier:
    """Loads the model in bfloat16 on the GPU. `render` builds the chat prompt; `logits3` returns float32 logits."""

    def __init__(self, model_name: str = QWEN_MODEL, revision: Optional[str] = None, device: str = "cuda",
                 batch_size: int = 16, tokenizer=None, model=None):
        import torch
        self.torch = torch
        self.model_name, self.device, self.batch_size = model_name, device, batch_size
        if tokenizer is None or model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if device.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise RuntimeError("no GPU: choose a GPU runtime (L4 or A100)")
                props = torch.cuda.get_device_properties(0)
                gib = props.total_memory / 1024 ** 3
                if not torch.cuda.is_bf16_supported() or gib < MIN_GPU_GIB:
                    raise RuntimeError(f"GPU {props.name} ({gib:.1f} GiB) is not supported: an L4 or an A100 is "
                                       f"required (bfloat16 and at least {MIN_GPU_GIB:.0f} GiB)")
            import transformers
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
            # transformers 5 takes `dtype`; version 4 silently ignores it and would load float32 (30 GB), so pick by version
            kw = ({"dtype": torch.bfloat16} if int(transformers.__version__.split(".")[0]) >= 5
                  else {"torch_dtype": torch.bfloat16})
            model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, device_map=device, **kw)
            got = next(model.parameters()).dtype
            if got != torch.bfloat16:
                raise RuntimeError(f"Qwen loaded in {got}, expected bfloat16")
        self.tok, self.model = tokenizer, model
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model.eval()
        ids = {L: self.tok.encode(L, add_special_tokens=False) for L in LETTERS}
        bad = [L for L, v in ids.items() if len(v) != 1]
        if bad:
            raise RuntimeError(f"letters {bad} are not single tokens of the tokenizer")
        self.letter_ids = [ids[L][0] for L in LETTERS]
        head = self.model.get_output_embeddings()
        self.W3 = head.weight[self.letter_ids].detach().float()                       # (3, hidden)
        bias = getattr(head, "bias", None)
        self.b3 = bias[self.letter_ids].detach().float() if bias is not None else None
        base = getattr(self.model, "model", None)
        self.base = base if base is not None else self.model.get_decoder()
        self.revision = revision or getattr(self.model.config, "_commit_hash", None) or self._cached_revision(model_name)

    @staticmethod
    def _cached_revision(model_name: str) -> Optional[str]:
        """Commit hash of the snapshot that the Hugging Face cache resolved for this model (the parent folder of its
        config.json is named by the commit). transformers 5 no longer sets config._commit_hash, which left the
        revision unrecorded in the v3.0 build; this fallback closes that gap."""
        try:
            from transformers.utils import cached_file
            import re
            path = cached_file(model_name, "config.json")
            m = re.search(r"/snapshots/([0-9a-f]{40})/", str(path).replace("\\", "/"))
            return m.group(1) if m else None
        except Exception:                                   # noqa: BLE001
            return None

    def render(self, context: str, question: str, option_a: str, option_b: str) -> str:
        msg = [{"role": "user", "content": QWEN_VERIFY_PROMPT.format(context=context, question=question,
                                                                      option_a=option_a, option_b=option_b)}]
        return self.tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    def logits3(self, texts: List[str]) -> List[List[float]]:
        torch = self.torch
        out = []
        dev = next(self.model.parameters()).device
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(texts[i:i + self.batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
                ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
                pos = (am.long().cumsum(-1) - 1).clamp(min=0)
                h = self.base(input_ids=ids, attention_mask=am, position_ids=pos).last_hidden_state[:, -1, :].float()
                lg = h @ self.W3.T
                if self.b3 is not None:
                    lg = lg + self.b3
                out.extend(lg.cpu().tolist())
        return out

    def scenario_texts(self, arms: Dict[str, str], question: str, options: Sequence[str]) -> List[str]:
        texts = []
        for arm in ARMS:
            texts.append(self.render(arms[arm], question, options[0], options[1]))
            texts.append(self.render(arms[arm], question, options[1], options[0]))
        return texts


def verify_records(verifier, records: List[dict], arms_of) -> None:
    """Runs Qwen on every record (stage 'haiku_passed') and sets its stage to 'qwen_passed' or 'qwen'.
    arms_of(record) -> (arms, question, options, answer)."""
    texts, owners = [], []
    for r in records:
        arms, question, options, _ = arms_of(r)
        texts.extend(verifier.scenario_texts(arms, question, options))
        owners.append(r)
    if not texts:
        return
    logits = verifier.logits3(texts)
    for k, r in enumerate(owners):
        _, _, _, answer = arms_of(r)
        ok, detail = scenario_check(logits[8 * k: 8 * k + 8], answer)
        r["qwen"] = detail
        if ok:
            r["stage"] = "qwen_passed"
        else:
            bad = next(a for a in ARMS if detail[a]["verdict"] != detail[a]["expected"])
            r["stage"], r["reason"] = "qwen", f"qwen arm {bad} read as {detail[bad]['verdict']!r}"
