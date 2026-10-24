"""QwenVerifier tensor path on a tiny random Qwen2 model (no download): the float32 letter logits computed from the
final hidden state with left padding must equal the model's own last-position logits for each unpadded text."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from cacsa2 import qwen_verify as qv


def tiny_components():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    words = ["<unk>", "<pad>", "A", "B", "U", "Read", "the", "context", "and", "decide", "which", "of", "two", "options",
             "is", "better", "thing", "to", "do", ".", "Use", "only", "CONTEXT", ":", "QUESTION", ")", "user",
             "assistant", "museum", "open", "cafe", "busy", "not", "What", "should", "Ana", "?", "visit", "stay", "home",
             "Answer", "with", "one", "letter", "or", "for", "option", "that", "gives", "a", "reason", "prefer", ","]
    vocab = {w: i for i, w in enumerate(words)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>")
    fast.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    fast.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    cfg = Qwen2Config(vocab_size=len(fast), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).float().eval()
    return fast, model


def test_logits3_matches_unpadded_forward():
    fast, model = tiny_components()
    v = qv.QwenVerifier(tokenizer=fast, model=model, device="cpu", batch_size=3)
    texts = v.scenario_texts({"O": "The museum is open , and the cafe is busy .",
                              "P": "the museum is open",
                              "N": "The museum is not open , and the cafe is busy .",
                              "S": "The museum is open , and the cafe is not busy ."},
                             "What should Ana do ?", ["visit the museum", "stay home"])
    assert len(texts) == 8 and texts[0] != texts[1]
    got = v.logits3(texts)
    with torch.no_grad():
        for t, g in zip(texts, got):
            ids = fast(t, add_special_tokens=False, return_tensors="pt")["input_ids"]
            ref = model(input_ids=ids).logits[0, -1, v.letter_ids].float().tolist()
            assert max(abs(a - b) for a, b in zip(ref, g)) < 1e-4, (ref, g)


def test_verify_records_sets_stages():
    fast, model = tiny_components()
    v = qv.QwenVerifier(tokenizer=fast, model=model, device="cpu", batch_size=4)
    rec = {"stage": "haiku_passed"}
    arms = {"O": "museum is open", "P": "museum open", "N": "museum is not open", "S": "cafe is not busy"}
    qv.verify_records(v, [rec], lambda r: (arms, "What should Ana do ?", ["visit the museum", "stay home"], "A"))
    assert rec["stage"] in ("qwen_passed", "qwen") and set(rec["qwen"]) == {"O", "P", "N", "S"}
    assert all(len(rec["qwen"][a]["ab"]) == 3 for a in "OPNS")
