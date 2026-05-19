# motion_qa/hf_llm.py
# Local HuggingFace LLM — replaces OpenAI in planner.py and answerer.py.
# Model: microsoft/Phi-3-mini-4k-instruct (MIT license, ~4 GB at 4-bit).
# First run downloads the model weights from HuggingFace Hub.

from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from motion_qa.config import HF_MODEL_ID

_tokenizer = None
_model = None


def _load_model() -> tuple:
    """Lazy singleton — downloads and caches the model on first call."""
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    print(f"[hf_llm] Loading model {HF_MODEL_ID} (first run downloads weights)…")

    _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)

    # Use 4-bit quantization when a CUDA GPU is available; CPU float32 otherwise.
    if torch.cuda.is_available():
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            quantization_config=quant_cfg,
            device_map="auto",
        )
    else:
        print("[hf_llm] No CUDA GPU found — loading in float32 on CPU (slow).")
        _model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            dtype=torch.float32,
            device_map="cpu",
        )

    _model.eval()
    print("[hf_llm] Model loaded.")
    return _tokenizer, _model


def generate(
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    """
    Run Phi-3 inference with a system + user message pair.
    Returns the model's response text (stripped).
    """
    tokenizer, model = _load_model()

    # Phi-3 chat template: <|system|>…<|end|>\n<|user|>…<|end|>\n<|assistant|>
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()
