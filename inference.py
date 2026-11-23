"""Generate local Qwen travel suggestions with an optional LoRA adapter."""

import argparse
import logging
from pathlib import Path
from threading import Lock

from data_prep_chinatravel import INSTRUCTION, build_input

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
V1_ADAPTER = ROOT / "checkpoints" / "qwen2.5-0.5b-trip-lora"
V2_ADAPTER = ROOT / "checkpoints" / "qwen2.5-0.5b-trip-lora-v2"
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

_models = {}
_load_lock = Lock()
_generation_lock = Lock()


def load_model(adapter_dir: str | Path | None = V2_ADAPTER):
    """Load once per resolved adapter path. None selects the unadapted base."""
    path = Path(adapter_dir).resolve() if adapter_dir is not None else None
    with _load_lock:
        if path in _models:
            return _models[path]
        if path is not None and not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"找不到 LoRA adapter：{path}")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        logging.getLogger(__name__).info("Loading SFT model: %s", path or MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
        if path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(path))
        model.to(device).eval()
        _models[path] = tokenizer, model, device
        return _models[path]


def generate_recommendation(
    user_input: str,
    *,
    instruction: str = INSTRUCTION,
    adapter_dir: str | Path | None = V2_ADAPTER,
) -> str:
    tokenizer, model, device = load_model(adapter_dir)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{instruction}\n{user_input}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    with _generation_lock:
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=256,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = generated[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=V2_ADAPTER)
    parser.add_argument("--query", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--days", required=True, type=int)
    parser.add_argument("--people", required=True, type=int)
    args = parser.parse_args()
    if args.days < 1 or args.people < 1:
        parser.error("--days and --people must be positive")
    user_input = build_input(args.query, args.start, args.target, args.days, args.people)
    print(generate_recommendation(user_input, adapter_dir=args.adapter))


if __name__ == "__main__":
    main()
