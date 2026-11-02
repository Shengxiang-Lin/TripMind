from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .models import POI, UserProfile
from .preference_prompt import PROMPT_VERSION, build_runtime_prompts

CHECKPOINT_CONFIG = "preference_config.json"
HEAD_FILE = "score_head.safetensors"


def resolve_device(value: str | None = None) -> torch.device:
    requested = (value or "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def preferred_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _hidden_size(config) -> int:
    for key in ("hidden_size", "d_model", "n_embd"):
        value = getattr(config, key, None)
        if value:
            return int(value)
    raise ValueError("Could not infer backbone hidden size")


class PreferenceScoringModel(nn.Module):
    """Transformer backbone plus a scalar preference head."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        hidden_size = _hidden_size(backbone.config)
        self.score_head = nn.Linear(hidden_size, 1, bias=True)

    @classmethod
    def from_base_model(
        cls,
        model_name: str,
        *,
        device: torch.device,
        gradient_checkpointing: bool = False,
    ) -> "PreferenceScoringModel":
        dtype = preferred_dtype(device)
        backbone = AutoModel.from_pretrained(model_name, dtype=dtype)
        if gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()
        if hasattr(backbone.config, "use_cache"):
            backbone.config.use_cache = False
        model = cls(backbone)
        return model.to(device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        last_index = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
        batch_index = torch.arange(hidden.size(0), device=hidden.device)
        pooled = hidden[batch_index, last_index]
        return self.score_head(pooled.float()).squeeze(-1)

    def save_checkpoint(
        self,
        output_dir: str | Path,
        tokenizer,
        *,
        base_model: str,
        max_length: int,
        history_char_limit: int,
        extra: dict | None = None,
    ) -> None:
        root = Path(output_dir)
        if root.exists():
            shutil.rmtree(root)
        (root / "backbone").mkdir(parents=True, exist_ok=True)
        (root / "tokenizer").mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(root / "backbone", safe_serialization=True)
        tokenizer.save_pretrained(root / "tokenizer")
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in self.score_head.state_dict().items()},
            str(root / HEAD_FILE),
        )
        payload = {
            "schema_version": 1,
            "prompt_version": PROMPT_VERSION,
            "base_model": base_model,
            "max_length": int(max_length),
            "history_char_limit": int(history_char_limit),
            "score_semantics": "unbounded_pairwise_preference_logit",
            "runtime_normalization": "candidate_set_zscore_then_sigmoid",
        }
        if extra:
            payload.update(extra)
        (root / CHECKPOINT_CONFIG).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: torch.device,
    ) -> tuple["PreferenceScoringModel", object, dict]:
        root = Path(checkpoint)
        config_path = root / CHECKPOINT_CONFIG
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "left"

        dtype = preferred_dtype(device)
        backbone = AutoModel.from_pretrained(root / "backbone", dtype=dtype)
        if hasattr(backbone.config, "use_cache"):
            backbone.config.use_cache = False
        model = cls(backbone)
        head_state = load_file(str(root / HEAD_FILE), device="cpu")
        model.score_head.load_state_dict(head_state)
        model.to(device)
        return model, tokenizer, metadata


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    return tokenizer


def normalize_candidate_scores(raw_scores: Sequence[float]) -> list[float]:
    values = [float(x) for x in raw_scores]
    if not values:
        return []
    if len(values) == 1:
        return [0.5]
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = math.sqrt(variance)
    if std < 1e-8:
        return [0.5 for _ in values]
    return [1.0 / (1.0 + math.exp(-(x - mean) / std)) for x in values]


class PreferenceScorer:
    """Inference wrapper used by TripMind's runtime ranker."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int | None = None,
        history_char_limit: int | None = None,
    ):
        self.checkpoint = str(checkpoint)
        self.device = resolve_device(device)
        self.model, self.tokenizer, metadata = PreferenceScoringModel.load_checkpoint(
            checkpoint, device=self.device
        )
        self.metadata = metadata
        self.batch_size = max(1, int(batch_size))
        self.max_length = int(max_length or metadata.get("max_length", 1024))
        self.history_char_limit = int(
            history_char_limit or metadata.get("history_char_limit", 6000)
        )
        self.model.eval()

    @torch.inference_mode()
    def score_prompts(self, prompts: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = list(prompts[start : start + self.batch_size])
            batch = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits = self.model(batch["input_ids"], batch["attention_mask"])
            scores.extend(float(x) for x in logits.detach().float().cpu())
        return scores

    def score_pois(self, profile: UserProfile, candidates: Sequence[POI]) -> list[float]:
        prompts = build_runtime_prompts(
            profile,
            candidates,
            history_char_limit=self.history_char_limit,
        )
        # Category-only prompts repeat across many POIs. Score each distinct
        # input once, then preserve the original POI order for downstream blending.
        unique_prompts = list(dict.fromkeys(prompts))
        unique_scores = self.score_prompts(unique_prompts)
        if len(unique_scores) != len(unique_prompts):
            raise RuntimeError("Preference scorer returned the wrong number of scores")
        score_by_prompt = dict(zip(unique_prompts, unique_scores))
        return [score_by_prompt[prompt] for prompt in prompts]
