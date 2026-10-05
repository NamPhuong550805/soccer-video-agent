from functools import lru_cache
from typing import Protocol

import numpy as np

from sva.config import load_config


class VLMBackend(Protocol):
    name: str

    def score_labels(self, frames: np.ndarray, prompt: str, labels: list[str]) -> np.ndarray:
        """Log-likelihood (da chia theo so token) cua tung label."""
        ...

    def generate(self, frames: np.ndarray, prompt: str, **kw) -> str:
        ...


def mcq_prompt(sample, labels=None):
    """labels mac dinh la key goc (O1, O3...); truyen ["A", "B"] neu muon danh lai bang chu cai."""
    labels = labels or sample.option_keys
    lines = [f"Question: {sample.question}", "Options:"]
    lines += [f"{label}. {sample.option_texts[k]}" for label, k in zip(labels, sample.option_keys)]
    lines.append("Answer with the option key only." if labels == sample.option_keys
                 else "Answer with the option letter only.")
    return "\n".join(lines)


@lru_cache(maxsize=None)
def build_backend(name=None):
    """Nap model 1 lan, cac lan sau dung lai."""
    cfg = load_config("models")
    name = name or cfg["default_vlm"]
    c = cfg["vlm"][name]
    if c["backend"] == "qwen_local":
        from sva.models.qwen_local import QwenVLBackend
        return QwenVLBackend(name, c["model_id"], quantization=c.get("quantization"),
                             max_new_tokens=c.get("max_new_tokens", 128))
    raise ValueError(f"chua ho tro backend {c['backend']!r}")
