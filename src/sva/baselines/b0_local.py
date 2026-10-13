from sva.config import load_config
from sva.evidence.builder import build_categorical
from sva.models.base import build_backend, mcq_prompt
from sva.perception.cache import PerceptionCache


class B0Local:
    """Hoi thang VLM: toan bo frame + cau hoi goc + moi option, 1 lan cham, khong phan ra."""

    name = "b0_local"

    def __init__(self, backend="qwen_local_3b", cache=None):
        cfg = load_config()["perception"]
        self.backend = build_backend(backend)
        self.cache = cache or PerceptionCache()
        self.n_single = cfg["n_frames_default"]
        self.budget = cfg["n_frames_total_multiview"]
        self.size = cfg["image_size"]

    def clips(self, sample, n_single=None, budget=None):
        return self.cache.get_clips(sample.materials, n_single or self.n_single, budget or self.budget, self.size)

    def score(self, sample, prompt=None, labels=None, n_single=None, budget=None, tool=None):
        frames, durations, timestamps = self.clips(sample, n_single, budget)
        labels = labels or sample.option_keys
        lp = self.backend.score_labels(frames, prompt or mcq_prompt(sample, labels), labels, durations)
        return build_categorical(tool or self.name, sample, lp, timestamps=timestamps)

    def __call__(self, sample):
        ev = self.score(sample)
        return {"pred": ev.pred, "dist": ev.dist, "confidence": ev.confidence, "evidence": [ev.to_json()]}
