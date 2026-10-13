"""Nhanh A (mot clip, can cong cu chuyen dung): q10 loai su kien tu UniSoccer; q11 khop binh luan theo loai su kien.

q11: P(phuong an k) ~ sum_c P_video(c) * P(binh luan k noi ve su kien c)
     P_video: UniSoccer tren clip; P(.|binh luan): VLM doc rieng cau binh luan (chi chu), cham 23 nhan.
"""
import hashlib
import json
import os

import numpy as np

from sva.constants import CACHE_DIR
from sva.data.loader import canon
from sva.evidence.builder import build_categorical
from sva.models.scoring import softmax
from sva.models.unisoccer import CLASS_NAMES

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVW"
EVENT_PROMPT = ("Here is a sentence of football commentary: \"{text}\"\nWhich event does it describe?\nOptions:\n"
                + "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, CLASS_NAMES))
                + "\nAnswer with the option letter only.")


def option_classes(sample):
    """q10: chi so lop UniSoccer cua tung phuong an."""
    return [CLASS_NAMES.index(canon(sample.option_texts[k])) for k in sample.option_keys]


def top_events(p, k=3):
    return ", ".join(f"{CLASS_NAMES[i]} {p[i]:.2f}" for i in np.argsort(-p)[:k])


class BranchA:
    name = "branch_a"
    TASKS = {"q10", "q11"}

    def __init__(self, scorer, classifier=None, with_holistic=(), eps=1e-4):
        from sva.models.unisoccer import ActionClassifier
        self.scorer, self.classifier = scorer, classifier or ActionClassifier()
        self.with_holistic, self.eps = set(with_holistic), eps
        self.cache_dir = CACHE_DIR / "tools" / "caption_event"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def supports(self, task_id):
        return task_id in self.TASKS

    def video_events(self, path):
        p = self.classifier.probs(path)
        return np.array([p[c] for c in CLASS_NAMES])

    def caption_events(self, text):
        """Phan phoi 23 loai su kien ma cau binh luan mo ta (cache theo model + cau)."""
        backend = self.scorer.backend
        f = self.cache_dir / f"{hashlib.sha1(f'{backend.name}|{text}'.encode()).hexdigest()}.json"
        if f.exists():
            return np.array(json.loads(f.read_text(encoding="utf-8")))
        msg = [{"role": "user", "content": [{"type": "text", "text": EVENT_PROMPT.format(text=text)}]}]
        p = softmax(backend.score_chat(msg, list(LETTERS)))
        tmp = f.with_name(f"{f.stem}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(p.tolist()), encoding="utf-8")
        os.replace(tmp, f)
        return p

    def run(self, sample):
        pv = self.video_events(sample.materials[0])
        if sample.task_id == "q10":
            ev = build_categorical("action_classifier", sample, np.log(pv[option_classes(sample)] + self.eps),
                                   raw_obs=top_events(pv))
        else:
            match = [pv @ self.caption_events(sample.option_texts[k]) for k in sample.option_keys]
            ev = build_categorical("event_match", sample, np.log(np.array(match) + self.eps),
                                   raw_obs=f"video: {top_events(pv)}")
        return ([self.scorer.score(sample)] if sample.task_id in self.with_holistic else []) + [ev]
