"""Nhanh B (q8, q13): VLM thay du frame, cau hoi goc va moi phuong an trong 1 luot.

q8 them kenh "camera dau / cuoi": moi phuong an co dang "From X to Y" nen cham rieng
P(camera dau clip = X) va P(camera cuoi clip = Y) tren chinh cac frame do, giu nguyen phan phoi
(khong chot nhan cung, khong qua buoc cat canh).
"""
import re

import numpy as np

from sva.constants import CAMERA_LABELS
from sva.data.loader import canon
from sva.evidence.builder import build_categorical
from sva.models.base import mcq_prompt
from sva.models.scoring import softmax

LETTERS = "ABCDEFGHIJKLM"
CAMERA_PROMPT = ("These frames come from a soccer broadcast. Which camera position is used?\nOptions:\n"
                 + "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, CAMERA_LABELS))
                 + "\nAnswer with the option letter only.")


# goi y theo task, chon tren dev (q13: 40.5 -> 44.3)
HINTS = {"q13": "First find the player who performs the action described in the question, "
                "then look at the color of that player's jersey.\n"}


def parse_switch(sample):
    """[(X, Y)] theo thu tu option neu moi option deu dang 'From X to Y', nguoc lai None."""
    index = {canon(c): i for i, c in enumerate(CAMERA_LABELS)}
    pairs = []
    for k in sample.option_keys:
        m = re.fullmatch(r"From (.+?) to (.+?)\.?", sample.option_texts[k].strip())
        if not m or canon(m.group(1)) not in index or canon(m.group(2)) not in index:
            return None
        pairs.append((index[canon(m.group(1))], index[canon(m.group(2))]))
    return pairs


class BranchB:
    name = "branch_b"

    def __init__(self, scorer, edge_frames=3):
        self.scorer = scorer              # B0Local: dung chung backend va cache frame
        self.edge = edge_frames           # so frame dau / cuoi clip dung cho kenh camera

    def run(self, sample):
        prompt = HINTS.get(sample.task_id, "") + mcq_prompt(sample)
        evs = [self.scorer.score(sample, prompt=prompt, tool="holistic")]
        pairs = parse_switch(sample) if sample.task_id == "q8" else None
        if pairs:
            evs.append(self.camera_switch(sample, pairs))
        return evs

    def camera_dist(self, frames, duration):
        lp = self.scorer.backend.score_labels(frames, CAMERA_PROMPT, list(LETTERS), duration)
        return softmax(lp)

    def camera_switch(self, sample, pairs):
        frames, durations, timestamps = self.scorer.clips(sample)
        f, d, k = frames[0], durations[0], self.edge
        seg = d * k / len(f)
        start, end = self.camera_dist(f[:k], seg), self.camera_dist(f[-k:], seg)
        lp = [np.log(start[x] + 1e-9) + np.log(end[y] + 1e-9) for x, y in pairs]
        top = lambda p: f"{CAMERA_LABELS[int(np.argmax(p))]} {p.max():.2f}"
        return build_categorical("camera_switch", sample, lp, raw_obs=f"dau: {top(start)} | cuoi: {top(end)}",
                                 timestamps=timestamps[0][:k] + timestamps[0][-k:])
