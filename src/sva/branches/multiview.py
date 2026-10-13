"""Nhanh C, q14: hoi VLM rieng tung goc quay, gop phan phoi cac goc thay vi bo phieu nhan cung."""
import re
from dataclasses import replace

import numpy as np

from sva.constants import ORDINAL_SCALES
from sva.data.loader import canon, parse_views
from sva.evidence import fusion
from sva.models.scoring import margin_confidence, softmax
from sva.schema import Evidence

SPEED = re.compile(r"[\d.]+x speed[.,]?\s*")
ACTIONS = {"challenge", "dive", "elbowing", "high leg", "holding", "pushing", "standing tackling", "tackling"}
YES_NO = [("multiple", ("multiple", "more than one", "several", "concurrent", "multi-foul")),
          ("handball", ("handball", "hand/arm", "hand contact", "handle the ball")),
          ("touch_ball", ("touch", "ball contact", "contact with the ball", "ball involved", "ball part")),
          ("try_to_play", ("attempt", "try", "trying", "intention", "effort", "make a play", "intercept"))]
QUALITY_PROMPT = ("Question: Is the foul clearly visible in this video, including any contact between the players?\n"
                  "Options:\nA. Yes\nB. No\nAnswer with the option letter only.")


def question_part(question):
    """Bo phan liet ke goc quay o dau cau, giu cau hoi chinh."""
    return SPEED.split(question)[-1].strip()


def classify_dimension(sample):
    """Chieu duoc hoi, doc tu bo phuong an (chinh xac) va cau hoi chinh (cho cac cau Yes/No)."""
    texts = {canon(t) for t in sample.option_texts.values()}
    if texts & set(ORDINAL_SCALES["severity"]):
        return "severity"
    if texts <= set(ORDINAL_SCALES["offence"]):
        return "offence"
    if texts <= {"with contact", "without contact"}:
        return "contact"
    if texts <= {"upper body", "under body"}:
        return "bodypart"
    if texts <= {"arms", "shoulder", "use of arms", "use of shoulder"}:
        return "upper_bodypart"
    if texts <= ACTIONS:
        return "action"
    if texts <= {"yes", "no", "maybe"}:
        q = question_part(sample.question).lower()
        return next((dim for dim, words in YES_NO if any(w in q for w in words)), "yes_no")
    return "other"


def ordinal_scale(sample, dim):
    scale = ORDINAL_SCALES.get(dim)
    return {k: scale[canon(sample.option_texts[k])] for k in sample.option_keys} if scale else None


class MultiView:
    """Moi goc: phan phoi tren phuong an + do ro (VLM tu danh gia) -> gop co trong so."""
    name = "multiview"

    def __init__(self, scorer, budget=24, fusion="log_pool", quality=True, ordinal=True):
        self.scorer, self.budget = scorer, budget
        self.fusion, self.quality, self.ordinal = fusion, quality, ordinal

    def views(self, sample):
        """[(Evidence cua goc v, P(thay ro))], cung so frame moi goc nhu khi dua ca mau vao B0."""
        cams = parse_views(sample.question)
        n = max(2, self.budget // len(sample.materials) // 2 * 2)
        out = []
        for i, path in enumerate(sample.materials):
            cam = cams[i]["camera"] if i < len(cams) else "another"
            sub = replace(sample, materials=[path],
                          question=f"This video shows the foul from the {cam} view. {question_part(sample.question)}")
            ev = self.scorer.score(sub, n_single=n, tool=f"view{i}")
            frames, durations, _ = self.scorer.clips(sub, n_single=n)
            lp = self.scorer.backend.score_labels(frames, QUALITY_PROMPT, ["A", "B"], durations)
            out.append((ev, float(softmax(lp)[0])))
        return out

    def fuse(self, sample, dists, quality):
        dim = classify_dimension(sample)
        w = quality if self.quality else None
        scale = ordinal_scale(sample, dim)
        if self.ordinal and scale:
            return fusion.ordinal_fusion(dists, scale, w), dim
        return getattr(fusion, self.fusion)(dists, w), dim

    def __call__(self, sample):
        views = self.views(sample)
        dist, dim = self.fuse(sample, [ev.dist for ev, _ in views], [q for _, q in views])
        obs = "; ".join(f"{ev.tool_name} {ev.pred} ro {q:.2f}" for ev, q in views)
        return Evidence(self.name, "ordinal" if ordinal_scale(sample, dim) else "categorical", sample.option_keys,
                        dist, margin_confidence(dist), f"{dim}: {obs}",
                        {"clips": sample.materials, "views": [ev.dist for ev, _ in views],
                         "quality": [round(q, 4) for _, q in views]},
                        ordinal_scale=ordinal_scale(sample, dim))
