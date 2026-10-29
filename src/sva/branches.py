"""Ba nhanh cua SVA: A (q10, q11: cong cu chuyen dung), B (q8, q13: VLM + kenh phu), C (q9 replay, q14 nhieu goc)."""
import hashlib
import json
import os
import re
from dataclasses import replace

import cv2
import numpy as np

from sva.config import CACHE_DIR, CAMERA_LABELS, ORDINAL_SCALES
from sva.data import canon, parse_views
from sva.evidence import (Evidence, build_categorical, build_positional, hard_vote, linear_pool, log_pool,
                          ordinal_fusion)
from sva.models import CLASS_NAMES, LETTERS, ActionClassifier, margin_confidence, mcq_prompt, softmax
from sva.perception import build_encoder


# ---------------------------------------------------------------- So khop replay (q9)
ORDINAL = re.compile(r"(\d+)(?:st|nd|rd|th)")


def strip_bars(frames, size=224):
    """Bo vien den cua anh letterbox roi dua ve size x size (giu toan bo noi dung, khong cat mep)."""
    lit = frames.max(axis=(0, 3)) > 8
    rows, cols = np.where(lit.any(axis=1))[0], np.where(lit.any(axis=0))[0]
    if len(rows) == 0:
        return np.stack([cv2.resize(f, (size, size)) for f in frames])
    crop = frames[:, rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in crop])


class FrameEncoder:
    """Encoder chi nap model khi cache thieu. square: bo vien den truoc (ten cache rieng, them _sq)."""

    def __init__(self, encoder_name, square=False, device="cuda"):
        self.base_name, self.square, self.device, self._enc = encoder_name, square, device, None
        self.name = f"{encoder_name}_sq" if square else encoder_name

    def encode(self, frames):
        if self._enc is None:
            self._enc = build_encoder(self.base_name, self.device)
        return self._enc.encode(strip_bars(frames) if self.square else frames)


def candidate_order(sample):
    """option_keys -> chi so ung vien (1..4) doc tu 'The 1st one.' ..."""
    out = []
    for k in sample.option_keys:
        m = ORDINAL.search(sample.option_texts[k])
        if not m:
            raise ValueError(f"khong doc duoc thu tu ung vien: {sample.option_texts[k]!r}")
        out.append(int(m.group(1)))
    return out


def chamfer(replay, cand, lam):
    """replay [N,d], cand [M,d] da chuan hoa -> (score', score, Omega, max cos)."""
    s = replay @ cand.T
    best = s.argmax(axis=1)
    omega = float(np.mean(best[1:] < best[:-1])) if len(best) > 1 else 0.0
    score = float(s.max(axis=1).mean())
    return score - lam * omega, score, omega, float(s.max())


class ReplayMatcher:
    name = "replay_match"

    def __init__(self, cache, encoder="clip_vit_l", n_frames=24, size=336, square=False, lam=0.0, temperature=0.01,
                 device="cuda"):
        self.cache, self.encoder = cache, FrameEncoder(encoder, square, device)
        self.n, self.size, self.lam, self.temperature = n_frames, size, lam, temperature

    def embed(self, path):
        return self.cache.get_embedding(path, self.n, self.size, self.encoder).astype(np.float32)

    def __call__(self, sample):
        replay = self.embed(sample.materials[0])
        rows = [chamfer(replay, self.embed(sample.materials[c]), self.lam) for c in candidate_order(sample)]
        obs = " | ".join(f"{k}: {r[1]:.3f} (lui {r[2]:.2f})" for k, r in zip(sample.option_keys, rows))
        ev = build_positional(self.name, sample, [r[0] for r in rows], raw_obs=obs, T=self.temperature)
        ev.material_ref["max_cos"] = [round(r[3], 4) for r in rows]
        return ev


# ---------------------------------------------------------------- Nhieu goc quay (q14)
SPEED = re.compile(r"[\d.]+x speed[.,]?\s*")
ACTIONS = {"challenge", "dive", "elbowing", "high leg", "holding", "pushing", "standing tackling", "tackling"}
YES_NO = [("multiple", ("multiple", "more than one", "several", "concurrent", "multi-foul")),
          ("handball", ("handball", "hand/arm", "hand contact", "handle the ball")),
          ("touch_ball", ("touch", "ball contact", "contact with the ball", "ball involved", "ball part")),
          ("try_to_play", ("attempt", "try", "trying", "intention", "effort", "make a play", "intercept"))]
FUSIONS = {"hard_vote": hard_vote, "linear_pool": linear_pool, "log_pool": log_pool}
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
            return ordinal_fusion(dists, scale, w), dim
        return FUSIONS[self.fusion](dists, w), dim

    def __call__(self, sample):
        views = self.views(sample)
        dist, dim = self.fuse(sample, [ev.dist for ev, _ in views], [q for _, q in views])
        obs = "; ".join(f"{ev.tool_name} {ev.pred} ro {q:.2f}" for ev, q in views)
        return Evidence(self.name, "ordinal" if ordinal_scale(sample, dim) else "categorical", sample.option_keys,
                        dist, margin_confidence(dist), f"{dim}: {obs}",
                        {"clips": sample.materials, "views": [ev.dist for ev, _ in views],
                         "quality": [round(q, 4) for _, q in views]},
                        ordinal_scale=ordinal_scale(sample, dim))


# ---------------------------------------------------------------- Nhanh A (q10, q11)
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
        p = softmax(backend.score_chat(msg, list(LETTERS[:len(CLASS_NAMES)])))
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


# ---------------------------------------------------------------- Nhanh B (q8, q13)
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
        lp = self.scorer.backend.score_labels(frames, CAMERA_PROMPT, list(LETTERS[:len(CAMERA_LABELS)]), duration)
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


# ---------------------------------------------------------------- Nhanh C (q9, q14)
class BranchC:
    name = "branch_c"
    TASKS = {"q9", "q14"}

    def __init__(self, scorer, cache, matching, multiview=None, budget=24):
        mv = dict(multiview or {})
        self.scorer, self.with_holistic = scorer, mv.pop("with_holistic", False)
        self.matcher = ReplayMatcher(cache, **matching)
        self.multiview = MultiView(scorer, budget, **mv)

    def supports(self, task_id):
        return task_id in self.TASKS

    def run(self, sample):
        if sample.task_id == "q9":
            return [self.matcher(sample)]
        if sample.task_id == "q14":
            return ([self.scorer.score(sample)] if self.with_holistic else []) + [self.multiview(sample)]
        raise NotImplementedError(f"nhanh C chua co cho {sample.task_id}")
