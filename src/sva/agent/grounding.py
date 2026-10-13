"""Grounding Agent: dung Sample chuan cho loi SVA khi khong co phuong an (Open) hoac tu video that.

Open: moi loai cau hoi co mot danh muc nhan (23 su kien, 169 cap camera, mau ao, nhan cua chieu loi q14...).
Danh muc lon thi loc truoc con 4 ung vien bang chinh cong cu cua SVA, roi loi SVA chay nhu cau trac nghiem.
Cau tra loi la nhan; neu co phuong an goc thi anh xa nhan vao phuong an (buoc Answer Selection nhu paper).
"""
import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from sva.constants import CAMERA_LABELS
from sva.data.loader import canon
from sva.models.base import mcq_prompt
from sva.models.scoring import softmax
from sva.models.unisoccer import CLASS_NAMES
from sva.perception.video import probe, segment, split_segment

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ORDINALS = ["1st", "2nd", "3rd"] + [f"{i}th" for i in range(4, 21)]
# mau ao: danh muc lay tu cac phuong an q13 tren dev (khong dung test)
COLORS = ["blue", "white", "red", "yellow", "black", "green", "pink", "orange", "purple", "gray", "brown", "maroon",
          "navy", "light blue", "dark blue", "violet", "cyan", "teal", "turquoise", "gold", "silver", "lime", "magenta",
          "olive", "coral", "lavender"]
# q14: nhan cua tung chieu (dung cac bo phuong an cua SoccerBench / SoccerNet-MVFoul, bo cac phuong an nhieu)
DIM_LABELS = {"severity": ["No card", "No card or yellow card", "Yellow card", "Yellow card or red card", "Red card"],
              "offence": ["Offence", "Between", "No offence"],
              "contact": ["With contact", "Without contact"],
              "bodypart": ["Upper body", "Under body"],
              "upper_bodypart": ["Use of arms", "Use of shoulder"],
              "action": ["Standing tackling", "Tackling", "Challenge", "Holding", "Elbowing", "High leg", "Pushing", "Dive"],
              "handball": ["No", "Yes"], "multiple": ["No", "Yes"], "try_to_play": ["No", "Yes"],
              "touch_ball": ["No", "Yes", "Maybe"]}
# chieu duoc hoi doc tu cau hoi (khi khong co phuong an); thu tu chon tren dev:
# "make contact on the body" la phan than, "hand contact" la tay, "ball contact" la cham bong
DIM_RULES = [("multiple", r"multiple|several|more than one|multi-foul|concurrent"),
             ("handball", r"handball|hand/arm|hand contact|handle the ball"),
             ("touch_ball", r"touch|ball contact|contact with the ball|ball involved|ball part"),
             ("try_to_play", r"attempt|\btry|trying|intention|effort|make a play|intercept|reach the ball"),
             ("upper_bodypart", r"upper body"),
             ("bodypart", r"body"),
             ("contact", r"contact|physical interaction"),
             ("severity", r"sever|serious|intensity|gravity|\bcard"),
             ("offence", r"attack|offensive")]
# replay tren dev dai 12.0 s (trung vi), toi da 21 s; nguoi dung thuong chi vao giua replay
REPLAY_HALF = 6.0          # doan replay = +-6 s quanh moc thoi gian
REPLAY_GUARD = 11.0        # ung vien phai ket thuc truoc moc thoi gian it nhat 11 s (nua replay dai nhat): tranh lay nham dau replay
CAPTION_PROMPT = ("This football clip most likely shows: {event}. Write one short sentence of live football commentary "
                  "for this clip. Use [PLAYER], [TEAM] and [REFEREE] instead of names.")
SELECT_PROMPT = ("Question: {question}\nThe answer found by watching the video is: \"{answer}\"\n"
                 "Which option says the same thing?\nOptions:\n{options}\nAnswer with the option key only.")


def dimension_from_question(question):
    from sva.branches.multiview import question_part
    q = question_part(question).lower()
    return next((d for d, p in DIM_RULES if re.search(p, q)), "action")


def with_options(sample, texts):
    keys = [f"O{i + 1}" for i in range(len(texts))]
    return replace(sample, option_keys=keys, option_texts=dict(zip(keys, texts)), gold_key=None)


def rouge_l(a, b):
    """ROUGE-L F1 tren tu (do giong chu giua cau binh luan sinh ra va cau that)."""
    x, y = a.lower().split(), b.lower().split()
    if not x or not y:
        return 0.0
    dp = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i in range(len(x)):
        for j in range(len(y)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if x[i] == y[j] else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    p, r = lcs / len(x), lcs / len(y)
    return 0.0 if lcs == 0 else 2 * p * r / (p + r)


class Grounding:
    def __init__(self, widen=0.5, captions=4):
        self.sva, self.widen, self.n_captions = None, widen, captions

    def attach(self, sva):
        self.sva = sva

    @property
    def scorer(self):
        return self.sva.scorer

    # ---------------------------------------------------------------- mo rong doan (hanh dong "expand")
    def can_expand(self, sample):
        """Chi doan ao (video that) moi mo rong duoc; clip benchmark la ca file nen khong."""
        return any(split_segment(p)[1] is not None for p in sample.materials)

    def expand(self, sample):
        """Moi doan rong them widen (moi ben widen/2), khong vuot qua dau / cuoi video; giu nguyen phuong an."""
        out = []
        for p in sample.materials:
            file, a, b = split_segment(p)
            if a is None:
                out.append(p)
                continue
            pad = (b - a) * self.widen / 2
            out.append(segment(file, max(0.0, a - pad), min(probe(file).duration, b + pad)))
        return replace(sample, materials=out)

    # ---------------------------------------------------------------- ung vien khi khong co phuong an
    def shortlist(self, sample, texts, k=4, prompt=None):
        """Cham ca danh muc trong 1 luot VLM (nhan chu cai), giu k nhan cao nhat."""
        full = with_options(sample, texts)
        labels = list(LETTERS[:len(texts)])
        ev = self.scorer.score(full, prompt=prompt or mcq_prompt(full, labels), labels=labels, tool="shortlist")
        order = np.argsort([-ev.dist[key] for key in full.option_keys])[:k]
        return [texts[i] for i in order]

    def camera_pairs(self, sample, k=4):
        """q8: 169 cap 'From X to Y.' -> 4 cap co P(dau = X) * P(cuoi = Y) cao nhat (kenh camera cua nhanh B)."""
        branch = self.sva.branches["B"]
        frames, durations, _ = self.scorer.clips(sample)
        f, d, e = frames[0], durations[0], branch.edge
        seg = d * e / len(f)
        start, end = branch.camera_dist(f[:e], seg), branch.camera_dist(f[-e:], seg)
        score = np.log(start + 1e-9)[:, None] + np.log(end + 1e-9)[None, :]
        best = np.dstack(np.unravel_index(np.argsort(-score.ravel())[:k], score.shape))[0]
        return [f"From {CAMERA_LABELS[x]} to {CAMERA_LABELS[y]}." for x, y in best]

    def captions(self, sample):
        """q11: sinh cau binh luan theo cac loai su kien UniSoccer thay nhieu nhat (sinh roi xep hang)."""
        pv = self.sva.branches["A"].video_events(sample.materials[0])
        frames, durations, _ = self.scorer.clips(sample)
        out = []
        for i in np.argsort(-pv)[:self.n_captions]:
            text = self.scorer.backend.generate(frames[0], CAPTION_PROMPT.format(event=CLASS_NAMES[i]), durations[0],
                                                max_new_tokens=48)
            out.append(text.strip().split("\n")[0])
        return list(dict.fromkeys(t for t in out if t)) or ["[PLAYER] ([TEAM]) is on the ball."]

    def candidates(self, sample):
        t = sample.task_id
        if t == "q8":
            return with_options(sample, self.camera_pairs(sample))
        if t == "q9":                                  # phuong an chi la vi tri clip: Open = trac nghiem
            return with_options(sample, [f"The {ORDINALS[i]} one." for i in range(len(sample.materials) - 1)])
        if t == "q10":
            return with_options(sample, CLASS_NAMES)
        if t == "q11":
            return with_options(sample, self.captions(sample))
        if t == "q13":
            from sva.branches.branch_b import HINTS
            full = with_options(sample, COLORS)
            labels = list(LETTERS[:len(COLORS)])
            return with_options(sample, self.shortlist(sample, COLORS, prompt=HINTS["q13"] + mcq_prompt(full, labels)))
        if t == "q14":
            return with_options(sample, DIM_LABELS[dimension_from_question(sample.question)])
        raise ValueError(f"khong co danh muc nhan cho {t}")

    def answer_text(self, cand, key):
        """Nhan tra loi; q9 tra ten file clip ung vien (de so voi dap an mo cua SoccerBench)."""
        if cand.task_id == "q9":
            return Path(split_segment(cand.materials[cand.option_keys.index(key) + 1])[0]).name
        return cand.option_texts[key]

    # ---------------------------------------------------------------- anh xa vao phuong an (kieu paper)
    def select_option(self, sample, answer):
        """Khop nguyen van thi lay luon; khong thi LLM chon phuong an noi cung y (Answer Selection cua paper)."""
        for k in sample.option_keys:
            if canon(sample.option_texts[k]) == canon(answer):
                return k, "exact"
        if sample.task_id == "q9":
            return None, "none"
        opts = "\n".join(f"{k}. {sample.option_texts[k]}" for k in sample.option_keys)
        msg = [{"role": "user", "content": SELECT_PROMPT.format(question=sample.question, answer=answer, options=opts)}]
        p = softmax(self.scorer.backend.score_chat(msg, sample.option_keys))
        return sample.option_keys[int(p.argmax())], "llm"

    def solve_open(self, sample):
        """Mot cau SoccerBench o che do Open: an phuong an khi suy luan, chi dung chung de cham kieu paper."""
        cand = self.candidates(sample)
        out = self.sva(cand)
        answer = self.answer_text(cand, out["pred"])
        if sample.task_id == "q9":                     # vi tri ung vien trung voi khoa phuong an goc
            key, how = sample.option_keys[cand.option_keys.index(out["pred"])], "position"
        else:
            key, how = self.select_option(sample, answer)
        return {"pred": key, "select": how, "open_pred": answer, "candidates": cand.option_texts,
                "open_dist": out["dist"], "confidence": out["confidence"], "evidence": out["evidence"],
                "steps": out["steps"], "branch": out["branch"]}

    # ---------------------------------------------------------------- video that -> Sample
    def view_name(self, path):
        """Ten camera cua mot goc quay (VLM cham 13 nhan tren vai frame) de viet cau hoi q14 theo khuon."""
        branch = self.sva.branches["B"]
        frames, durations, _ = self.scorer.cache.get_clips([path], 4, 24, self.scorer.size)
        return CAMERA_LABELS[int(np.argmax(branch.camera_dist(frames[0], durations[0])))]

    def from_video(self, intent, videos, question, options, vmap=None):
        """(Sample, None) hoac (None, cau hoi lai nguoi dung). videos: danh sach file; vmap: ban do cua videos[0]."""
        from sva.branches.multiview import question_part
        from sva.schema import Sample
        t, time = intent.task, intent.time
        texts = list(options.values()) if options else ["?"]
        if t == "q9" and len(videos) == 5:
            mats = list(videos)
        elif t == "q9":
            if not time:                       # tu tim replay khong dang tin (3/25 video ghep) -> hoi lai
                return None, "Please give the time of the replay (e.g. 'the replay at 2:40')."
            at = (time[0] + time[1]) / 2
            cands = [s for s in vmap["shots"] if s["end"] <= at - REPLAY_GUARD and at - s["start"] <= 120
                     and s["end"] - s["start"] >= 1.5][-12:]
            if not cands:
                return None, "No earlier play found before the replay. Please give the replay time (e.g. 'at 2:40')."
            replay = segment(videos[0], max(0.0, at - REPLAY_HALF), min(vmap["duration"], at + REPLAY_HALF))
            mats = [replay] + [segment(videos[0], s["start"], s["end"]) for s in cands]
            texts = [f"The {ORDINALS[i]} one." for i in range(len(cands))]
        elif t == "q14":
            if len(videos) >= 2:
                mats = list(videos[:4])
            else:
                if not time and vmap["duration"] > 40:
                    return None, "The video is long. Please give the time of the foul (e.g. 'at 1:10')."
                center = time[0] if time else vmap["duration"] / 2
                near = [s for s in vmap["shots"] if s["end"] > center - 5 and s["start"] < center + 25][:4]
                mats = [segment(videos[0], s["start"], s["end"]) for s in near] or [videos[0]]
            views = ", ".join(f"{self.view_name(p)} lens at 1.0x speed" for p in mats)
            question = f"There are {len(mats)} video replays of the same foul, including: {views}. {question_part(question)}"
        else:
            duration = probe(videos[0]).duration
            half = 4.0 if t == "q8" else 15.0
            if time:
                a, b = time
                a, b = max(0.0, (a + b) / 2 - half), min(duration, (a + b) / 2 + half)
                mats = [segment(videos[0], a, b)]
            elif duration <= 40:
                mats = [videos[0]]
            else:
                return None, f"The video is {duration / 60:.1f} minutes long. Please say when (e.g. 'at 1:10')."
        keys = [f"O{i + 1}" for i in range(len(texts))]
        return Sample(t, 0, question, mats, keys, dict(zip(keys, texts)), None), None
