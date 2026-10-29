"""SVA-Agent: Question / Video / Grounding / Answer agent quanh SVA, tra loi cau hoi tren video that."""
import hashlib
import json
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
import torch

from sva.branches import HINTS, question_part
from sva.config import CACHE_DIR, CAMERA_LABELS, load_config
from sva.data import Sample, canon
from sva.models import CLASS_NAMES, LETTERS, margin_confidence, mcq_prompt, softmax
from sva.perception import probe, sample_frames, segment, split_segment
from sva.pipeline import SVA


# ---------------------------------------------------------------- Question Agent
TASKS = [("q8", "how the camera position or camera angle changes during the video"),
         ("q9", "which earlier clip or moment a replay (a repeated, often slow-motion clip) shows"),
         ("q10", "what type of football event happens (corner, goal, foul, substitution, throw-in, ...)"),
         ("q11", "a live commentary sentence that describes the video"),
         ("q13", "the jersey color of the team or player who does an action"),
         ("q14", "a property of a foul (offence, card, contact, body part, handball, ...), often seen from several angles"),
         ("out", "anything else (statistics, player names, scores, history, rules, ...) that the video cannot answer")]
# thu tu chon tren dev: cau q14 nhac ten camera va "video replays" nen xet truoc q8, q9
RULES = [("q13", r"jersey|colou?r"),
         ("q14", r"\bfouls?\b|offen[cs]e|\bcards?\b|handball|tackl"),
         ("q8", r"camera"),
         ("q9", r"replay|slow[- ]?motion|played (again|back)|shown again"),
         ("q11", r"comment"),
         ("q10", r"event|categor|label|classif|type of|kind of|what happens|what happened")]
LONG_OPTIONS = 40          # phuong an dai hon -> cau binh luan q11 (dev: q10 <= 18, q11 >= 47 ky tu)
ROUTE_PROMPT = ("You route questions about a soccer video to the skill that can answer them.\n"
                "Question: {question}\n{options}Which skill is needed?\n"
                + "\n".join(f"{l}. {d}" for l, (_, d) in zip(LETTERS, TASKS))
                + "\nAnswer with the letter only.")
CLOCK = re.compile(r"(\d{1,2}):(\d{2})")


@dataclass
class Intent:
    task: str                         # q8 ... q14 hoac "out"
    source: str                       # "clips", "rule", "options", "llm"
    dist: dict = field(default_factory=dict)      # phan phoi LLM tren 7 nhan (neu co goi LLM)
    confidence: float = 1.0
    time: tuple = None                # (bat dau, ket thuc) giay neu cau hoi nhac moc thoi gian
    open: bool = False                # khong co phuong an -> tu tao ung vien


def parse_time(question):
    """'at 1:10' -> (70, 70); 'between 0:15 and 0:22' -> (15, 22); 'minute 23' -> (1380, 1380); 'at 40 seconds' -> (40, 40)."""
    q = question.lower()
    clocks = [int(m) * 60 + int(s) for m, s in CLOCK.findall(q)]
    if clocks:
        return (min(clocks), max(clocks))
    m = re.search(r"minute (\d+)", q)
    if m:
        return (int(m.group(1)) * 60,) * 2
    m = re.search(r"(\d+(?:\.\d+)?) ?(?:seconds?|secs?|s)\b", q)
    if m:
        return (float(m.group(1)),) * 2
    return None


def vocab_task(options):
    """Moi phuong an deu thuoc bo nhan cua mot task -> task do (q14 xet truoc q10 vi ca hai co 'Yellow card')."""
    texts = [canon(t) for t in options.values()]
    if all(t.startswith("from ") and " to " in t for t in texts):
        return "q8"
    for task, vocab in (("q14", [l for v in DIM_LABELS.values() for l in v]), ("q13", COLORS), ("q10", CLASS_NAMES)):
        if all(t in {canon(v) for v in vocab} for t in texts):
            return task
    return None


def rule_task(question, n_videos=1, options=None):
    """(task, nguon) theo luat; (None, None) neu khong luat nao khop."""
    if n_videos == 5:
        return "q9", "clips"                   # 1 replay + 4 ung vien
    if 2 <= n_videos <= 4:
        return "q14", "clips"                  # cung mot pha tu nhieu goc quay
    q = question.lower()
    for task, pattern in RULES:
        if re.search(pattern, q):
            return task, "rule"
    if options and vocab_task(options):
        return vocab_task(options), "options"
    if options and np.mean([len(t) for t in options.values()]) > LONG_OPTIONS:
        return "q11", "options"
    return None, None


class QuestionAgent:
    def __init__(self, backend=None, use_llm=True):
        self.backend, self.use_llm = backend, use_llm

    def llm_dist(self, question, options=None):
        opts = "".join(f"Option: {t}\n" for t in (options or {}).values())
        msg = [{"role": "user", "content": ROUTE_PROMPT.format(question=question, options=opts)}]
        p = softmax(self.backend.score_chat(msg, list(LETTERS[:len(TASKS)])))
        return {task: float(x) for (task, _), x in zip(TASKS, p)}

    def __call__(self, question, n_videos=1, options=None, check_scope=False):
        """check_scope: cau hoi tu do ngoai doi -> LLM duoc quyen bac luat khi chac chan la ngoai pham vi."""
        task, source = rule_task(question, n_videos, options)
        dist, conf = {}, 1.0
        if (task is None or check_scope) and self.use_llm and self.backend is not None:
            dist = self.llm_dist(question, options)
            conf = margin_confidence(dist)
            best = max(dist, key=dist.get)
            if task is None or (best == "out" and dist["out"] > 0.5):
                task, source = best, "llm"
        return Intent(task or "out", source or "none", dist, conf, parse_time(question), open=not options)


# ---------------------------------------------------------------- Video Agent
def detect_shots(path, min_len=1.0):
    """[(bat dau, ket thuc)] giay; khong thay canh nao thi ca video la mot canh. Canh qua ngan gop vao canh truoc."""
    from scenedetect import ContentDetector, detect
    duration = probe(path).duration
    cuts = [(a.get_seconds(), b.get_seconds()) for a, b in detect(str(path), ContentDetector())] or [(0.0, duration)]
    shots = []
    for a, b in cuts:
        if shots and b - a < min_len:
            shots[-1] = (shots[-1][0], b)
        else:
            shots.append((a, b))
    return shots


class VideoAgent:
    def __init__(self, classifier=None, window=30.0, stride=10.0, cache_dir=CACHE_DIR / "agent" / "maps"):
        self.classifier, self.window, self.stride = classifier, window, stride
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def windows(self, duration):
        if duration <= self.window:
            return [(0.0, duration)]
        starts = np.arange(0.0, duration - self.window + 1e-6, self.stride).tolist()
        if starts[-1] + self.window < duration:
            starts.append(duration - self.window)
        return [(a, a + self.window) for a in starts]

    def build(self, path, events=True):
        """events=False: chi cat canh (du cho q9, q14), khong chay UniSoccer tren tung cua so."""
        path = split_segment(path)[0]
        key = hashlib.sha1(f"{os.path.abspath(path)}|{os.path.getsize(path)}|{events}".encode()).hexdigest()
        f = self.cache_dir / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        meta = probe(path)
        shots = detect_shots(path)
        out = {"path": str(path), "duration": meta.duration, "fps": meta.fps, "size": [meta.width, meta.height],
               "shots": [{"start": a, "end": b} for a, b in shots], "events": []}
        if events and self.classifier is not None:
            for a, b in self.windows(meta.duration):
                p = self.classifier.probs(segment(path, a, b))
                out["events"].append({"start": a, "end": b, "top": dict(list(p.items())[:3])})
        tmp = f.with_name(f"{f.stem}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(out), encoding="utf-8")
        os.replace(tmp, f)
        return out


# ---------------------------------------------------------------- Grounding Agent
ORDINALS = ["1st", "2nd", "3rd"] + [f"{i}th" for i in range(4, 21)]
# mau ao: danh muc lay tu cac phuong an q13 tren dev (khong dung test)
COLORS = ["blue", "white", "red", "yellow", "black", "green", "pink", "orange", "purple", "gray", "brown", "maroon",
          "navy", "light blue", "dark blue", "violet", "cyan", "teal", "turquoise", "gold", "silver", "lime", "magenta",
          "olive", "coral", "lavender"]
# q14: nhan cua tung chieu loi (theo SoccerNet-MVFoul)
DIM_LABELS = {"severity": ["No card", "No card or yellow card", "Yellow card", "Yellow card or red card", "Red card"],
              "offence": ["Offence", "Between", "No offence"],
              "contact": ["With contact", "Without contact"],
              "bodypart": ["Upper body", "Under body"],
              "upper_bodypart": ["Use of arms", "Use of shoulder"],
              "action": ["Standing tackling", "Tackling", "Challenge", "Holding", "Elbowing", "High leg", "Pushing", "Dive"],
              "handball": ["No", "Yes"], "multiple": ["No", "Yes"], "try_to_play": ["No", "Yes"],
              "touch_ball": ["No", "Yes", "Maybe"]}
# chieu loi duoc hoi, doc tu cau hoi khi khong co phuong an (thu tu chon tren dev)
DIM_RULES = [("multiple", r"multiple|several|more than one|multi-foul|concurrent"),
             ("handball", r"handball|hand/arm|hand contact|handle the ball"),
             ("touch_ball", r"touch|ball contact|contact with the ball|ball involved|ball part"),
             ("try_to_play", r"attempt|\btry|trying|intention|effort|make a play|intercept|reach the ball"),
             ("upper_bodypart", r"upper body"),
             ("bodypart", r"body"),
             ("contact", r"contact|physical interaction"),
             ("severity", r"sever|serious|intensity|gravity|\bcard"),
             ("offence", r"attack|offensive")]
# replay tren dev dai 12 s (trung vi), toi da 21 s
REPLAY_HALF = 6.0          # doan replay = +-6 s quanh moc thoi gian nguoi dung chi
REPLAY_GUARD = 11.0        # ung vien phai ket thuc truoc moc it nhat 11 s, tranh lay nham phan dau cua replay
CAPTION_PROMPT = ("This football clip most likely shows: {event}. Write one short sentence of live football commentary "
                  "for this clip. Use [PLAYER], [TEAM] and [REFEREE] instead of names.")
SELECT_PROMPT = ("Question: {question}\nThe answer found by watching the video is: \"{answer}\"\n"
                 "Which option says the same thing?\nOptions:\n{options}\nAnswer with the option key only.")
CLASS_PROMPT = "Which football event category is closest to \"{option}\"?\n{classes}\nAnswer with the letter only."


def dimension_from_question(question):
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
            full = with_options(sample, COLORS)
            labels = list(LETTERS[:len(COLORS)])
            return with_options(sample, self.shortlist(sample, COLORS, prompt=HINTS["q13"] + mcq_prompt(full, labels)))
        if t == "q14":
            return with_options(sample, DIM_LABELS[dimension_from_question(sample.question)])
        raise ValueError(f"khong co danh muc nhan cho {t}")

    def to_classes(self, texts):
        """q10 voi phuong an tu do ('corner kick'): moi phuong an -> lop UniSoccer (khop nguyen van, khong thi LLM chon)."""
        names = [canon(c) for c in CLASS_NAMES]
        listing = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(CLASS_NAMES))
        out = []
        for t in texts:
            if canon(t) in names:
                out.append(CLASS_NAMES[names.index(canon(t))])
                continue
            msg = [{"role": "user", "content": CLASS_PROMPT.format(option=t, classes=listing)}]
            p = softmax(self.scorer.backend.score_chat(msg, list(LETTERS[:len(CLASS_NAMES)])))
            out.append(CLASS_NAMES[int(p.argmax())])
        return out

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
        """(Sample, None) hoac (None, cau hoi lai nguoi dung). vmap: ban do canh cua videos[0]."""
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
            if t == "q10" and options:                     # nhanh A chi hieu 23 ten lop
                texts = self.to_classes(texts)
        keys = [f"O{i + 1}" for i in range(len(texts))]
        return Sample(t, 0, question, mats, keys, dict(zip(keys, texts)), None), None


# ---------------------------------------------------------------- Answer Agent
def clock(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def span(path):
    file, a, b = split_segment(path)
    return f"{clock(a)}-{clock(b)}" if a is not None else "the whole video"


def save_frame(path, out_dir, name):
    """Frame giua doan -> PNG; tra ve duong dan file."""
    frames, _ = sample_frames(path, 1)
    out = Path(out_dir) / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
    return str(out)


def compose(sample, label, prob, dist_labels, abstain_below=0.0):
    """(cau tra loi, trang thai). dist_labels: [(nhan, xac suat)] giam dan."""
    t, where = sample.task_id, span(sample.materials[0])
    if prob < abstain_below:
        (a, pa), (b, pb) = dist_labels[0], dist_labels[1] if len(dist_labels) > 1 else ("-", 0.0)
        return (f"Not sure. Most likely: {a} ({pa:.0%}), then {b} ({pb:.0%}). Please check {where}.", "abstained")
    conf = f"(confidence {prob:.2f})"
    if t == "q9":
        return f"The replay at {where} shows the play at {label} {conf}.", "answered"
    if t == "q8":
        return f"At {where} the camera changes {label[0].lower() + label[1:].rstrip('.')} {conf}.", "answered"
    if t == "q10":
        return f"At {where}: {label} {conf}.", "answered"
    if t == "q11":
        return f"Commentary for {where}: \"{label}\" {conf}", "answered"
    if t == "q13":
        return f"The player wears a {label} jersey {conf}.", "answered"
    return f"{label} {conf}.", "answered"


# ---------------------------------------------------------------- SVAAgent
class SVAAgent:
    def __init__(self, backend="qwen_local_3b", theta=None, name="sva_agent", policy=None, open_mode=False):
        self.name, self.open_mode = name, open_mode
        self.cfg = load_config()["agent"]
        self.grounding = Grounding()
        self.sva = SVA(backend, theta, name, policy, grounding=self.grounding)
        self.grounding.attach(self.sva)
        self.question = QuestionAgent(self.sva.scorer.backend)
        self.video = VideoAgent(self.sva.branches["A"].classifier)
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 6e9:
            # GPU nho (4 GB): CLIP cho doan video moi chay tren CPU, khong chen voi VLM tren GPU
            self.sva.branches["C"].matcher.encoder.device = "cpu"

    def supports(self, task_id):
        return True

    # ---------------------------------------------------------------- benchmark
    def __call__(self, sample):
        """Mot cau SoccerBench: task_id that bi an, Question Agent tu doan."""
        options = None if self.open_mode else sample.option_texts
        intent = self.question(sample.question, len(sample.materials), options)
        if intent.task == "out":
            return {"pred": None, "task_pred": "out", "route": intent.source, "error": "ngoai pham vi"}
        s = replace(sample, task_id=intent.task)
        info = {"task_pred": intent.task, "route": intent.source}
        if not self.open_mode:
            return {**info, **self.sva(s)}
        return {**info, **self.grounding.solve_open(s)}

    # ---------------------------------------------------------------- ung dung
    def ask(self, videos, question, options=None, out_dir=None):
        """videos: 1 file (co the dai) hoac 2-5 file; options: None (Open) hoac {"O1": ..., ...} (2-4 phuong an).
        Tra ve dict: status (answered / abstained / clarify / out_of_scope / error), answer, task, label,
        key (khoa phuong an goc cua nguoi dung), confidence, ..."""
        videos = [str(v) for v in videos]
        intent = self.question(question, len(videos), options, check_scope=True)
        base = {"task": intent.task, "route": intent.source, "time": intent.time}
        if intent.task == "out":
            return {**base, "status": "out_of_scope",
                    "answer": "This question is outside what can be answered from the video "
                              "(camera changes, replays, events, commentary, jersey colors, fouls)."}
        if intent.source == "llm" and intent.confidence < self.cfg["route_margin"]:
            return {**base, "status": "clarify", "answer": "Is the question about the camera, a replay, the event, "
                    "the commentary, a jersey color or a foul?"}
        try:
            return {**base, **self.answer(intent, videos, question, options, out_dir)}
        except Exception as e:                         # khong de loi tho lot ra ngoai nguoi dung
            return {**base, "status": "error", "answer": f"Could not answer this question ({type(e).__name__}: {e})."}

    def answer(self, intent, videos, question, options, out_dir):
        need_map = intent.task == "q9" and len(videos) == 1 or intent.task == "q14" and len(videos) == 1
        vmap = self.video.build(videos[0], events=False) if need_map else None
        sample, ask_back = self.grounding.from_video(intent, videos, question, options, vmap)
        if sample is None:
            return {"status": "clarify", "answer": ask_back}
        cand = sample if options else self.grounding.candidates(sample)
        out = self.sva(cand)
        key = out["pred"]
        i = cand.option_keys.index(key)
        if cand.task_id == "q9" and split_segment(cand.materials[1])[1] is not None:
            label = span(cand.materials[i + 1])
        elif options and cand.task_id != "q9":         # phuong an goc nguoi dung go (q10 co the da anh xa ve ten lop)
            label = list(options.values())[i]
        else:
            label = self.grounding.answer_text(cand, key)
        if cand.task_id == "q9" and split_segment(cand.materials[1])[1] is not None:   # ung vien la doan cua video dai
            shown = {k: span(m) for k, m in zip(cand.option_keys, cand.materials[1:])}
        elif options and cand.task_id != "q9":
            shown = dict(zip(cand.option_keys, options.values()))
        else:
            shown = cand.option_texts
        ranked = sorted(out["dist"].items(), key=lambda kv: -kv[1])
        text, status = compose(cand, label, out["dist"][key], [(shown[k], p) for k, p in ranked],
                               self.cfg["abstain_below"])
        frames = []
        if out_dir:
            frames.append(save_frame(cand.materials[0], out_dir, "segment"))
            if cand.task_id == "q9":
                frames.append(save_frame(cand.materials[i + 1], out_dir, "matched_play"))
        chosen = cand.materials[i + 1] if cand.task_id == "q9" else cand.materials[0]
        return {"status": status, "answer": text, "label": label, "key": list(options)[i] if options else None,
                "chosen": chosen, "confidence": out["dist"][key], "segments": cand.materials,
                "candidates": cand.option_texts, "dist": out["dist"], "evidence": out["evidence"], "steps": out["steps"],
                "frames": frames}
