"""Question Agent: doc cau hoi (tieng Anh), so video va phuong an -> loai cau hoi (1 trong 6 task hoac ngoai pham vi)
va moc thoi gian. Luat truoc (so video, tu khoa, do dai phuong an); luat khong khop thi LLM cham diem 7 nhan."""
import re
from dataclasses import dataclass, field

import numpy as np

from sva.models.scoring import margin_confidence, softmax

LETTERS = "ABCDEFG"
TASKS = [("q8", "how the camera position or camera angle changes during the video"),
         ("q9", "which earlier clip or moment a replay (a repeated, often slow-motion clip) shows"),
         ("q10", "what type of football event happens (corner, goal, foul, substitution, throw-in, ...)"),
         ("q11", "a live commentary sentence that describes the video"),
         ("q13", "the jersey color of the team or player who does an action"),
         ("q14", "a property of a foul (offence, card, contact, body part, handball, ...), often seen from several angles"),
         ("out", "anything else (statistics, player names, scores, history, rules, ...) that the video cannot answer")]
# thu tu quan trong (chon tren dev): cau mau ao co the nhac "foul", "card"; cau q14 nhac ten camera
# ("Main camera center lens") va "video replays of the same foul" nen xet truoc camera va replay
RULES = [("q13", r"jersey|colou?r"),
         ("q14", r"\bfouls?\b|offen[cs]e|\bcards?\b|handball|tackl"),
         ("q8", r"camera"),
         ("q9", r"replay|slow[- ]?motion|played (again|back)|shown again"),
         ("q11", r"comment"),
         ("q10", r"event|categor|label|classif|type of|kind of|what happens|what happened")]
LONG_OPTIONS = 40          # phuong an trung binh dai hon -> cau binh luan (q11); chon tren dev: q10 <= 18, q11 >= 47
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
    if options and np.mean([len(t) for t in options.values()]) > LONG_OPTIONS:
        return "q11", "options"
    return None, None


class QuestionAgent:
    def __init__(self, backend=None, use_llm=True):
        self.backend, self.use_llm = backend, use_llm

    def llm_dist(self, question, options=None):
        opts = "".join(f"Option: {t}\n" for t in (options or {}).values())
        msg = [{"role": "user", "content": ROUTE_PROMPT.format(question=question, options=opts)}]
        p = softmax(self.backend.score_chat(msg, list(LETTERS)))
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
