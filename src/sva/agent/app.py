"""SVAAgent: diem vao chung cho benchmark (Sample SoccerBench) va ung dung (video + cau hoi that, tieng Anh).

Benchmark (__call__): Question Agent doan task tu cau hoi (an task_id that) -> loi SVA nhu cu.
  open_mode: an ca phuong an -> Grounding tu tao ung vien -> nhan tra loi -> anh xa vao phuong an (kieu paper).
Ung dung (ask): Question Agent -> Video Agent -> Grounding -> loi SVA (+ Controller) -> Answer Agent.
"""
from dataclasses import replace
from pathlib import Path

import torch

from sva.agent.answer import compose, save_frame, span
from sva.agent.question import QuestionAgent
from sva.config import load_config
from sva.pipeline import SVA
from sva.perception.video import probe, split_segment


class SVAAgent:
    def __init__(self, backend="qwen_local_3b", theta=None, name="sva_agent", policy=None, open_mode=False):
        from sva.agent.grounding import Grounding
        from sva.agent.video import VideoAgent
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
        Tra ve dict: status (answered / abstained / clarify / out_of_scope), answer, task, label, key, confidence, ..."""
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
        need_map = intent.task == "q9" and len(videos) == 1 or intent.task == "q14" and len(videos) == 1
        vmap = self.video.build(videos[0], events=False) if need_map else None
        sample, ask_back = self.grounding.from_video(intent, videos, question, options, vmap)
        if sample is None:
            return {**base, "status": "clarify", "answer": ask_back}
        cand = sample if options else self.grounding.candidates(sample)
        out = self.sva(cand)
        key = out["pred"]
        label = (span(cand.materials[cand.option_keys.index(key) + 1]) if cand.task_id == "q9" and
                 split_segment(cand.materials[1])[1] is not None else self.grounding.answer_text(cand, key))
        ranked = sorted(out["dist"].items(), key=lambda kv: -kv[1])
        text, status = compose(cand, label, out["dist"][key], [(cand.option_texts[k], p) for k, p in ranked],
                               self.cfg["abstain_below"])
        frames = []
        if out_dir:
            frames.append(save_frame(cand.materials[0], out_dir, "segment"))
            if cand.task_id == "q9":
                frames.append(save_frame(cand.materials[cand.option_keys.index(key) + 1], out_dir, "matched_play"))
        chosen = cand.materials[cand.option_keys.index(key) + 1] if cand.task_id == "q9" else cand.materials[0]
        return {**base, "status": status, "answer": text, "label": label, "key": key if options else None, "chosen": chosen,
                "confidence": out["dist"][key], "segments": cand.materials, "candidates": cand.option_texts,
                "dist": out["dist"], "evidence": out["evidence"], "steps": out["steps"], "frames": frames}
