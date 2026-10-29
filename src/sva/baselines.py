"""Baseline hoi thang: B0-local (Qwen tai cho, cham phan phoi) va B0-api (Gemini, dem tan suat)."""
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import cv2
from dotenv import load_dotenv

from sva.config import PROJECT_ROOT, load_config
from sva.evidence import Evidence, build_categorical, rel_path
from sva.models import build_backend, dist_from_sampling, margin_confidence, mcq_prompt, parse_answer
from sva.perception import PerceptionCache

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- B0-local: VLM hoi thang
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


# ---------------------------------------------------------------- B0-api: Gemini hoi thang
def next_reset(now=None):
    """Quota ngay cua Gemini hoi luc nua dem gio Pacific; tra ve thoi diem do (+5 phut cho chac)."""
    la = ZoneInfo("America/Los_Angeles")
    now = (now or datetime.now(la)).astimezone(la)
    return (now + timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)


def jpeg(frame):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


class B0Api:
    """Hoi thang MLLM thuong mai (Gemini). Khong co logprob nen hoi n lan roi dem tan suat."""

    name = "b0_api"

    def __init__(self, model="gemini", client=None, cache=None):
        load_dotenv(PROJECT_ROOT / ".env")
        cfg = load_config("models")["vlm"][model]
        perc = load_config()["perception"]
        self.model_id = cfg["model_id"]
        self.n_samples = cfg["sampling_n"]
        self.temperature = cfg["temperature"]
        self.thinking_level = cfg.get("thinking_level")
        self.max_output_tokens = cfg.get("max_output_tokens", 512)
        self.multi_candidates = True      # tat neu API khong nhan candidate_count
        self.min_interval = 60 / cfg.get("rpm", 15)   # goi cach nhau de khong vuot quota theo phut
        self._last_call = 0.0
        self._lock = threading.Lock()
        self.n_single, self.budget, self.size = perc["n_frames_default"], perc["n_frames_total_multiview"], perc["image_size"]
        self.cache = cache or PerceptionCache()
        if client is None:
            from google import genai
            keys = [k for k in re.split(r"[,\s]+", os.environ.get("GOOGLE_API_KEYS", "")) if k]
            keys = keys or [k for k in [os.environ.get(cfg["api_key_env"])] if k]
            if not keys:
                raise RuntimeError(f"chua co {cfg['api_key_env']} trong file .env")
            self.clients = [genai.Client(api_key=k) for k in keys]
            self.key_names = ["..." + k[-4:] for k in keys]
            self.wait_reset = True        # het ca cac key thi ngu den luc quota hoi lai
        else:
            self.clients = client if isinstance(client, list) else [client]
            self.key_names = [f"#{i}" for i in range(len(self.clients))]
            self.wait_reset = False
        self.cur = 0
        self.exhausted = set()
        self.dead = set()                 # key bi khoa (403), bo luon
        self.reset_at = None

    def _rotate(self, msg, dead=False):
        """Key hien tai het quota ngay (hoac bi khoa): chuyen key khac; het ca thi ngu den luc reset."""
        with self._lock:
            (self.dead if dead else self.exhausted).add(self.cur)
            if dead:
                log.warning("key %s bi khoa: %s", self.key_names[self.cur], msg[:120])
            if not dead and self.reset_at is None:
                self.reset_at = next_reset()
            if self.reset_at and datetime.now(self.reset_at.tzinfo) >= self.reset_at:
                self.exhausted.clear()    # qua gio reset: cac key het quota dung lai duoc
                self.reset_at = None
            free = [i for i in range(len(self.clients)) if i not in self.exhausted | self.dead]
            if free:
                if not dead:
                    log.warning("key %s het quota ngay, chuyen sang key %s", self.key_names[self.cur],
                                self.key_names[free[0]])
                self.cur = free[0]
                return
            if not self.wait_reset or len(self.dead) == len(self.clients):
                raise RuntimeError(msg)
            until = self.reset_at
            log.warning("ca %d key het quota ngay, cho den %s (gio Pacific) roi chay tiep",
                        len(self.clients) - len(self.dead), until)
            time.sleep(max(0.0, (until - datetime.now(until.tzinfo)).total_seconds()))
            self.exhausted.clear()
            self.reset_at = None
            self.cur = min(set(range(len(self.clients))) - self.dead)

    def _contents(self, sample):
        from google.genai import types
        frames, _, _ = self.cache.get_clips(sample.materials, self.n_single, self.budget, self.size)
        parts = []
        for i, clip in enumerate(frames):
            if len(frames) > 1:
                parts.append(f"Clip {i + 1}:")
            parts += [types.Part.from_bytes(data=jpeg(f), mime_type="image/jpeg") for f in clip]
        parts.append(mcq_prompt(sample))
        return parts

    def _call(self, contents, n):
        from google.genai import types
        thinking = types.ThinkingConfig(thinking_level=self.thinking_level) if self.thinking_level else None
        config = types.GenerateContentConfig(temperature=self.temperature, candidate_count=n,
                                             max_output_tokens=self.max_output_tokens, thinking_config=thinking)
        with self._lock:
            wait = self._last_call + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
        resp = self.clients[self.cur].models.generate_content(model=self.model_id, contents=contents, config=config)
        # lay phan tra loi, bo phan "thought" neu model tra kem
        return ["".join(p.text or "" for p in c.content.parts if not getattr(p, "thought", False))
                if c.content and c.content.parts else "" for c in resp.candidates]

    def ask(self, sample, retries=6):
        contents = self._contents(sample)
        answers = []
        attempt = 0
        while True:
            try:
                if self.multi_candidates:
                    try:
                        return self._call(contents, self.n_samples)
                    except Exception as e:
                        if "candidate" not in str(e).lower():
                            raise
                        self.multi_candidates = False
                while len(answers) < self.n_samples:      # giu lai cac cau da co khi phai thu lai
                    answers += self._call(contents, 1)
                return answers
            except Exception as e:
                msg = str(e)
                dead = "PERMISSION_DENIED" in msg or "UNAUTHENTICATED" in msg    # key/tai khoan bi khoa
                if "PerDay" in msg or dead:
                    self._rotate(msg, dead=dead)   # khong tinh vao so lan thu
                    continue
                attempt += 1
                if attempt >= retries:
                    raise
                if "503" in msg or "UNAVAILABLE" in msg:
                    time.sleep(30 * (attempt + 1))         # qua tai: doi lau hon
                    continue
                m = re.search(r"retry in ([\d.]+)s", msg)
                if m:
                    time.sleep(float(m.group(1)) + 1)
                elif "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    time.sleep(60)                         # quota: doi qua 1 phut roi thu lai
                else:
                    time.sleep(5 * 2 ** attempt)

    def __call__(self, sample):
        try:
            answers = self.ask(sample)
        except Exception as e:
            msg = str(e)
            if any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "PERMISSION_DENIED", "UNAUTHENTICATED")):
                return {"skip": True, "reason": msg[:400]}   # het quota / qua tai: khong tinh la sai, chay lai sau
            ev = Evidence.failure(self.name, sample.option_keys, f"{type(e).__name__}: {e}")
            return {"pred": None, "dist": ev.dist, "confidence": 0.0, "evidence": [ev.to_json()], "error": ev.raw_observation}

        dist = dist_from_sampling(answers, sample.option_keys, sample.option_texts)
        ev = Evidence(self.name, "categorical", sample.option_keys, dist, margin_confidence(dist),
                      " | ".join(answers), {"clips": [rel_path(p) for p in sample.materials]})
        parsed = [parse_answer(a, sample.option_keys, sample.option_texts) for a in answers]
        return {"pred": ev.pred, "dist": dist, "confidence": ev.confidence, "evidence": [ev.to_json()],
                "answers": answers, "unparsed": sum(p is None for p in parsed)}
