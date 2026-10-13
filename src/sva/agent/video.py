"""Video Agent: doc mot video that (dai, fps / kich thuoc bat ky) thanh ban do, tinh mot lan roi luu cache.

Ban do gom:
  shots   cac canh (cat canh bang PySceneDetect)
  events  phan phoi 23 su kien UniSoccer tren cua so 30 giay truot 10 giay

Da thu tu dong tim replay bang do chuyen dong (replay quay cham thi it chuyen dong): chi dung 3/25 video ghep
(canh can canh, khan dai cung it chuyen dong) -> bo; khong co moc thoi gian cua replay thi agent hoi lai nguoi dung.
"""
import hashlib
import json
import os

import numpy as np

from sva.constants import CACHE_DIR
from sva.perception.video import probe, segment, split_segment


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

