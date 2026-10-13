import hashlib
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np

from sva.constants import EMBEDDING_CACHE_DIR, FRAME_CACHE_DIR, SOCCERBENCH_ROOT
from sva.perception.video import letterbox, sample_frames, split_segment


def rel_key(path):
    # duong dan tuong doi trong SoccerBench, nen may khac (server) van ra cung key
    p = Path(path).resolve()
    try:
        return p.relative_to(SOCCERBENCH_ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def cache_key(path, n_frames, size, encoder=None):
    # key theo file vat ly vi nhieu mau dung chung 1 video
    s = f"{rel_key(path)}|{n_frames}|{size}|{encoder or ''}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def legacy_key(path, n_frames, size, encoder=None):
    # key cu (truoc 08/10) theo duong dan tuyet doi, van doc lai duoc
    s = f"{Path(path).resolve()}|{n_frames}|{size}|{encoder or ''}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def adopt_legacy(old, new):
    # chep file cache key cu sang ten key moi (ghi tam roi doi ten)
    if old.exists() and not new.exists():
        tmp = new.with_name(f"{new.stem}.{os.getpid()}.tmp{new.suffix}")
        shutil.copyfile(old, tmp)
        os.replace(tmp, new)


class PerceptionCache:
    def __init__(self, frame_dir=FRAME_CACHE_DIR, embedding_dir=EMBEDDING_CACHE_DIR, memory=256):
        self.frame_dir = Path(frame_dir)
        self.embedding_dir = Path(embedding_dir)
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        # SVA_FRAME_CACHE=0 (server, quota o dia nho): frame chi giu trong RAM, khong ghi ra dia
        self.to_disk = os.environ.get("SVA_FRAME_CACHE", "1") != "0"
        self.memory, self.max_memory = OrderedDict(), memory

    def get_frames(self, path, n=8, size=336):
        if not self.to_disk:
            key = cache_key(path, n, size)
            if key in self.memory:
                self.hits += 1
                self.memory.move_to_end(key)
                return self.memory[key]
            self.misses += 1
            frames, timestamps = sample_frames(path, n)
            self.memory[key] = np.stack([letterbox(x, size) for x in frames]), timestamps
            if len(self.memory) > self.max_memory:
                self.memory.popitem(last=False)
            return self.memory[key]

        f = self.frame_dir / f"{cache_key(path, n, size)}.npz"
        adopt_legacy(self.frame_dir / f"{legacy_key(path, n, size)}.npz", f)
        if f.exists():
            self.hits += 1
            data = np.load(f)
            return data["frames"], data["timestamps"].tolist()

        self.misses += 1
        frames, timestamps = sample_frames(path, n)
        frames = np.stack([letterbox(x, size) for x in frames])
        # ghi ra file tam roi doi ten, de tien trinh khac khong doc phai file dang ghi do
        tmp = f.with_name(f"{f.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, frames=frames, timestamps=np.array(timestamps))
        os.replace(tmp, f)
        return frames, timestamps

    def get_clips(self, paths, n_single=8, budget=24, size=336):
        """Frame cho ca mau: 1 clip lay n_single frame, nhieu clip chia deu budget (so chan, >= 2)."""
        n = n_single if len(paths) == 1 else max(2, budget // len(paths) // 2 * 2)
        frames, durations, timestamps = [], [], []
        for p in paths:
            f, ts = self.get_frames(p, n, size)
            frames.append(f)
            timestamps.append(ts)
            _, a, b = split_segment(p)
            durations.append(b - a if a is not None else 2 * n * ts[0])    # ts[0] = 0.5 / n * duration
        return frames, durations, timestamps

    def get_embedding(self, path, n, size, encoder):
        f = self.embedding_dir / f"{cache_key(path, n, size, encoder.name)}_{encoder.name}.npy"
        adopt_legacy(self.embedding_dir / f"{legacy_key(path, n, size, encoder.name)}_{encoder.name}.npy", f)
        if f.exists():
            self.hits += 1
            return np.load(f)

        self.misses += 1
        frames, _ = self.get_frames(path, n, size)
        emb = encoder.encode(frames).astype(np.float16)
        tmp = f.with_name(f"{f.stem}.{os.getpid()}.tmp.npy")
        np.save(tmp, emb)
        os.replace(tmp, f)
        return emb
