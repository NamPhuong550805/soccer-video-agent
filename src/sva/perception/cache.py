import hashlib
from pathlib import Path

import numpy as np

from sva.constants import EMBEDDING_CACHE_DIR, FRAME_CACHE_DIR
from sva.perception.video import letterbox, sample_frames


def cache_key(path, n_frames, size, encoder=None):
    # key theo file vat ly vi nhieu mau dung chung 1 video
    s = f"{Path(path).resolve()}|{n_frames}|{size}|{encoder or ''}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


class PerceptionCache:
    def __init__(self, frame_dir=FRAME_CACHE_DIR, embedding_dir=EMBEDDING_CACHE_DIR):
        self.frame_dir = Path(frame_dir)
        self.embedding_dir = Path(embedding_dir)
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def get_frames(self, path, n=8, size=336):
        f = self.frame_dir / f"{cache_key(path, n, size)}.npz"
        if f.exists():
            self.hits += 1
            data = np.load(f)
            return data["frames"], data["timestamps"].tolist()

        self.misses += 1
        frames, timestamps = sample_frames(path, n)
        frames = np.stack([letterbox(x, size) for x in frames])
        np.savez_compressed(f, frames=frames, timestamps=np.array(timestamps))
        return frames, timestamps

    def get_embedding(self, path, n, size, encoder):
        f = self.embedding_dir / f"{cache_key(path, n, size, encoder.name)}_{encoder.name}.npy"
        if f.exists():
            self.hits += 1
            return np.load(f)

        self.misses += 1
        frames, _ = self.get_frames(path, n, size)
        emb = encoder.encode(frames).astype(np.float16)
        np.save(f, emb)
        return emb
