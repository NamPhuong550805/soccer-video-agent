"""Tang tri giac: doc video, lay frame theo thoi gian, doan ao, cache frame / embedding, encoder anh."""
import hashlib
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from sva.config import EMBEDDING_CACHE_DIR, FRAME_CACHE_DIR, SOCCERBENCH_ROOT, load_config


# ---------------------------------------------------------------- Doc video, lay frame
@dataclass
class VideoMeta:
    path: str
    fps: float
    n_frames: int
    duration: float
    width: int
    height: int


def probe(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"khong mo duoc video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return VideoMeta(str(path), fps, n_frames, n_frames / fps, width, height)


SEGMENT = re.compile(r"^(.*)#t=([\d.]+),([\d.]+)$")
SQUEEZED = (398, 224)   # (cao, rong) cua clip SoccerReplay-1988 luu dang bop


def segment(path, start, end):
    """Doan ao: 'video.mp4#t=55.00,85.00' - dung nhu mot clip, khong cat file."""
    return f"{split_segment(path)[0]}#t={start:.2f},{end:.2f}"


def split_segment(path):
    """'video.mp4#t=55,85' -> ('video.mp4', 55.0, 85.0); duong dan thuong -> (path, None, None)."""
    m = SEGMENT.match(str(path))
    return (m.group(1), float(m.group(2)), float(m.group(3))) if m else (str(path), None, None)


def timestamps_for(n, duration):
    return [(k + 0.5) / n * duration for k in range(n)]


def sample_frames(path, n=8, start=0.0, end=None):
    """n frame rai deu theo thoi gian trong [start, end] (hoac doan ao) -> (frames [n,H,W,3] RGB, timestamps)."""
    path, seg_start, seg_end = split_segment(path)
    if seg_start is not None:
        start, end = seg_start, seg_end
    meta = probe(path)
    end = meta.duration if end is None else min(end, meta.duration)
    timestamps = [start + t for t in timestamps_for(n, end - start)]

    cap = cv2.VideoCapture(str(path))
    frames = []
    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        back = 0.04
        while not ok and back <= 1.0:   # seek sat cuoi clip (fps thay doi) co the hut, lui dan toi da 1s
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0, (t - back) * 1000))
            ok, frame = cap.read()
            back += 0.1
        if not ok:
            cap.release()
            raise OSError(f"khong doc duoc frame t={t:.2f}s: {Path(path).name}")
        frames.append(unsqueeze(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return np.stack(frames), timestamps


def unsqueeze(img):
    # clip SoccerReplay-1988 bi luu 224x398 (hinh ngang bi bop lai) -> keo ve 398x224;
    # video doc that (vd quay bang dien thoai) giu nguyen
    h, w = img.shape[:2]
    if (h, w) == SQUEEZED:
        return cv2.resize(img, (h, w), interpolation=cv2.INTER_LINEAR)
    return img


def letterbox(img, size=336):
    """Dua anh ve size x size, giu ti le, phan thieu dem den."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = round(h * scale), round(w * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    out = np.zeros((size, size, 3), dtype=img.dtype)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    out[top:top + new_h, left:left + new_w] = resized
    return out


# ---------------------------------------------------------------- Cache frame va embedding
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


# ---------------------------------------------------------------- Encoder anh
def l2_normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


class OpenCLIPEncoder:
    """CLIP va SigLIP (qua open_clip)."""

    def __init__(self, name, model_id, pretrained, device="cpu"):
        import open_clip
        self.name = name
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_id, pretrained=pretrained)
        self.model.eval().to(device)
        self.tokenizer = open_clip.get_tokenizer(model_id)

    @torch.no_grad()
    def encode(self, frames):
        batch = torch.stack([self.preprocess(Image.fromarray(f)) for f in frames]).to(self.device)
        return l2_normalize(self.model.encode_image(batch).float().cpu().numpy())

    @torch.no_grad()
    def encode_text(self, texts):
        return l2_normalize(self.model.encode_text(self.tokenizer(texts).to(self.device)).float().cpu().numpy())


class DINOv2Encoder:
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, name, repo, model_id, device="cpu"):
        self.name = name
        self.device = device
        self.model = torch.hub.load(repo, model_id).eval().to(device)

    @torch.no_grad()
    def encode(self, frames):
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = ((x - self.MEAN) / self.STD).to(self.device)
        return l2_normalize(self.model(x).float().cpu().numpy())


class HSVEncoder:
    """Moc khong hoc: histogram HSV 8x8x8 chuan hoa L2."""
    name = "hsv"

    def encode(self, frames):
        h = np.array([cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_RGB2HSV)], [0, 1, 2], None, [8, 8, 8],
                                   [0, 180, 0, 256, 0, 256]).ravel() for f in frames])
        return l2_normalize(h)


def build_encoder(name, device="cpu"):
    """name: clip_vit_l | siglip | dinov2_vitb14 (xem configs/models.yaml)"""
    encoders = load_config("models")["encoder"]
    if name not in encoders:
        raise ValueError(f"khong co encoder {name!r}, chon trong {list(encoders)}")
    cfg = encoders[name]
    if cfg["lib"] == "open_clip":
        return OpenCLIPEncoder(name, cfg["model_id"], cfg["pretrained"], device)
    return DINOv2Encoder(name, cfg["repo"], cfg["model_id"], device)
