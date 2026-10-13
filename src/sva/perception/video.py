import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


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
    """Lay n frame theo timestamp (khong theo chi so frame vi fps cua q8, q9 khong co dinh).

    start, end (giay): chi lay trong doan nay, mac dinh ca clip; path co the la doan ao 'video.mp4#t=a,b'.
    Tra ve (frames [n,H,W,3] RGB uint8, timestamps tinh tu dau file).
    """
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
