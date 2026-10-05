from dataclasses import dataclass

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


def timestamps_for(n, duration):
    return [(k + 0.5) / n * duration for k in range(n)]


def sample_frames(path, n=8):
    """Lay n frame theo timestamp (khong theo chi so frame vi fps cua q8, q9 khong co dinh).

    Tra ve (frames [n,H,W,3] RGB uint8, timestamps).
    """
    meta = probe(path)
    timestamps = timestamps_for(n, meta.duration)

    cap = cv2.VideoCapture(str(path))
    frames = []
    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:   # seek sat cuoi clip, lui lai 1 frame
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0, meta.duration * 1000 - 40))
            ok, frame = cap.read()
        if not ok:
            cap.release()
            raise OSError(f"khong doc duoc frame t={t:.2f}s: {path}")
        frames.append(unsqueeze(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return np.stack(frames), timestamps


def unsqueeze(img):
    # clip lay tu SoccerReplay-1988 bi luu 224x398 (hinh ngang bi bop lai), keo ve 398x224
    h, w = img.shape[:2]
    if h > w:
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
