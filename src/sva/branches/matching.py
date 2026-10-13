"""Nhanh C, q9: tim clip goc cua replay bang so khop hinh anh, khong qua binh luan chu.

score(u) = trung binh theo frame replay i cua max_j cos(e_i^R, e_j^u)        (chamfer mem)
Omega(u) = ty le cap frame lien tiep i, i+1 ma vi tri khop tot nhat bi lui (j*(i+1) < j*(i))
score'(u) = score(u) - lam * Omega(u)        (replay quay cham van giu thu tu dien bien)
"""
import re

import cv2
import numpy as np

from sva.evidence.builder import build_positional

ORDINAL = re.compile(r"(\d+)(?:st|nd|rd|th)")


def strip_bars(frames, size=224):
    """Bo vien den cua anh letterbox roi dua ve size x size (giu toan bo noi dung, khong cat mep)."""
    lit = frames.max(axis=(0, 3)) > 8
    rows, cols = np.where(lit.any(axis=1))[0], np.where(lit.any(axis=0))[0]
    if len(rows) == 0:
        return np.stack([cv2.resize(f, (size, size)) for f in frames])
    crop = frames[:, rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in crop])


class FrameEncoder:
    """Encoder chi nap model khi cache thieu. square: bo vien den truoc (ten cache rieng, them _sq)."""

    def __init__(self, encoder_name, square=False, device="cuda"):
        self.base_name, self.square, self.device, self._enc = encoder_name, square, device, None
        self.name = f"{encoder_name}_sq" if square else encoder_name

    def encode(self, frames):
        if self._enc is None:
            from sva.perception.encoder import build_encoder
            self._enc = build_encoder(self.base_name, self.device)
        return self._enc.encode(strip_bars(frames) if self.square else frames)


def candidate_order(sample):
    """option_keys -> chi so ung vien (1..4) doc tu 'The 1st one.' ..."""
    out = []
    for k in sample.option_keys:
        m = ORDINAL.search(sample.option_texts[k])
        if not m:
            raise ValueError(f"khong doc duoc thu tu ung vien: {sample.option_texts[k]!r}")
        out.append(int(m.group(1)))
    return out


def chamfer(replay, cand, lam):
    """replay [N,d], cand [M,d] da chuan hoa -> (score', score, Omega, max cos)."""
    s = replay @ cand.T
    best = s.argmax(axis=1)
    omega = float(np.mean(best[1:] < best[:-1])) if len(best) > 1 else 0.0
    score = float(s.max(axis=1).mean())
    return score - lam * omega, score, omega, float(s.max())


class ReplayMatcher:
    name = "replay_match"

    def __init__(self, cache, encoder="clip_vit_l", n_frames=24, size=336, square=False, lam=0.0, temperature=0.01,
                 device="cuda"):
        self.cache, self.encoder = cache, FrameEncoder(encoder, square, device)
        self.n, self.size, self.lam, self.temperature = n_frames, size, lam, temperature

    def embed(self, path):
        return self.cache.get_embedding(path, self.n, self.size, self.encoder).astype(np.float32)

    def __call__(self, sample):
        replay = self.embed(sample.materials[0])
        rows = [chamfer(replay, self.embed(sample.materials[c]), self.lam) for c in candidate_order(sample)]
        obs = " | ".join(f"{k}: {r[1]:.3f} (lui {r[2]:.2f})" for k, r in zip(sample.option_keys, rows))
        ev = build_positional(self.name, sample, [r[0] for r in rows], raw_obs=obs, T=self.temperature)
        ev.material_ref["max_cos"] = [round(r[3], 4) for r in rows]
        return ev
