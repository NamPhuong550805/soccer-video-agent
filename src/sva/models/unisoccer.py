import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from sva.config import load_config
from sva.constants import CACHE_DIR, PROJECT_ROOT
from sva.perception.cache import adopt_legacy, rel_key
from sva.perception.video import split_segment

# thu tu lop dung nhu dau ra cua checkpoint pretrained_classification
CLASS_NAMES = ["var", "end of half game", "clearance", "second yellow card", "injury", "ball possession",
               "throw in", "show added time", "shot off target", "start of half game", "substitution",
               "saved by goal-keeper", "red card", "lead to corner", "ball out of play", "off side", "goal",
               "penalty", "yellow card", "foul lead to penalty", "corner", "free kick", "foul with no card"]


def middle_indices(n_frames, vlen):
    # chia video thanh n doan bang nhau, lay frame giua moi doan (sample="middle" cua UniSoccer)
    n = min(n_frames, vlen)
    edges = np.linspace(0, vlen, n + 1).astype(int)
    idx = [(a + b - 1) // 2 for a, b in zip(edges[:-1], edges[1:])]
    return idx + [idx[-1]] * (n_frames - len(idx))


def read_by_index(path, indices):
    cap = cv2.VideoCapture(str(path))
    wanted, frames, i = set(indices), {}, min(indices)
    if i > 0:                                  # doan ao giua video dai: tua toi frame dau thay vi doc tu dau
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    while len(frames) < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    if not frames:
        raise OSError(f"khong doc duoc video: {Path(path).name}")
    last = frames[max(frames)]
    return [frames.get(k, last) for k in indices]


class _TupleOut(torch.nn.Module):
    # code goc viet cho transformers 4 (encoder layer tra ve tuple); ban 5 tra ve tensor
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, mask=None):
        return (self.layer(x, mask),)


class ActionClassifier:
    """Action Classifier cua SoccerAgent: MatchVision (UniSoccer), 30 frame, softmax tren 23 lop."""

    def __init__(self, device="cuda"):
        cfg = load_config("models")["unisoccer"]
        self.cfg = cfg
        self.device = device
        self.cache_dir = CACHE_DIR / "tools" / "unisoccer_cls"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def _load(self):
        from transformers import AutoImageProcessor, SiglipVisionModel
        from transformers.utils import logging as hf_logging

        # transformers 5 bo lop vision_model ben trong SiglipVisionModel, code goc van goi model.vision_model
        if not hasattr(SiglipVisionModel, "vision_model"):
            SiglipVisionModel.vision_model = property(lambda self: self)
        hf_logging.set_verbosity_error()
        code = PROJECT_ROOT / self.cfg["code_dir"] / "toolbox" / "unisoccer"
        sys.path.insert(0, str(code))
        from model.MatchVision_classifier import MatchVision_Classifier

        model = MatchVision_Classifier(model_name=self.cfg["siglip"])
        state = torch.load(PROJECT_ROOT / self.cfg["classification_ckpt"], map_location="cpu")["state_dict"]
        model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
        for blk in model.siglip_model.timesformer.resblocks:
            blk.encoder = _TupleOut(blk.encoder)
        # fp16: tren 40 clip trung top-1 voi fp32, xac suat lech < 0.002, nhanh hon ~10 lan
        self.model = model.half().to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(self.cfg["siglip"])

    @torch.no_grad()
    def probs(self, path):
        n = self.cfg["num_frames"]
        f = self.cache_dir / f"{hashlib.sha1(f'{rel_key(path)}|{n}'.encode()).hexdigest()}.json"
        adopt_legacy(self.cache_dir / f"{hashlib.sha1(f'{Path(path).resolve()}|{n}'.encode()).hexdigest()}.json", f)
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        if self.model is None:
            self._load()
        file, start, end = split_segment(path)
        cap = cv2.VideoCapture(str(file))
        vlen, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        first, last = (0, vlen) if start is None else (int(start * fps), min(vlen, int(end * fps)))
        frames = read_by_index(file, [first + i for i in middle_indices(self.cfg["num_frames"], last - first)])
        x = self.processor(images=frames, return_tensors="pt")["pixel_values"]   # [T,C,H,W]
        x = x.permute(1, 0, 2, 3).unsqueeze(0).to(self.device, torch.float16)    # [1,C,T,H,W]
        logits = self.model.get_logits(x).float()
        p = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
        out = dict(sorted(zip(CLASS_NAMES, p), key=lambda kv: -kv[1]))
        tmp = f.with_name(f"{f.stem}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(out), encoding="utf-8")
        os.replace(tmp, f)
        return out
