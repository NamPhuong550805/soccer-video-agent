"""Model: cham diem (logprob -> phan phoi), Qwen2.5-VL chay tai cho, Action Classifier cua UniSoccer."""
import hashlib
import json
import os
import re
import string
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

from sva.config import CACHE_DIR, PROJECT_ROOT, load_config
from sva.data import canon
from sva.perception import rel_key, split_segment


# ---------------------------------------------------------------- Cham diem: logprob -> phan phoi
LETTERS = string.ascii_uppercase           # nhan A, B, C... khi danh lai phuong an / danh muc


def softmax(x, temperature=1.0):
    z = np.asarray(x, dtype=np.float64) / temperature
    z = np.exp(z - z.max())
    return z / z.sum()


def dist_from_logprobs(logprobs, keys, temperature=1.0):
    """Softmax co nhiet do tren dung khong gian key cua mau."""
    if len(logprobs) != len(keys):
        raise ValueError(f"{len(logprobs)} logprob nhung co {len(keys)} key")
    return dict(zip(keys, softmax(logprobs, temperature).tolist()))


def parse_answer(text, keys, option_texts):
    """Doi cau tra loi cua model ve key: nhan ca 'O2', 'O2. red' lan 'red'."""
    for k in re.findall(r"O\d+", text):
        if k in keys:
            return k
    t = canon(text)
    for k in keys:
        if canon(option_texts[k]) == t:
            return k
    return None


def dist_from_sampling(samples, keys, option_texts, alpha=1.0):
    """Dung cho backend API khong co logits: dem tan suat + Laplace smoothing."""
    counts = Counter(parse_answer(s, keys, option_texts) for s in samples)
    total = sum(counts[k] for k in keys) + alpha * len(keys)
    return {k: (counts[k] + alpha) / total for k in keys}


def margin_confidence(dist):
    vals = sorted(dist.values(), reverse=True)
    return vals[0] - (vals[1] if len(vals) > 1 else 0.0)


def entropy(dist):
    p = np.array(list(dist.values()))
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


# ---------------------------------------------------------------- Prompt trac nghiem, nap backend
def mcq_prompt(sample, labels=None):
    """labels mac dinh la key goc (O1, O3...); truyen ["A", "B"] neu muon danh lai bang chu cai."""
    labels = labels or sample.option_keys
    lines = [f"Question: {sample.question}", "Options:"]
    lines += [f"{label}. {sample.option_texts[k]}" for label, k in zip(labels, sample.option_keys)]
    lines.append("Answer with the option key only." if labels == sample.option_keys
                 else "Answer with the option letter only.")
    return "\n".join(lines)


@lru_cache(maxsize=None)
def build_backend(name=None):
    """Nap model 1 lan, cac lan sau dung lai."""
    cfg = load_config("models")
    name = name or cfg["default_vlm"]
    c = cfg["vlm"][name]
    if c["backend"] == "qwen_local":
        return QwenVLBackend(name, c["model_id"], quantization=c.get("quantization"),
                             max_new_tokens=c.get("max_new_tokens", 128))
    raise ValueError(f"chua ho tro backend {c['backend']!r}")


# ---------------------------------------------------------------- Qwen2.5-VL chay tai cho
class QwenVLBackend:
    def __init__(self, name, model_id, quantization=None, max_new_tokens=128, device_map="auto"):
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        from transformers.integrations import sdpa_attention
        # Windows khong co flash-attention: lap lai key/value truoc de sdpa dung kernel tiet kiem bo nho (-2 GB)
        sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False
        self.name = name
        quant = None
        if quantization == "4bit":
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.float16)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.float16 if quant else torch.bfloat16,
            device_map=device_map, quantization_config=quant)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.max_new_tokens = max_new_tokens

    def _inputs(self, frames, prompt, duration=None):
        """frames: 1 clip [T,H,W,3] hoac list nhieu clip; duration tuong ung (giay)."""
        clips = [frames] if isinstance(frames, np.ndarray) else list(frames)
        durations = duration if isinstance(duration, (list, tuple)) else [duration] * len(clips)

        content = []
        for i, (c, d) in enumerate(zip(clips, durations)):
            if len(clips) > 1:
                content.append({"type": "text", "text": f"Clip {i + 1}:"})
            content.append({"type": "video", "video": c, "duration": d})
        content.append({"type": "text", "text": prompt})
        return self._chat_inputs([{"role": "user", "content": content}])

    def _chat_inputs(self, messages):
        """messages kieu chat; content la str hoac list cac {"type": "text"/"image"/"video", ...}."""
        msgs, images, videos, meta = [], [], [], []
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            items = []
            for c in content:
                if c["type"] == "image":
                    images.append(c["image"])
                    items.append({"type": "image"})
                elif c["type"] == "video":
                    v, d = np.asarray(c["video"]), c.get("duration")
                    videos.append(v)
                    # fps cua chuoi frame da lay, de model biet khoang cach thoi gian giua cac frame
                    meta.append({"total_num_frames": len(v), "fps": len(v) / d if d else 2.0,
                                 "frames_indices": list(range(len(v)))})
                    items.append({"type": "video"})
                else:
                    items.append({"type": "text", "text": c["text"]})
            msgs.append({"role": m["role"], "content": items})
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        kw = {}
        if images:
            kw["images"] = images
        if videos:
            kw.update(videos=videos, video_metadata=meta)
        return self.processor(text=[text], return_tensors="pt", **kw).to(self.model.device)

    @torch.no_grad()
    def _forward(self, inputs, extra_ids):
        x = dict(inputs)
        x["input_ids"] = torch.cat([inputs["input_ids"], extra_ids], dim=1)
        x["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(extra_ids)], dim=1)
        if "mm_token_type_ids" in x:
            x["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros_like(extra_ids)], dim=1)
        # logits fp32 o n vi tri cuoi: fp16 lam tron lam 2 option gan nhau bang nhau
        n = extra_ids.shape[1]
        hidden = self.model.model(**x).last_hidden_state[0, -(n + 1):-1].float()
        weight = self.model.lm_head.weight
        logits = torch.cat([hidden @ weight[i:i + 16384].float().T for i in range(0, weight.shape[0], 16384)], dim=1)
        return torch.log_softmax(logits, dim=-1)     # [n, vocab], dong i du doan extra_ids[i]

    def score_labels(self, frames, prompt, labels, duration=None):
        """Teacher forcing: log p(label | video, prompt), chia cho so token cua label."""
        return self._score(self._inputs(frames, prompt, duration), labels)

    def score_chat(self, messages, labels):
        return self._score(self._chat_inputs(messages), labels)

    @torch.no_grad()
    def _score(self, inputs, labels):
        ids = [self.tokenizer(l, add_special_tokens=False).input_ids for l in labels]
        device = self.model.device

        # nhanh: cac label chi khac nhau o token cuoi (vd "O1".."O4" = "O" + so) -> 1 luot la du
        head = ids[0][:-1]
        if all(i[:-1] == head for i in ids) and len({i[-1] for i in ids}) == len(ids):
            x = torch.tensor([head + [ids[0][-1]]], device=device)
            logp = self._forward(inputs, x)
            head_lp = sum(logp[t, tok].item() for t, tok in enumerate(head))
            return np.array([(head_lp + logp[len(head), i[-1]].item()) / len(i) for i in ids])

        scores = []
        for i in ids:
            x = torch.tensor([i], device=device)
            logp = self._forward(inputs, x)
            scores.append(logp.gather(1, x[0][:, None]).sum().item() / len(i))
        return np.array(scores)

    def generate(self, frames, prompt, duration=None, **kw):
        return self._generate(self._inputs(frames, prompt, duration), **kw)

    def chat(self, messages, **kw):
        return self._generate(self._chat_inputs(messages), **kw)

    @torch.no_grad()
    def _generate(self, inputs, **kw):
        kw.setdefault("max_new_tokens", self.max_new_tokens)
        kw.setdefault("do_sample", False)
        if kw.get("stop_strings"):
            kw["tokenizer"] = self.tokenizer
        out = self.model.generate(**inputs, **kw)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------- Action Classifier cua UniSoccer
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
