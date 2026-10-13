import numpy as np
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from transformers.integrations import sdpa_attention

# torch tren Windows khong co flash-attention, nen sdpa voi enable_gqa chi chay kernel "math" (bo nho ~L^2,
# prompt 3.8k token can them 2.5GB). Lap lai key/value truoc thi dung duoc kernel memory-efficient (0.5GB).
sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False


class QwenVLBackend:
    def __init__(self, name, model_id, quantization=None, max_new_tokens=128, device_map="auto"):
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
        # chi tinh logits o n vi tri cuoi va o fp32: logits fp16 lam tron nen 2 option gan nhau
        # co the ra dung cung mot gia tri, roi argmax luon chon option dung truoc
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
