import numpy as np
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration


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
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # fps cua chuoi frame da lay, de model biet khoang cach thoi gian giua cac frame
        n = len(frames)
        fps = n / duration if duration else 2.0
        meta = [{"total_num_frames": n, "fps": fps, "frames_indices": list(range(n))}]
        inputs = self.processor(text=[text], videos=[np.asarray(frames)], video_metadata=meta, return_tensors="pt")
        return inputs.to(self.model.device)

    @torch.no_grad()
    def score_labels(self, frames, prompt, labels, duration=None):
        """Teacher forcing: log p(label | video, prompt), chia cho so token cua label."""
        inputs = self._inputs(frames, prompt, duration)
        scores = []
        for label in labels:
            label_ids = self.tokenizer(label, add_special_tokens=False, return_tensors="pt").input_ids.to(self.model.device)
            n = label_ids.shape[1]
            x = dict(inputs)
            x["input_ids"] = torch.cat([inputs["input_ids"], label_ids], dim=1)
            x["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(label_ids)], dim=1)
            if "mm_token_type_ids" in x:
                x["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros_like(label_ids)], dim=1)

            # chi can logits o n+1 vi tri cuoi, tranh tinh logits cho ca chuoi (vocab 151k)
            logits = self.model(**x, logits_to_keep=n + 1).logits[0, :-1].float()
            logp = torch.log_softmax(logits, dim=-1).gather(1, label_ids[0][:, None]).sum()
            scores.append(logp.item() / n)
        return np.array(scores)

    @torch.no_grad()
    def generate(self, frames, prompt, duration=None, **kw):
        inputs = self._inputs(frames, prompt, duration)
        kw.setdefault("max_new_tokens", self.max_new_tokens)
        kw.setdefault("do_sample", False)
        out = self.model.generate(**inputs, **kw)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
