import numpy as np
import torch
from PIL import Image

from sva.config import load_config


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

    @torch.no_grad()
    def encode(self, frames):
        batch = torch.stack([self.preprocess(Image.fromarray(f)) for f in frames]).to(self.device)
        return l2_normalize(self.model.encode_image(batch).float().cpu().numpy())


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


def build_encoder(name, device="cpu"):
    """name: clip_vit_l | siglip | dinov2_vitb14 (xem configs/models.yaml)"""
    encoders = load_config("models")["encoder"]
    if name not in encoders:
        raise ValueError(f"khong co encoder {name!r}, chon trong {list(encoders)}")
    cfg = encoders[name]
    if cfg["lib"] == "open_clip":
        return OpenCLIPEncoder(name, cfg["model_id"], cfg["pretrained"], device)
    return DINOv2Encoder(name, cfg["repo"], cfg["model_id"], device)
