"""Evidence: phan phoi tren dung cac phuong an cua mau (dau ra chung cua moi cong cu), cach dung va cach gop."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from sva.config import SOCCERBENCH_ROOT
from sva.data import canon
from sva.models import dist_from_logprobs, margin_confidence


# ---------------------------------------------------------------- Evidence
ClaimType = Literal["categorical", "ordinal", "positional", "text", "failure"]


@dataclass
class Evidence:
    """Ket qua cua mot tool: phan phoi tren dung cac option cua mau, khong phai mot chuoi."""
    tool_name: str
    claim_type: ClaimType
    option_keys: list[str]
    dist: dict[str, float]          # tong = 1, chi tren option_keys
    confidence: float               # p(top1) - p(top2)
    raw_observation: str = ""
    material_ref: dict = field(default_factory=dict)   # {"clips": [...], "timestamps": [...]}
    ordinal_scale: Optional[dict[str, float]] = None
    failed: bool = False

    def __post_init__(self):
        if set(self.dist) != set(self.option_keys):
            raise ValueError(f"dist {sorted(self.dist)} khong khop option_keys {self.option_keys}")
        if abs(sum(self.dist.values()) - 1.0) > 1e-6:
            raise ValueError(f"tong dist = {sum(self.dist.values())}")

    @property
    def pred(self):
        return max(self.option_keys, key=self.dist.get)

    @classmethod
    def failure(cls, tool_name, keys, reason):
        n = len(keys)
        return cls(tool_name, "failure", list(keys), {k: 1 / n for k in keys}, 0.0, reason, failed=True)

    def to_json(self):
        return asdict(self)


# ---------------------------------------------------------------- Dung Evidence
def rel_path(path):
    # luu duong dan tuong doi trong ket qua, khong de lo duong dan tren may
    try:
        return Path(path).resolve().relative_to(SOCCERBENCH_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _ref(sample, timestamps=None):
    ref = {"clips": [rel_path(p) for p in sample.materials]}
    if timestamps is not None:
        ref["timestamps"] = list(timestamps)
    return ref


def build_categorical(tool_name, sample, logprobs, raw_obs="", T=1.0, timestamps=None):
    dist = dist_from_logprobs(logprobs, sample.option_keys, T)
    return Evidence(tool_name, "categorical", sample.option_keys, dist, margin_confidence(dist),
                    raw_obs, _ref(sample, timestamps))


def build_ordinal(tool_name, sample, logprobs, scale, raw_obs="", T=1.0, timestamps=None):
    """Nhu categorical nhung gan them gia tri thu tu cho tung option (vd no card=0, yellow=1, red=2)."""
    ordinal = {}
    for k in sample.option_keys:
        text = canon(sample.option_texts[k])
        if text not in scale:
            raise ValueError(f"option {k}={sample.option_texts[k]!r} khong co trong thang {list(scale)}")
        ordinal[k] = scale[text]
    dist = dist_from_logprobs(logprobs, sample.option_keys, T)
    return Evidence(tool_name, "ordinal", sample.option_keys, dist, margin_confidence(dist),
                    raw_obs, _ref(sample, timestamps), ordinal_scale=ordinal)


def build_positional(tool_name, sample, scores, raw_obs="", T=0.05):
    """q9: diem giong nhau cua tung ung vien (vd cosine similarity), T nho vi diem nam trong [-1, 1]."""
    dist = dist_from_logprobs(scores, sample.option_keys, T)
    return Evidence(tool_name, "positional", sample.option_keys, dist, margin_confidence(dist),
                    raw_obs, _ref(sample))


def expected_value(ev):
    """Gia tri ky vong tren thang thu tu: sum_k p(k) * scale(k)."""
    return sum(ev.dist[k] * ev.ordinal_scale[k] for k in ev.option_keys)


# ---------------------------------------------------------------- Gop nhieu phan phoi
def _norm(weights, n):
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    return w / w.sum()


def hard_vote(dists, weights=None):
    """Kieu SoccerAgent: moi goc chot 1 nhan, dem phieu. Hoa phieu thi cac nhan hoa chia deu."""
    keys = list(dists[0])
    votes = dict.fromkeys(keys, 0.0)
    for d, w in zip(dists, _norm(weights, len(dists))):
        votes[max(d, key=d.get)] += float(w)
    return {k: v / sum(votes.values()) for k, v in votes.items()}


def linear_pool(dists, weights=None):
    w = _norm(weights, len(dists))
    return {k: float(sum(wi * d[k] for wi, d in zip(w, dists))) for k in dists[0]}


def log_pool(dists, weights=None):
    """p(k) ~ prod_v p_v(k)^w_v, tinh trong log de khong tran so."""
    w = _norm(weights, len(dists))
    keys = list(dists[0])
    lp = sum(wi * np.log(np.clip([d[k] for k in keys], 1e-12, None)) for wi, d in zip(w, dists))
    p = np.exp(lp - lp.max())
    return dict(zip(keys, (p / p.sum()).tolist()))


def ordinal_fusion(dists, scale, weights=None, tau=0.25):
    """Thang thu tu (vd the phat): mu = sum_v w_v * E_v[scale], phan phoi ~ exp(-|scale(k) - mu| / tau)."""
    w = _norm(weights, len(dists))
    keys = list(dists[0])
    mu = sum(wi * sum(d[k] * scale[k] for k in keys) for wi, d in zip(w, dists))
    s = -np.abs(np.array([scale[k] for k in keys]) - mu) / tau
    p = np.exp(s - s.max())
    return dict(zip(keys, (p / p.sum()).tolist()))
