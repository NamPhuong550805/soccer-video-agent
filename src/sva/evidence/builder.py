from pathlib import Path

from sva.constants import SOCCERBENCH_ROOT
from sva.data.loader import canon
from sva.models.scoring import dist_from_logprobs, margin_confidence
from sva.schema import Evidence


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
