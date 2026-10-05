import re
from collections import Counter

import numpy as np

from sva.data.loader import canon


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
