import math
from collections import Counter

import numpy as np


def correct(r):
    return r.get("pred") is not None and r["pred"] == r["gold"]


def accuracy(recs):
    return sum(correct(r) for r in recs) / len(recs) if recs else float("nan")


def accuracy_ci(recs, alpha=0.05):
    """Khoang tin cay Wilson cho ty le dung."""
    n = len(recs)
    if n == 0:
        return float("nan"), float("nan")
    z = {0.05: 1.959964, 0.01: 2.575829}[alpha]
    p = accuracy(recs)
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def chance(recs):
    """Muc doan mo: trung binh 1 / so option."""
    return float(np.mean([1 / r["n_options"] for r in recs])) if recs else float("nan")


def top_prob(r):
    return max(r["dist"].values())


def ece(recs, n_bins=10):
    """Expected calibration error, do tin = xac suat cua phuong an duoc chon."""
    recs = [r for r in recs if r.get("dist")]
    if not recs:
        return float("nan")
    conf = np.array([top_prob(r) for r in recs])
    acc = np.array([correct(r) for r in recs], dtype=float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return float(sum((bins == b).mean() * abs(acc[bins == b].mean() - conf[bins == b].mean())
                     for b in range(n_bins) if (bins == b).any()))


def reliability(recs, n_bins=10):
    """(do tin trung binh, ty le dung, so mau) theo tung bin, de ve reliability diagram."""
    conf = np.array([top_prob(r) for r in recs])
    acc = np.array([correct(r) for r in recs], dtype=float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return [(conf[bins == b].mean(), acc[bins == b].mean(), int((bins == b).sum()))
            for b in range(n_bins) if (bins == b).any()]


def recall_at_1(recs):
    # q9 chi co 1 dap an dung nen Recall@1 trung voi accuracy
    return accuracy(recs)


def ordinal_mae(recs, samples, scale):
    """Sai so tuyet doi trung binh tren thang thu tu (vd no card=0 ... red card=2)."""
    from sva.data.loader import canon
    by_index = {s.index: s for s in samples}
    errs = []
    for r in recs:
        s = by_index[r["index"]]
        texts = {k: canon(t) for k, t in s.option_texts.items()}
        if r.get("pred") and all(t in scale for t in texts.values()):
            errs.append(abs(scale[texts[r["pred"]]] - scale[texts[r["gold"]]]))
    return float(np.mean(errs)) if errs else float("nan")


def accuracy_by(recs, key):
    groups = {}
    for r in recs:
        groups.setdefault(key(r), []).append(r)
    return {g: (accuracy(rs), len(rs)) for g, rs in sorted(groups.items(), key=lambda x: str(x[0]))}


def position_rate(recs):
    """Ty le chon option o vi tri dau tien, so voi ty le dap an dung nam o vi tri dau."""
    first = lambda r, k: k == sorted(r["dist"], key=lambda x: int(x[1:]))[0]
    recs = [r for r in recs if r.get("dist")]
    return (sum(first(r, r["pred"]) for r in recs) / len(recs),
            sum(first(r, r["gold"]) for r in recs) / len(recs))


def pred_counts(recs):
    return Counter(r.get("pred") for r in recs)
