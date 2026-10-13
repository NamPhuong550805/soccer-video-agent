"""Gop nhieu phan phoi (vd moi goc quay mot phan phoi) thanh mot. dists: list dict tren cung cac key."""
import numpy as np


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
