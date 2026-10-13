"""Chon cau hinh so khop replay (q9) tren dev; ghi results/dev/q9_matching.json cho notebook.

    python scripts/07_dev_matching.py

Thu: encoder (CLIP / SigLIP / DINOv2) x so frame moi clip (4 / 12 / 24) x co bo vien den hay khong,
phat thu tu thoi gian lambda, nhiet do softmax; moc HSV histogram cua spec; kenh phu UniSoccer (1 - JS).
"""
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sva.branches.matching import FrameEncoder, candidate_order, chamfer
from sva.config import record_run, setup_logging
from sva.constants import RESULTS_DIR
from sva.eval.runner import load_split
from sva.perception.cache import PerceptionCache
from sva.perception.encoder import HSVEncoder

log = logging.getLogger(__name__)
FRAMES = (4, 12, 24)
LAMBDAS = (0, 0.05, 0.1, 0.2, 0.5)
TEMPS = (0.002, 0.005, 0.01, 0.02, 0.05)


def variant_rows(samples, cache, enc, n):
    """Moi mau: [score, Omega, max cos] cua tung ung vien (lambda = 0)."""
    rows = []
    for s in samples:
        rep = cache.get_embedding(s.materials[0], n, 336, enc).astype(np.float32)
        cands = [cache.get_embedding(s.materials[c], n, 336, enc).astype(np.float32) for c in candidate_order(s)]
        rows.append([[round(v, 5) for v in chamfer(rep, c, 0)[1:]] for c in cands])
    return rows


def unisoccer_rows(samples):
    from sva.models.unisoccer import ActionClassifier
    cls = ActionClassifier()
    vec = lambda p: np.array([p[k] for k in sorted(p)])

    def js(a, b):
        m = (a + b) / 2
        kl = lambda x, y: float((x * np.log((x + 1e-12) / (y + 1e-12))).sum())
        return (kl(a, m) + kl(b, m)) / 2 / np.log(2)

    def entropy(paths):
        return float(np.mean([-(vec(cls.probs(p)) * np.log(vec(cls.probs(p)) + 1e-12)).sum() for p in paths]))

    rows = [[round(1 - js(vec(cls.probs(s.materials[0])), vec(cls.probs(s.materials[c]))), 5)
             for c in candidate_order(s)] for s in samples]
    ent = {"q9": entropy([p for s in samples for p in s.materials]),
           "q10": entropy([s.materials[0] for s in load_split("q10", "dev")])}
    return rows, ent


def softmax(x, T):
    z = x / T
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def main():
    setup_logging("07_dev_matching")
    samples, cache = load_split("q9", "dev"), PerceptionCache()
    gold = np.array([s.option_keys.index(s.gold_key) for s in samples])
    variants = [(e, n, True) for e in ("clip_vit_l", "siglip", "dinov2_vitb14") for n in FRAMES]
    variants += [("clip_vit_l", n, False) for n in FRAMES] + [("hsv", n, False) for n in FRAMES]

    out = {"index": [s.index for s in samples], "gold": gold.tolist(), "variants": {}}
    for enc_name, n, square in variants:
        enc = HSVEncoder() if enc_name == "hsv" else FrameEncoder(enc_name, square)
        key = f"{enc_name}|{n}|{'sq' if square else 'box'}"
        out["variants"][key] = variant_rows(samples, cache, enc, n)
        del enc
        gc.collect()
        log.info("xong %s", key)

    log.info("\n%-22s %s   d'   trung>0.95 (dung / sai)", "bien the", "  ".join(f"lam={l:<4}" for l in LAMBDAS))
    best = None
    for key, rows in out["variants"].items():
        R = np.array(rows)                                   # [mau, ung vien, (score, Omega, max cos)]
        accs = [100 * ((R[..., 0] - l * R[..., 1]).argmax(1) == gold).mean() for l in LAMBDAS]
        sc = R[..., 0]
        g, w = sc[np.arange(len(gold)), gold], np.array([np.delete(x, i) for x, i in zip(sc, gold)]).ravel()
        d = (g.mean() - w.mean()) / np.sqrt((g.var() + w.var()) / 2)    # do tach khong phu thuoc thang do
        mx = R[..., 2] > 0.95
        dup = 100 * mx[np.arange(len(gold)), gold].mean(), 100 * np.mean([np.delete(x, i).any() for x, i in zip(mx, gold)])
        log.info("%-22s %s   %.2f  %4.0f%% / %3.0f%%", key, "  ".join(f"{a:8.1f}" for a in accs), d, *dup)
        for a, l in zip(accs, LAMBDAS):
            if key.startswith("hsv") or (best and a <= best[0]):
                continue
            best = (a, key, l)

    acc, key, lam = best
    R = np.array(out["variants"][key])
    X = R[..., 0] - lam * R[..., 1]
    nll = {T: float(-np.log(softmax(X, T)[np.arange(len(gold)), gold]).mean()) for T in TEMPS}
    T = min(nll, key=nll.get)
    log.info("\nchon %s lambda=%s: accuracy dev %.1f; nhiet do theo NLL %s -> T=%s", key, lam, acc,
             {t: round(v, 3) for t, v in nll.items()}, T)

    uni, ent = unisoccer_rows(samples)
    U = np.array(uni)
    z = lambda A: (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-9)
    fused = {w: float(100 * ((z(X) + w * z(U)).argmax(1) == gold).mean()) for w in (0, 0.1, 0.25, 0.5, 1.0)}
    log.info("UniSoccer 1-JS mot minh: %.1f; entropy q9 %.2f vs q10 %.2f; gop voi trong so w: %s",
             100 * (U.argmax(1) == gold).mean(), ent["q9"], ent["q10"], {w: round(a, 1) for w, a in fused.items()})
    out.update(unisoccer=uni, entropy=ent, chosen={"variant": key, "lam": lam, "temperature": T, "accuracy": acc},
               nll=nll, fused=fused)

    path = RESULTS_DIR / "dev" / "q9_matching.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out), encoding="utf-8")
    log.info("da ghi %s", path)
    hsv = max(100 * (np.array(r)[..., 0].argmax(1) == gold).mean() for k, r in out["variants"].items() if k.startswith("hsv"))
    record_run(f"q9 dev: chon {key} lam={lam} T={T}, accuracy {acc:.1f}; HSV tot nhat {hsv:.1f}; "
               f"UniSoccer mot minh {100 * (U.argmax(1) == gold).mean():.1f}, gop tot nhat {max(fused.values()):.1f}")


if __name__ == "__main__":
    main()
