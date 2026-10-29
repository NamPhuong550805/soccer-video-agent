"""Chon cau hinh cac nhanh tren dev (khong dung test); ghi results/dev/*.json cho notebook.

    python scripts/04_dev_tuning.py q9         # so khop replay: encoder x so frame x vien den, lambda, nhiet do, UniSoccer
    python scripts/04_dev_tuning.py q14        # gop goc quay: bo phieu / linear pool / log pool / do ro / thang thu tu
    python scripts/04_dev_tuning.py branch_a   # nhanh A: kenh b0 / action (q10), b0 / event / clip (q11) va cach gop
"""
import argparse
import gc
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sva.config import RESULTS_DIR, record_run, setup_logging
from sva.data import canon, parse_views
from sva.eval import load_predictions, load_split
from sva.evidence import hard_vote, linear_pool, log_pool, ordinal_fusion
from sva.perception import PerceptionCache

log = logging.getLogger(__name__)
DEV = RESULTS_DIR / "dev"
TEMPS = (0.002, 0.005, 0.01, 0.02, 0.05)


# ---------------------------------------------------------------- q9: so khop replay
FRAMES = (4, 12, 24)
LAMBDAS = (0, 0.05, 0.1, 0.2, 0.5)


def variant_rows(samples, cache, enc, n):
    """Moi mau: [score, Omega, max cos] cua tung ung vien (lambda = 0)."""
    from sva.branches import candidate_order, chamfer
    rows = []
    for s in samples:
        rep = cache.get_embedding(s.materials[0], n, 336, enc).astype(np.float32)
        cands = [cache.get_embedding(s.materials[c], n, 336, enc).astype(np.float32) for c in candidate_order(s)]
        rows.append([[round(v, 5) for v in chamfer(rep, c, 0)[1:]] for c in cands])
    return rows


def unisoccer_rows(samples):
    """Kenh phu: 1 - JS giua phan phoi 23 lop cua replay va ung vien; entropy q9 vs q10."""
    from sva.branches import candidate_order
    from sva.models import ActionClassifier
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


def softmax_rows(x, T):
    z = x / T
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def part_q9():
    from sva.branches import FrameEncoder
    from sva.perception import HSVEncoder
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
        d = (g.mean() - w.mean()) / np.sqrt((g.var() + w.var()) / 2)    # do tach dung / sai, khong phu thuoc thang do
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
    nll = {T: float(-np.log(softmax_rows(X, T)[np.arange(len(gold)), gold]).mean()) for T in TEMPS}
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

    path = DEV / "q9_matching.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    log.info("da ghi %s", path)
    hsv = max(100 * (np.array(r)[..., 0].argmax(1) == gold).mean() for k, r in out["variants"].items() if k.startswith("hsv"))
    return (f"q9 dev: chon {key} lam={lam} T={T}, accuracy {acc:.1f}; HSV tot nhat {hsv:.1f}; "
            f"UniSoccer mot minh {100 * (U.argmax(1) == gold).mean():.1f}, gop tot nhat {max(fused.values()):.1f}")


# ---------------------------------------------------------------- q14: gop nhieu goc quay
def collect_views(samples, backend):
    """Moi cau: B0 (ca mau 1 prompt), phan phoi va do ro (VLM tu danh gia) cua tung goc."""
    from sva.baselines import B0Local
    from sva.branches import MultiView, classify_dimension, ordinal_scale
    scorer = B0Local(backend)
    mv = MultiView(scorer)
    rows = []
    for i, s in enumerate(samples):
        views = mv.views(s)
        rows.append({"index": s.index, "gold": s.gold_key, "keys": s.option_keys, "dim": classify_dimension(s),
                     "texts": {k: canon(t) for k, t in s.option_texts.items()},
                     "cameras": [v["camera"] for v in parse_views(s.question)],
                     "scale": ordinal_scale(s, classify_dimension(s)),
                     "holistic": scorer.score(s).dist, "views": [ev.dist for ev, _ in views],
                     "quality": [q for _, q in views]})
        if (i + 1) % 10 == 0:
            log.info("%d/%d", i + 1, len(samples))
    return rows


def fusions(r):
    """Du doan cua tung cach gop cho 1 cau."""
    v, q, hol = r["views"], r["quality"], r["holistic"]
    out = {"B0 (ca mau 1 prompt)": hol, "goc 1 (main camera)": v[0], "goc 2": v[1],
           "bo phieu nhan cung": hard_vote(v), "linear pool": linear_pool(v),
           "log pool": log_pool(v), "log pool + do ro": log_pool(v, q)}
    if r["scale"]:
        out["thang thu tu"] = ordinal_fusion(v, r["scale"])
    out["B0 + log pool (trung binh)"] = linear_pool([hol, out["log pool"]])
    out["B0 + log pool + do ro"] = linear_pool([hol, out["log pool + do ro"]])
    return out


def tie_correct(dist, gold):
    """Hoa (bo phieu) thi tinh dung mot phan: 1 / so nhan hoa."""
    top = max(dist.values())
    best = [k for k, p in dist.items() if abs(p - top) < 1e-9]
    return (gold in best) / len(best)


def prior_correct(rows):
    """Moc khong nhin video: chon nhan hay dung nhat cua chieu do o cac cau khac (leave-one-out)."""
    out = []
    for i, r in enumerate(rows):
        c = Counter(x["texts"][x["gold"]] for j, x in enumerate(rows) if j != i and x["dim"] == r["dim"])
        out.append(max(r["keys"], key=lambda k: c[r["texts"][k]]) == r["gold"])
    return out


def part_q14():
    path = DEV / "q14_multiview.json"
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
        log.info("doc lai %s (%d cau)", path, len(rows))
    else:
        rows = collect_views(load_split("q14", "dev"), "qwen_local_3b")

    var = [fusions(r) for r in rows]
    names = list(dict.fromkeys(k for v in var for k in v))
    dims = sorted({r["dim"] for r in rows})
    log.info("\n%-28s %6s  %s", "cach", "tat ca", "  ".join(f"{d[:9]:>9}" for d in dims))
    log.info("%-28s %6d  %s", "(so cau)", len(rows), "  ".join(f"{sum(r['dim'] == d for r in rows):9d}" for d in dims))
    acc = {}
    for n in names:
        ok = [(r["dim"], tie_correct(v[n], r["gold"])) for r, v in zip(rows, var) if n in v]
        acc[n] = 100 * np.mean([c for _, c in ok])
        per = [np.mean([c for d, c in ok if d == dim]) * 100 if any(d == dim for d, _ in ok) else float("nan")
               for dim in dims]
        log.info("%-28s %6.1f  %s", n, acc[n], "  ".join(f"{x:9.1f}" for x in per))

    two = [v for r, v in zip(rows, var) if len(r["views"]) == 2]
    ties = np.mean([len(set(max(d, key=d.get) for d in r["views"])) > 1 for r in rows if len(r["views"]) == 2])
    log.info("\ncau 2 goc: %d; hai goc chon khac nhau (bo phieu hoa 1-1): %.0f%%", len(two), 100 * ties)
    log.info("moc khong nhin video (nhan pho bien theo chieu, leave-one-out): %.1f", 100 * np.mean(prior_correct(rows)))
    q = np.array([x for r in rows for x in r["quality"]])
    log.info("do ro tu danh gia: trung binh %.2f, do lech %.2f", q.mean(), q.std())

    path.write_text(json.dumps({"rows": rows, "accuracy": acc}), encoding="utf-8")
    log.info("da ghi %s", path)
    best = max(acc, key=acc.get)
    return (f"q14 dev ({len(rows)} cau): B0 {acc['B0 (ca mau 1 prompt)']:.1f}; bo phieu {acc['bo phieu nhan cung']:.1f}; "
            f"log pool {acc['log pool']:.1f}; tot nhat {best} {acc[best]:.1f}; hoa 1-1 {100 * ties:.0f}%")


# ---------------------------------------------------------------- nhanh A (q10, q11)
# kenh: b0 = VLM hoi thang (buoc dau cua b0_loop_trace tren dev); action = UniSoccer chi giu lop co trong phuong an (q10);
# event = sum_c P_video(c) * P(cau binh luan noi ve c) (q11); clip = cosine CLIP frame - cau binh luan (q11)
PLACEHOLDER = {"[PLAYER]": "a player", "[TEAM]": "the team", "[REFEREE]": "the referee", "[COACH]": "the coach"}


def plain(text):
    for k, v in PLACEHOLDER.items():
        text = text.replace(k, v)
    return re.sub(r"\s+", " ", text).strip()


def clip_scores(samples):
    """[mau][phuong an] cosine trung binh giua 8 frame (cung frame B0 dung) va cau binh luan."""
    from sva.perception import build_encoder
    cache, enc = PerceptionCache(), build_encoder("clip_vit_l", "cuda")
    rows = []
    for s in samples:
        img = cache.get_embedding(s.materials[0], 8, 336, enc).astype(np.float32)
        txt = enc.encode_text([plain(s.option_texts[k]) for k in s.option_keys])
        rows.append((img @ txt.T).mean(0).tolist())
    del enc
    gc.collect()
    return rows


def b0_dists(task_id, samples):
    trace = {r["index"]: r for r in load_predictions("b0_loop_trace", task_id, "dev")}
    return [trace[s.index]["evidence"][0]["dist"] for s in samples]


def softmax_dist(scores, keys, T):
    z = np.array(scores) / T
    z = np.exp(z - z.max())
    return dict(zip(keys, (z / z.sum()).tolist()))


def acc_of(dists, samples):
    return 100 * np.mean([max(d, key=d.get) == s.gold_key for d, s in zip(dists, samples)])


def nll_of(dists, samples):
    return float(-np.mean([np.log(max(d[s.gold_key], 1e-12)) for d, s in zip(dists, samples)]))


def part_branch_a():
    import torch
    from sva.baselines import B0Local
    from sva.branches import BranchA
    q10, q11 = load_split("q10", "dev"), load_split("q11", "dev")
    clip = clip_scores(q11)
    gc.collect()
    torch.cuda.empty_cache()
    branch = BranchA(B0Local("qwen_local_3b"))
    action = [branch.run(s)[0].dist for s in q10]
    event = []
    for i, s in enumerate(q11):
        event.append(branch.run(s)[0].dist)
        if (i + 1) % 50 == 0:
            log.info("q11 event %d/%d", i + 1, len(q11))

    res = {"q10": {"index": [s.index for s in q10], "gold": [s.gold_key for s in q10],
                   "b0": b0_dists("q10", q10), "action": action},
           "q11": {"index": [s.index for s in q11], "gold": [s.gold_key for s in q11],
                   "b0": b0_dists("q11", q11), "event": event, "clip_raw": clip}}
    clip_by_t = {T: [softmax_dist(c, s.option_keys, T) for c, s in zip(clip, q11)] for T in TEMPS}
    T = min(TEMPS, key=lambda t: nll_of(clip_by_t[t], q11))           # nhiet do kenh CLIP theo NLL tren dev
    res["q11"]["clip"], res["q11"]["clip_T"] = clip_by_t[T], T

    summary = {}
    for task, samples, chans in (("q10", q10, ["b0", "action"]), ("q11", q11, ["b0", "event", "clip"])):
        r = res[task]
        rows = {c: r[c] for c in chans}
        for i, a in enumerate(chans):
            for b in chans[i + 1:]:
                rows[f"{a} + {b} (trung binh)"] = [linear_pool([x, y]) for x, y in zip(r[a], r[b])]
                rows[f"{a} x {b} (log pool)"] = [log_pool([x, y]) for x, y in zip(r[a], r[b])]
        if len(chans) == 3:
            rows["ca 3 (log pool)"] = [log_pool(list(x)) for x in zip(*(r[c] for c in chans))]
        log.info("\n%s dev (%d cau)%s", task, len(samples), f", CLIP T={T}" if task == "q11" else "")
        for name, d in rows.items():
            summary[f"{task} {name}"] = acc_of(d, samples)
            log.info("  %-28s %5.1f   NLL %.3f", name, acc_of(d, samples), nll_of(d, samples))

    path = DEV / "branch_a.json"
    path.write_text(json.dumps({"rows": res, "accuracy": summary}), encoding="utf-8")
    log.info("da ghi %s", path)
    return "dev: " + "; ".join(f"{k} {v:.1f}" for k, v in summary.items() if " x " not in k and "+" not in k)


def main():
    parts = {"q9": part_q9, "q14": part_q14, "branch_a": part_branch_a}
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=list(parts))
    args = parser.parse_args()
    setup_logging(f"04_dev_tuning_{args.part}")
    DEV.mkdir(parents=True, exist_ok=True)
    record_run(parts[args.part]())


if __name__ == "__main__":
    main()
