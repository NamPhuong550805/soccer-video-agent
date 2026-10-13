"""Chon cach lam nhanh A (q10, q11) tren dev; ghi results/dev/branch_a.json cho notebook.

    python scripts/09_dev_branch_a.py

Cac kenh (moi kenh la phan phoi tren phuong an):
  b0      VLM hoi thang (lay tu lan chay b0_loop_trace tren dev, buoc dau)
  action  q10: phan phoi 23 lop cua UniSoccer, chi giu cac lop co trong phuong an
  event   q11: sum_c P_video(c) * P(binh luan noi ve c)
  clip    q11: cosine CLIP giua frame va cau binh luan (trung binh theo frame)
"""
import gc
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sva.config import record_run, setup_logging
from sva.constants import RESULTS_DIR
from sva.evidence import fusion
from sva.eval.runner import load_predictions, load_split
from sva.perception.cache import PerceptionCache

log = logging.getLogger(__name__)
OUT = RESULTS_DIR / "dev" / "branch_a.json"
PLACEHOLDER = {"[PLAYER]": "a player", "[TEAM]": "the team", "[REFEREE]": "the referee", "[COACH]": "the coach"}


def plain(text):
    for k, v in PLACEHOLDER.items():
        text = text.replace(k, v)
    return re.sub(r"\s+", " ", text).strip()


def clip_scores(samples):
    """[mau][phuong an] cosine trung binh giua 8 frame (cung frame B0 dung) va cau binh luan."""
    from sva.perception.encoder import build_encoder
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


def acc(dists, samples):
    return 100 * np.mean([max(d, key=d.get) == s.gold_key for d, s in zip(dists, samples)])


def nll(dists, samples):
    return float(-np.mean([np.log(max(d[s.gold_key], 1e-12)) for d, s in zip(dists, samples)]))


def main():
    setup_logging("09_dev_branch_a")
    from sva.baselines.b0_local import B0Local
    from sva.branches.branch_a import BranchA
    q10, q11 = load_split("q10", "dev"), load_split("q11", "dev")
    clip = clip_scores(q11)
    torch_free()
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
    # nhiet do cho kenh CLIP theo NLL tren dev
    temps = (0.002, 0.005, 0.01, 0.02, 0.05)
    clip_by_t = {T: [softmax_dist(c, s.option_keys, T) for c, s in zip(clip, q11)] for T in temps}
    T = min(temps, key=lambda t: nll(clip_by_t[t], q11))
    res["q11"]["clip"], res["q11"]["clip_T"] = clip_by_t[T], T

    summary = {}
    for task, samples, chans in (("q10", q10, ["b0", "action"]), ("q11", q11, ["b0", "event", "clip"])):
        r = res[task]
        rows = {c: r[c] for c in chans}
        for i, a in enumerate(chans):
            for b in chans[i + 1:]:
                rows[f"{a} + {b} (trung binh)"] = [fusion.linear_pool([x, y]) for x, y in zip(r[a], r[b])]
                rows[f"{a} x {b} (log pool)"] = [fusion.log_pool([x, y]) for x, y in zip(r[a], r[b])]
        if len(chans) == 3:
            rows["ca 3 (log pool)"] = [fusion.log_pool(list(x)) for x in zip(*(r[c] for c in chans))]
        log.info("\n%s dev (%d cau)%s", task, len(samples), f", CLIP T={T}" if task == "q11" else "")
        for name, d in rows.items():
            summary[f"{task} {name}"] = acc(d, samples)
            log.info("  %-28s %5.1f   NLL %.3f", name, acc(d, samples), nll(d, samples))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": res, "accuracy": summary}), encoding="utf-8")
    log.info("da ghi %s", OUT)
    record_run("dev: " + "; ".join(f"{k} {v:.1f}" for k, v in summary.items() if " x " not in k and "+" not in k))


def torch_free():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
