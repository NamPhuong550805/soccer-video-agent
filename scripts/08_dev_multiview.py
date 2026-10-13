"""Chon cach gop da goc quay (q14) tren dev; ghi results/dev/q14_multiview.json cho notebook.

    python scripts/08_dev_multiview.py

Moi cau: B0 (ca mau trong 1 prompt), phan phoi tung goc quay va do ro tung goc (VLM tu danh gia).
So sanh: bo phieu nhan cung / linear pool / log pool / log pool + do ro / gop theo thang thu tu.
"""
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sva.baselines.b0_local import B0Local
from sva.branches.multiview import MultiView, classify_dimension, ordinal_scale
from sva.config import record_run, setup_logging
from sva.constants import RESULTS_DIR
from sva.data.loader import canon, parse_views
from sva.eval.runner import load_split
from sva.evidence import fusion

log = logging.getLogger(__name__)
OUT = RESULTS_DIR / "dev" / "q14_multiview.json"


def collect(samples, backend):
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


def variants(r):
    """Du doan cua tung cach gop cho 1 cau."""
    v, q, hol = r["views"], r["quality"], r["holistic"]
    out = {"B0 (ca mau 1 prompt)": hol, "goc 1 (main camera)": v[0], "goc 2": v[1],
           "bo phieu nhan cung": fusion.hard_vote(v), "linear pool": fusion.linear_pool(v),
           "log pool": fusion.log_pool(v), "log pool + do ro": fusion.log_pool(v, q)}
    if r["scale"]:
        out["thang thu tu"] = fusion.ordinal_fusion(v, r["scale"])
    out["B0 + log pool (trung binh)"] = fusion.linear_pool([hol, out["log pool"]])
    out["B0 + log pool + do ro"] = fusion.linear_pool([hol, out["log pool + do ro"]])
    return out


def correct(dist, gold):
    """Hoa (bo phieu) thi tinh dung mot phan: 1 / so nhan hoa."""
    top = max(dist.values())
    best = [k for k, p in dist.items() if abs(p - top) < 1e-9]
    return (gold in best) / len(best)


def main():
    setup_logging("08_dev_multiview")
    samples = load_split("q14", "dev")
    if OUT.exists():
        rows = json.loads(OUT.read_text(encoding="utf-8"))["rows"]
        log.info("doc lai %s (%d cau)", OUT, len(rows))
    else:
        rows = collect(samples, "qwen_local_3b")

    var = [variants(r) for r in rows]
    names = list(dict.fromkeys(k for v in var for k in v))
    dims = sorted({r["dim"] for r in rows})
    log.info("\n%-28s %6s  %s", "cach", "tat ca", "  ".join(f"{d[:9]:>9}" for d in dims))
    log.info("%-28s %6d  %s", "(so cau)", len(rows), "  ".join(f"{sum(r['dim'] == d for r in rows):9d}" for d in dims))
    acc = {}
    for n in names:
        ok = [(r["dim"], correct(v[n], r["gold"])) for r, v in zip(rows, var) if n in v]
        acc[n] = 100 * np.mean([c for _, c in ok])
        per = [np.mean([c for d, c in ok if d == dim]) * 100 if any(d == dim for d, _ in ok) else float("nan")
               for dim in dims]
        log.info("%-28s %6.1f  %s", n, acc[n], "  ".join(f"{x:9.1f}" for x in per))

    # bo phieu nhan cung: ty le hoa o cau 2 goc
    two = [v for r, v in zip(rows, var) if len(r["views"]) == 2]
    ties = np.mean([len(set(max(d, key=d.get) for d in r["views"])) > 1 for r in rows if len(r["views"]) == 2])
    log.info("\ncau 2 goc: %d; hai goc chon khac nhau (bo phieu hoa 1-1): %.0f%%", len(two), 100 * ties)
    log.info("moc khong nhin video (nhan pho bien theo chieu, leave-one-out): %.1f", 100 * np.mean(prior_correct(rows)))
    q = np.array([x for r in rows for x in r["quality"]])
    log.info("do ro tu danh gia: trung binh %.2f, do lech %.2f", q.mean(), q.std())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "accuracy": acc}), encoding="utf-8")
    log.info("da ghi %s", OUT)
    best = max(acc, key=acc.get)
    record_run(f"q14 dev ({len(rows)} cau): B0 {acc['B0 (ca mau 1 prompt)']:.1f}; bo phieu {acc['bo phieu nhan cung']:.1f}; "
               f"log pool {acc['log pool']:.1f}; tot nhat {best} {acc[best]:.1f}; hoa 1-1 {100 * ties:.0f}%")


def prior_correct(rows):
    """Moc khong nhin video: chon phuong an co nhan hay dung nhat cua chieu do o cac cau khac (leave-one-out)."""
    out = []
    for i, r in enumerate(rows):
        c = Counter(x["texts"][x["gold"]] for j, x in enumerate(rows) if j != i and x["dim"] == r["dim"])
        out.append(max(r["keys"], key=lambda k: c[r["texts"][k]]) == r["gold"])
    return out


if __name__ == "__main__":
    main()
