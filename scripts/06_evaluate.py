"""Tinh metric tu results/predictions/, ghi bang ra results/tables/

    python scripts/06_evaluate.py --systems b0_local
    python scripts/06_evaluate.py --systems b0_local b0_api --compare b0_local b0_api
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sva.config import record_run, setup_logging
from sva.constants import RESULTS_DIR, TASK_IDS, TASK_META
from sva.eval.metrics import accuracy, accuracy_ci, chance, ece, position_rate
from sva.eval.runner import load_predictions
from sva.eval.stats import mcnemar

log = logging.getLogger(__name__)


def table(systems, split):
    lines = [f"| system | task | paper | n | accuracy | 95% CI | ngau nhien | ECE | chon O1 / dap an O1 | loi |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name in systems:
        accs = []
        for t in TASK_IDS:
            recs = load_predictions(name, t, split)
            if not recs:
                continue
            lo, hi = accuracy_ci(recs)
            pf, gf = position_rate(recs)
            errors = sum(1 for r in recs if r.get("error"))
            accs.append(accuracy(recs))
            lines.append(f"| {name} | {t} | {TASK_META[t]['paper_id']} | {len(recs)} | {100 * accuracy(recs):.1f} | "
                         f"{100 * lo:.1f}-{100 * hi:.1f} | {100 * chance(recs):.1f} | {ece(recs):.3f} | "
                         f"{100 * pf:.0f}% / {100 * gf:.0f}% | {errors} |")
        if accs:
            lines.append(f"| {name} | **trung binh** | | | **{100 * sum(accs) / len(accs):.1f}** | | | | | |")
    return lines


def compare(a, b, split):
    lines = [f"| task | n | {a} | {b} | b ({a} sai, {b} dung) | c ({a} dung, {b} sai) | chi2 | p |",
             "|---|---|---|---|---|---|---|---|"]
    for t in TASK_IDS:
        ra, rb = load_predictions(a, t, split), load_predictions(b, t, split)
        if not ra or not rb:
            continue
        common = {r["index"] for r in ra} & {r["index"] for r in rb}
        ra = [r for r in ra if r["index"] in common]
        rb = [r for r in rb if r["index"] in common]
        m = mcnemar(ra, rb)
        lines.append(f"| {t} | {m['n']} | {100 * accuracy(ra):.1f} | {100 * accuracy(rb):.1f} | {m['b']} | {m['c']} | "
                     f"{m['chi2']:.2f} | {m['p_value']:.4f} |")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", required=True)
    parser.add_argument("--compare", nargs=2, default=None)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    args = parser.parse_args()

    setup_logging("06_evaluate")
    lines = [f"# Ket qua ({args.split})", ""] + table(args.systems, args.split)
    if args.compare:
        lines += ["", f"## McNemar: {args.compare[0]} vs {args.compare[1]}", ""] + compare(*args.compare, args.split)

    out = RESULTS_DIR / "tables" / f"{'_'.join(args.systems)}_{args.split}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("\n".join(lines))
    log.info("\nda ghi %s", out)
    record_run(f"bang {out.relative_to(RESULTS_DIR).as_posix()}")


if __name__ == "__main__":
    main()
