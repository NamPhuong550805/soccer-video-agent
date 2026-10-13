import pytest

from sva.eval.metrics import accuracy, accuracy_ci, chance, ece, ordinal_mae, position_rate
from sva.eval.stats import mcnemar
from sva.schema import Sample


def rec(i, gold, pred, p=0.8, n=4):
    keys = [f"O{k}" for k in range(1, n + 1)]
    rest = (1 - p) / (n - 1)
    return {"index": i, "gold": gold, "pred": pred, "n_options": n,
            "dist": {k: (p if k == pred else rest) for k in keys}}


def test_accuracy_counts_missing_pred_as_wrong():
    recs = [rec(0, "O1", "O1"), rec(1, "O2", "O1"), {"index": 2, "gold": "O1", "pred": None, "n_options": 4}]
    assert accuracy(recs) == pytest.approx(1 / 3)


def test_wilson_ci_contains_estimate():
    recs = [rec(i, "O1", "O1" if i < 60 else "O2") for i in range(100)]
    lo, hi = accuracy_ci(recs)
    assert lo < 0.6 < hi
    assert hi - lo == pytest.approx(0.19, abs=0.01)


def test_chance():
    assert chance([rec(0, "O1", "O1", n=4), rec(1, "O1", "O1", n=2)]) == pytest.approx(0.375)


def test_ece_example():
    # 5 mau tu tin 0.9 dung 3, 5 mau tu tin 0.6 dung 3 -> ECE = 0.5*0.3 + 0.5*0 = 0.15
    recs = [rec(i, "O1", "O1" if i < 3 else "O2", p=0.9) for i in range(5)]
    recs += [rec(i, "O1", "O1" if i < 8 else "O2", p=0.6) for i in range(5, 10)]
    for r in recs[3:5]:
        r["pred"], r["dist"] = "O2", {"O1": 0.1 / 3, "O2": 0.9, "O3": 0.1 / 3, "O4": 0.1 / 3}
    assert ece(recs, n_bins=10) == pytest.approx(0.15)


def test_position_rate():
    recs = [rec(0, "O2", "O1"), rec(1, "O1", "O1"), rec(2, "O3", "O3"), rec(3, "O4", "O1")]
    pred_first, gold_first = position_rate(recs)
    assert pred_first == 0.75 and gold_first == 0.25


def test_ordinal_mae():
    s = Sample("q14", 0, "?", [], ["O1", "O2", "O4"],
               {"O1": "No card", "O2": "Yellow card", "O4": "Red card"}, "O2")
    from sva.constants import ORDINAL_SCALES
    assert ordinal_mae([rec(0, "O2", "O4", n=4)], [s], ORDINAL_SCALES["severity"]) == 1.0


def test_mcnemar():
    # A dung 15 mau ma B sai, B dung 30 mau ma A sai -> chi2 = 14^2 / 45
    a = [{"index": i, "gold": "O1", "pred": "O1" if i < 15 else "O2"} for i in range(45)]
    b = [{"index": i, "gold": "O1", "pred": "O2" if i < 15 else "O1"} for i in range(45)]
    r = mcnemar(a, b)
    assert (r["b"], r["c"]) == (30, 15)
    assert r["chi2"] == pytest.approx(196 / 45)
    assert r["p_value"] == pytest.approx(0.0369, abs=1e-3)
