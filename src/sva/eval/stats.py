from scipy.stats import binomtest, chi2

from sva.eval.metrics import correct


def mcnemar(recs_a, recs_b):
    """So 2 he thong tren cung tap mau.

    b = so mau A sai B dung, c = so mau A dung B sai.
    chi2 = (|b - c| - 1)^2 / (b + c), hieu chinh lien tuc; b + c < 25 thi dung kiem dinh nhi thuc chinh xac.
    """
    a = {r["index"]: correct(r) for r in recs_a}
    b_map = {r["index"]: correct(r) for r in recs_b}
    common = a.keys() & b_map.keys()
    b = sum(1 for i in common if not a[i] and b_map[i])
    c = sum(1 for i in common if a[i] and not b_map[i])
    n = b + c
    if n == 0:
        return {"n": len(common), "b": 0, "c": 0, "chi2": 0.0, "p_value": 1.0}
    stat = (abs(b - c) - 1) ** 2 / n
    p = binomtest(b, n, 0.5).pvalue if n < 25 else chi2.sf(stat, 1)
    return {"n": len(common), "b": b, "c": c, "chi2": stat, "p_value": float(p)}
