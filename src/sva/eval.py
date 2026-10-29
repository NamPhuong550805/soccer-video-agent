"""Danh gia: metric (accuracy, khoang tin cay, ECE...), McNemar, chay he thong va doc / ghi ket qua."""
import json
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.stats import binomtest, chi2
from tqdm import tqdm

from sva.config import INDEX_DIR, RESULTS_DIR, load_config
from sva.data import canon, load_clean

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- Metric
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


def ordinal_mae(recs, samples, scale):
    """Sai so tuyet doi trung binh tren thang thu tu (vd no card=0 ... red card=2)."""
    by_index = {s.index: s for s in samples}
    errs = []
    for r in recs:
        s = by_index[r["index"]]
        texts = {k: canon(t) for k, t in s.option_texts.items()}
        if r.get("pred") and all(t in scale for t in texts.values()):
            errs.append(abs(scale[texts[r["pred"]]] - scale[texts[r["gold"]]]))
    return float(np.mean(errs)) if errs else float("nan")


def position_rate(recs):
    """Ty le chon option o vi tri dau tien, so voi ty le dap an dung nam o vi tri dau."""
    first = lambda r, k: k == sorted(r["dist"], key=lambda x: int(x[1:]))[0]
    recs = [r for r in recs if r.get("dist")]
    return (sum(first(r, r["pred"]) for r in recs) / len(recs),
            sum(first(r, r["gold"]) for r in recs) / len(recs))


# ---------------------------------------------------------------- Kiem dinh
def mcnemar(recs_a, recs_b):
    """So 2 he tren cung tap cau. b = A sai B dung, c = A dung B sai; b + c < 25 thi dung kiem dinh nhi thuc chinh xac."""
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


# ---------------------------------------------------------------- Chay he thong, doc ket qua
PREDICTIONS_DIR = RESULTS_DIR / "predictions"


def load_split(task_id, split):
    with open(INDEX_DIR / "splits.json", encoding="utf-8") as f:
        idx = set(json.load(f)[task_id][split])
    return [s for s in load_clean(task_id) if s.index in idx]


def result_name(name, backend, llm=None):
    """Moi backend ghi ra file rieng (tag trong models.yaml, vd _7b), tranh lan ket qua giua cac model."""
    vlm = load_config("models")["vlm"]
    if vlm[backend].get("tag"):
        name = f"{name}_{vlm[backend]['tag']}"
    if llm and llm != backend:
        name = f"{name}_llm-{vlm[llm].get('tag', llm)}"
    return name


def prediction_path(system_name, task_id, split="test"):
    suffix = "" if split == "test" else f"_{split}"
    return PREDICTIONS_DIR / f"{system_name}_{task_id}{suffix}.jsonl"


def load_predictions(system_name, task_id, split="test"):
    path = prediction_path(system_name, task_id, split)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_system(system, task_id, split="test", limit=None, workers=1, max_skips=10):
    """Chay system tren 1 task, ghi tung dong jsonl; chay lai thi bo qua cau da co.
    workers > 1 chi dung cho API. Cau tra ve {"skip": True} (het quota) khong ghi, lan sau lam lai."""
    samples = load_split(task_id, split)[:limit]
    path = prediction_path(system.name, task_id, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["index"] for r in load_predictions(system.name, task_id, split)}
    todo = [s for s in samples if s.index not in done]

    lock = threading.Lock()
    stop = threading.Event()
    skips = {"run": 0, "total": 0}

    def work(s):
        if stop.is_set():
            return
        t0 = time.time()
        try:
            out = system(s)
        except Exception as e:      # 1 mau loi khong lam hong ca lan chay
            out = {"pred": None, "error": f"{type(e).__name__}: {e}"}
        with lock:
            if out.get("skip"):
                skips["run"] += 1
                skips["total"] += 1
                skips["last"] = out.get("reason", "")
                if skips["run"] >= max_skips:
                    stop.set()
                return
            skips["run"] = 0
            rec = {"index": s.index, "gold": s.gold_key, "n_options": s.n_options, **out,
                   "latency_ms": round(1000 * (time.time() - t0))}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    with open(path, "a", encoding="utf-8") as f:
        bar = tqdm(total=len(todo), desc=f"{system.name} {task_id}")
        if workers <= 1:
            for s in todo:
                work(s)
                bar.update()
                if stop.is_set():
                    break
        else:
            with ThreadPoolExecutor(workers) as ex:
                for _ in ex.map(work, todo):
                    bar.update()
        bar.close()

    if skips["total"]:
        log.warning("%s %s: %d mau bi bo qua (het quota?), chay lai lenh de lam tiep", system.name, task_id, skips["total"])
    if stop.is_set():
        raise RuntimeError(f"dung vi {max_skips} mau lien tiep bi bo qua. Loi cuoi: {skips.get('last')}")
    return path
