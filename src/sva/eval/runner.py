import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from sva.constants import INDEX_DIR, RESULTS_DIR
from sva.data.clean import load_clean

PREDICTIONS_DIR = RESULTS_DIR / "predictions"
log = logging.getLogger(__name__)


def load_split(task_id, split):
    with open(INDEX_DIR / "splits.json", encoding="utf-8") as f:
        idx = set(json.load(f)[task_id][split])
    return [s for s in load_clean(task_id) if s.index in idx]


def result_name(name, backend, llm=None):
    """Moi backend ghi ra file rieng (tag trong models.yaml, vd _7b), tranh lan ket qua giua cac model."""
    from sva.config import load_config
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
    """Chay system tren 1 task, ghi tung dong jsonl. Chay lai thi bo qua mau da co (chay tiep duoc).

    workers > 1: chay song song (dung cho API, khong dung cho model tren GPU).
    Mau tra ve {"skip": True} (vd het quota) khong duoc ghi, de lan chay sau lam lai;
    gap max_skips mau skip lien tiep thi dung.
    """
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
