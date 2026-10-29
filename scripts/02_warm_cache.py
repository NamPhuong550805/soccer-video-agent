"""Trich frame (va embedding neu co --encoder) cho toan bo video, luu vao data/cache/.

    python scripts/02_warm_cache.py --task q8
    python scripts/02_warm_cache.py --task all --encoder clip_vit_l
    python scripts/02_warm_cache.py --task q14 --limit 20
    python scripts/02_warm_cache.py --task q10 --unisoccer     # dau ra Action Classifier cho B1
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sva.config import TASK_IDS, load_config, record_run, setup_logging
from sva.data import load_clean
from sva.perception import PerceptionCache

log = logging.getLogger(__name__)


def main():
    cfg = load_config()["perception"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", nargs="+", default=["all"])
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--n-frames", type=int, default=cfg["n_frames_default"])
    parser.add_argument("--image-size", type=int, default=cfg["image_size"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--unisoccer", action="store_true", help="chay Action Classifier cua UniSoccer")
    args = parser.parse_args()

    setup_logging("02_warm_cache")
    task_ids = TASK_IDS if args.task == ["all"] else args.task

    encoder = None
    if args.encoder:
        import torch
        from sva.perception import build_encoder
        encoder = build_encoder(args.encoder, "cuda" if torch.cuda.is_available() else "cpu")

    classifier = None
    if args.unisoccer:
        from sva.models import ActionClassifier
        classifier = ActionClassifier()

    cache = PerceptionCache()
    n_declared, files, n_errors = 0, set(), 0
    start = time.time()

    for task_id in task_ids:
        samples = load_clean(task_id)[:args.limit]
        t0 = time.time()
        for s in samples:
            for path in s.materials:
                n_declared += 1
                files.add(path)
                try:
                    if classifier:
                        classifier.probs(path)
                    elif encoder:
                        cache.get_embedding(path, args.n_frames, args.image_size, encoder)
                    else:
                        cache.get_frames(path, args.n_frames, args.image_size)
                except OSError as e:
                    n_errors += 1
                    log.warning("loi: %s", e)
        log.info("%-4s %4d mau  %.1fs", task_id, len(samples), time.time() - t0)

    log.info("")
    log.info("%d luot khai bao / %d file (trung %d)", n_declared, len(files), n_declared - len(files))
    log.info("cache hit=%d miss=%d, %d loi, %.1fs", cache.hits, cache.misses, n_errors, time.time() - start)
    record_run(f"{' '.join(task_ids)}: {len(files)} file, cache hit={cache.hits} miss={cache.misses}, {n_errors} loi")
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
