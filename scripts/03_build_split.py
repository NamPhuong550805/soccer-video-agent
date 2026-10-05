"""Chia dev/test theo nhom noi dung -> data/processed/index/"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sva.config import load_config, setup_logging
from sva.constants import TASK_IDS
from sva.data.clean import load_clean
from sva.data.split import build_index, build_splits, verify_no_leakage, write_index_files

log = logging.getLogger(__name__)


def main():
    cfg = load_config()["split"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-ratio", type=float, default=cfg["dev_ratio"])
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    args = parser.parse_args()

    setup_logging("03_build_split")
    splits = build_splits(args.dev_ratio, args.seed)

    ok = True
    for task_id in TASK_IDS:
        samples = load_clean(task_id)
        split = splits[task_id]
        leak_free = verify_no_leakage(samples, split)
        ok = ok and leak_free
        log.info("%-4s %4d nhom  dev=%4d  test=%4d%s", task_id, len(build_index(samples)),
                 len(split["dev"]), len(split["test"]), "" if leak_free else "  <-- RO RI")

    for path in write_index_files(splits):
        log.info("da ghi %s", path)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
