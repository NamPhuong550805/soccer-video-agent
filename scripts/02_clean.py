"""Lam sach du lieu -> data/processed/clean/{task}_clean.json"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sva.config import setup_logging
from sva.constants import EXPECTED_COUNTS_CLEAN, TASK_IDS
from sva.data.clean import clean_task, write_clean_task
from sva.data.loader import load_raw

log = logging.getLogger(__name__)


def main():
    setup_logging("02_clean")
    ok = True
    for task_id in TASK_IDS:
        samples = clean_task(task_id)
        write_clean_task(task_id, samples)

        n_raw = len(load_raw(task_id))
        n_noted = sum(1 for s in samples if s.cleaning_note)
        match = len(samples) == EXPECTED_COUNTS_CLEAN[task_id]
        ok = ok and match
        log.info("%-4s %4d -> %4d  (%d mau co ghi chu)%s", task_id, n_raw, len(samples), n_noted,
                 "" if match else "  <-- SAI SO LUONG")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
