"""Phan loai loi annotation cua 6 task -> data/processed/audit/audit_report.{json,md}"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sva.config import setup_logging
from sva.constants import AUDIT_DIR
from sva.data.audit import CATEGORIES, audit_all, write_audit_reports

log = logging.getLogger(__name__)


def main():
    setup_logging("01_audit")
    reports = audit_all()

    log.info("%-5s %5s  %s", "task", "n", "  ".join(f"{c:>16}" for c in CATEGORIES))
    for r in reports:
        log.info("%-5s %5d  %s", r.task_id, r.n_total,
                 "  ".join(f"{r.counts.get(c, 0):>16}" for c in CATEGORIES))

    for path in write_audit_reports(reports, AUDIT_DIR):
        log.info("da ghi %s", path)


if __name__ == "__main__":
    main()
