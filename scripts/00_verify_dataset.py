"""Kiem tra dataset SoccerBench truoc khi lam gi khac (chi doc, khong sua).

    python scripts/00_verify_dataset.py
    python scripts/00_verify_dataset.py --skip-materials
"""
import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sva.config import load_config, setup_logging
from sva.constants import (AUDIT_DIR, EXPECTED_COUNTS, EXPECTED_OPTION_COUNTS, SOCCERBENCH_ROOT,
                           TASK_IDS, TASK_META, TASKS_WITH_LIST_OPENA)
from sva.data.loader import load_raw, parse_option_keys

log = logging.getLogger(__name__)
REQUIRED_KEYS = ["Q", "materials", "openA", "closeA"]


def verify_task(task_id, check_materials=True):
    records = load_raw(task_id)
    n_options = Counter()
    n_clips = Counter()
    open_types = Counter()
    noncontiguous, missing_keys, missing_files, bad_closeA, empty_openA = [], [], [], [], []
    files = set()

    for idx, rec in enumerate(records):
        if any(k not in rec for k in REQUIRED_KEYS):
            missing_keys.append(idx)
            continue

        keys = parse_option_keys(rec)
        n_options[len(keys)] += 1
        if [int(k[1:]) for k in keys] != list(range(1, len(keys) + 1)):
            noncontiguous.append(idx)
        if rec["closeA"] not in keys:
            bad_closeA.append(idx)

        open_types[type(rec["openA"]).__name__] += 1
        if rec["openA"] == "":
            empty_openA.append(idx)

        n_clips[len(rec["materials"])] += 1
        for m in rec["materials"]:
            files.add(m)
            if check_materials and not (SOCCERBENCH_ROOT / m).is_file():
                missing_files.append(m)

    has_list = open_types.get("list", 0) > 0
    checks = {
        "count": len(records) == EXPECTED_COUNTS[task_id],
        "keys": not missing_keys,
        "options": dict(n_options) == EXPECTED_OPTION_COUNTS[task_id],
        "materials": not missing_files if check_materials else None,
        "openA_type": has_list == (task_id in TASKS_WITH_LIST_OPENA),
    }
    return {
        "task_id": task_id,
        "n": len(records),
        "n_options": dict(sorted(n_options.items())),
        "n_clips": dict(sorted(n_clips.items())),
        "openA_types": dict(open_types),
        "n_files": len(files),
        "n_declared": sum(k * v for k, v in n_clips.items()),
        "noncontiguous": noncontiguous,
        "closeA_out_of_range": bad_closeA,
        "openA_empty": empty_openA,
        "missing_files": missing_files[:20],
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=TASK_IDS, choices=TASK_IDS)
    parser.add_argument("--skip-materials", action="store_true")
    args = parser.parse_args()

    setup_logging("00_verify_dataset")
    if not SOCCERBENCH_ROOT.is_dir():
        log.error("khong thay dataset o %s", SOCCERBENCH_ROOT)
        return 1

    results = [verify_task(t, not args.skip_materials) for t in args.tasks]

    log.info("%-5s %-5s %6s  %-20s %-18s %-10s %6s", "task", "paper", "n", "options", "clips", "openA", "files")
    for r in results:
        log.info("%-5s %-5s %6d  %-20s %-18s %-10s %6d", r["task_id"], TASK_META[r["task_id"]]["paper_id"],
                 r["n"], r["n_options"], r["n_clips"], ",".join(r["openA_types"]), r["n_files"])

    log.info("")
    for r in results:
        notes = []
        if r["noncontiguous"]:
            notes.append(f"{len(r['noncontiguous'])} mau option khong lien tuc")
        if r["closeA_out_of_range"]:
            notes.append(f"{len(r['closeA_out_of_range'])} mau closeA ngoai range")
        if r["openA_empty"]:
            notes.append(f"openA rong o index {r['openA_empty']}")
        if notes:
            log.info("%s: %s", r["task_id"], "; ".join(notes))

    # configs/tasks.yaml phai khop voi TASK_META
    tasks_cfg = load_config("tasks")["tasks"]
    config_ok = all(tasks_cfg[t][k] == TASK_META[t][k] for t in TASK_IDS for k in TASK_META[t])

    failed = [(r["task_id"], name) for r in results for name, ok in r["checks"].items() if ok is False]
    if not config_ok:
        failed.append(("configs", "tasks.yaml"))
    skipped = args.skip_materials
    passed = not failed and not skipped

    log.info("")
    if failed:
        log.error("FAIL: %s", failed)
    elif skipped:
        log.info("cac check khac OK, chua kiem tra file video (--skip-materials)")
    else:
        log.info("OK: %d task, du lieu khop voi mo ta", len(results))

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out = AUDIT_DIR / "verify_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"passed": passed, "tasks": {r["task_id"]: r for r in results}}, f, indent=2, ensure_ascii=False)
    log.info("da ghi %s", out)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
