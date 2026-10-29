"""Chuan bi du lieu SoccerBench: kiem tra -> kiem toan nhan -> lam sach -> chia dev / test.

    python scripts/01_data.py verify [--skip-materials]   # chi doc, -> data/processed/audit/verify_report.json
    python scripts/01_data.py audit                       # phan loai loi nhan -> data/processed/audit/
    python scripts/01_data.py clean                       # -> data/processed/clean/{task}_clean.json
    python scripts/01_data.py split                       # chia theo nhom noi dung -> data/processed/index/
    python scripts/01_data.py all
"""
import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sva.config import (AUDIT_DIR, EXPECTED_COUNTS, EXPECTED_COUNTS_CLEAN, EXPECTED_OPTION_COUNTS, SOCCERBENCH_ROOT,
                        TASK_IDS, TASK_META, TASKS_WITH_LIST_OPENA, load_config, setup_logging)
from sva.data import (CATEGORIES, audit_all, build_index, build_splits, clean_task, load_clean, load_raw,
                      parse_option_keys, verify_no_leakage, write_audit_reports, write_clean_task, write_index_files)

log = logging.getLogger(__name__)
REQUIRED_KEYS = ["Q", "materials", "openA", "closeA"]


# ---------------------------------------------------------------- verify: du lieu co dung nhu mo ta khong
def verify_task(task_id, check_materials=True):
    records = load_raw(task_id)
    n_options, n_clips, open_types = Counter(), Counter(), Counter()
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

    checks = {"count": len(records) == EXPECTED_COUNTS[task_id], "keys": not missing_keys,
              "options": dict(n_options) == EXPECTED_OPTION_COUNTS[task_id],
              "materials": not missing_files if check_materials else None,
              "openA_type": (open_types.get("list", 0) > 0) == (task_id in TASKS_WITH_LIST_OPENA)}
    return {"task_id": task_id, "n": len(records), "n_options": dict(sorted(n_options.items())),
            "n_clips": dict(sorted(n_clips.items())), "openA_types": dict(open_types), "n_files": len(files),
            "n_declared": sum(k * v for k, v in n_clips.items()), "noncontiguous": noncontiguous,
            "closeA_out_of_range": bad_closeA, "openA_empty": empty_openA, "missing_files": missing_files[:20],
            "checks": checks}


def verify(skip_materials=False):
    if not SOCCERBENCH_ROOT.is_dir():
        log.error("khong thay dataset o %s", SOCCERBENCH_ROOT)
        return False
    results = [verify_task(t, not skip_materials) for t in TASK_IDS]
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

    failed = [(r["task_id"], name) for r in results for name, ok in r["checks"].items() if ok is False]
    log.info("")
    if failed:
        log.error("FAIL: %s", failed)
    elif skip_materials:
        log.info("cac check khac OK, chua kiem tra file video (--skip-materials)")
    else:
        log.info("OK: %d task, du lieu khop voi mo ta", len(results))

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out = AUDIT_DIR / "verify_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"passed": not failed and not skip_materials, "tasks": {r["task_id"]: r for r in results}}, f,
                  indent=2, ensure_ascii=False)
    log.info("da ghi %s", out)
    return not failed


# ---------------------------------------------------------------- audit: phan loai loi nhan
def audit():
    reports = audit_all()
    log.info("%-5s %5s  %s", "task", "n", "  ".join(f"{c:>16}" for c in CATEGORIES))
    for r in reports:
        log.info("%-5s %5d  %s", r.task_id, r.n_total, "  ".join(f"{r.counts.get(c, 0):>16}" for c in CATEGORIES))
    for path in write_audit_reports(reports, AUDIT_DIR):
        log.info("da ghi %s", path)
    return True


# ---------------------------------------------------------------- clean: sua / bo mau loi
def clean():
    ok = True
    for task_id in TASK_IDS:
        samples = clean_task(task_id)
        write_clean_task(task_id, samples)
        n_noted = sum(1 for s in samples if s.cleaning_note)
        match = len(samples) == EXPECTED_COUNTS_CLEAN[task_id]
        ok = ok and match
        log.info("%-4s %4d -> %4d  (%d mau co ghi chu)%s", task_id, len(load_raw(task_id)), len(samples), n_noted,
                 "" if match else "  <-- SAI SO LUONG")
    return ok


# ---------------------------------------------------------------- split: chia dev / test theo nhom
def split(dev_ratio, seed):
    splits = build_splits(dev_ratio, seed)
    ok = True
    for task_id in TASK_IDS:
        samples, sp = load_clean(task_id), splits[task_id]
        leak_free = verify_no_leakage(samples, sp)
        ok = ok and leak_free
        log.info("%-4s %4d nhom  dev=%4d  test=%4d%s", task_id, len(build_index(samples)), len(sp["dev"]),
                 len(sp["test"]), "" if leak_free else "  <-- RO RI")
    for path in write_index_files(splits):
        log.info("da ghi %s", path)
    return ok


def main():
    cfg = load_config()["split"]
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=["verify", "audit", "clean", "split", "all"])
    parser.add_argument("--skip-materials", action="store_true", help="verify: khong kiem tra file video")
    parser.add_argument("--dev-ratio", type=float, default=cfg["dev_ratio"])
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    args = parser.parse_args()
    setup_logging(f"01_data_{args.part}")
    steps = {"verify": lambda: verify(args.skip_materials), "audit": audit, "clean": clean,
             "split": lambda: split(args.dev_ratio, args.seed)}
    for name in (list(steps) if args.part == "all" else [args.part]):
        log.info("---------------- %s", name)
        if not steps[name]():
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
