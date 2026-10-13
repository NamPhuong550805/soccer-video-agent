import json
import re
from pathlib import Path

from sva.constants import (CLEAN_DIR, Q13_CLOSEA_FIX, Q13_OPENA_EMPTY_INDICES, Q14_CLOSEA_FIX,
                           Q14_DROP_INDICES, SOCCERBENCH_ROOT)
from sva.data.audit import NO_MATCH, OPENA_EMPTY, audit_task
from sva.data.loader import load_raw, parse_option_keys, parse_views
from sva.schema import Sample

MATCH_NAME_PATTERN = r"(\d{4}-\d{2}-\d{2}[ _-][^/]*)"


class DataDriftError(RuntimeError):
    pass


def check_against_constants(report):
    """Ket qua audit phai trung voi danh sach loi da ghi trong constants.py."""
    t = report.task_id
    expected_drop = Q14_DROP_INDICES if t == "q14" else []
    expected_fix = {"q14": Q14_CLOSEA_FIX, "q13": Q13_CLOSEA_FIX}.get(t, {})
    expected_empty = Q13_OPENA_EMPTY_INDICES if t == "q13" else []
    empty = [r.index for r in report.by_category(OPENA_EMPTY)]

    if report.by_category(NO_MATCH):
        raise DataDriftError(f"{t}: co mau khong phan loai duoc: {report.by_category(NO_MATCH)[:3]}")
    if sorted(report.drop_indices) != sorted(expected_drop):
        raise DataDriftError(f"{t}: drop {report.drop_indices} != {expected_drop}")
    if report.fix_map != expected_fix:
        raise DataDriftError(f"{t}: fix {report.fix_map} != {expected_fix}")
    if sorted(empty) != sorted(expected_empty):
        raise DataDriftError(f"{t}: openA rong {empty} != {expected_empty}")


def extract_meta(task_id, rec):
    first = rec["materials"][0]
    if task_id == "q9":
        m = re.search(MATCH_NAME_PATTERN, first)
        return {"match_name": m.group(1) if m else None}
    if task_id == "q14":
        m = re.search(r"(action_\d+)", first)
        return {"action_id": m.group(1) if m else None, "views": parse_views(rec["Q"])}
    return {}


def clean_task(task_id):
    raw = load_raw(task_id)
    report = audit_task(task_id)
    check_against_constants(report)

    empty = {r.index for r in report.by_category(OPENA_EMPTY)}
    samples = []
    for idx, rec in enumerate(raw):
        if idx in report.drop_indices:
            continue

        keys = parse_option_keys(rec)
        meta = {}
        note = None
        if idx in report.fix_map:
            note = "closeA_fixed"
        elif idx in empty:
            note = "openA_empty"
            meta["low_confidence"] = True

        open_answer = rec["openA"]
        if isinstance(open_answer, list):   # q9
            meta["open_answer_raw"] = open_answer
            open_answer = Path(open_answer[0]).name

        meta.update(extract_meta(task_id, rec))

        samples.append(Sample(
            task_id=task_id,
            index=idx,
            question=rec["Q"],
            materials=[str((SOCCERBENCH_ROOT / m).resolve()) for m in rec["materials"]],
            option_keys=keys,
            option_texts={k: rec[k] for k in keys},
            gold_key=report.fix_map.get(idx, rec["closeA"]),
            open_answer=open_answer,
            cleaning_note=note,
            meta=meta,
        ))

    assert all(s.gold_key in s.option_keys for s in samples)
    return samples


def write_clean_task(task_id, samples, out_dir=CLEAN_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task_id}_clean.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.to_json() for s in samples], f, indent=2, ensure_ascii=False)
    return path


def local_path(path):
    """File sach luu duong dan tuyet doi cua may da tao ra no; ghep lai theo SoccerBench cua may dang chay."""
    parts = Path(str(path).replace("\\", "/")).parts
    if "materials" not in parts:
        return str(path)
    return str((SOCCERBENCH_ROOT / Path(*parts[parts.index("materials"):])).resolve())


def load_clean(task_id, clean_dir=CLEAN_DIR):
    with open(clean_dir / f"{task_id}_clean.json", encoding="utf-8") as f:
        samples = [Sample.from_json(d) for d in json.load(f)]
    for s in samples:
        s.materials = [local_path(p) for p in s.materials]
    return samples
