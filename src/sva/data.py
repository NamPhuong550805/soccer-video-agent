"""Du lieu SoccerBench: Sample, doc JSON goc, kiem toan nhan, lam sach, chia dev / test."""
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from sva.config import (CLEAN_DIR, INDEX_DIR, LABEL_ALIASES, OPTION_KEY_PATTERN, Q13_CLOSEA_FIX,
                        Q13_OPENA_EMPTY_INDICES, Q14_CLOSEA_FIX, Q14_DROP_INDICES, QA_DIR, SOCCERBENCH_ROOT, TASK_IDS,
                        TASK_META)


# ---------------------------------------------------------------- Sample
@dataclass
class Sample:
    task_id: str
    index: int                     # index trong file json goc
    question: str
    materials: list[str]           # duong dan tuyet doi
    option_keys: list[str]         # co the khong lien tuc, vd ["O1", "O3"]
    option_texts: dict[str, str]
    gold_key: Optional[str]        # None khi hoi that (ung dung), khong co dap an
    open_answer: Optional[str] = None
    cleaning_note: Optional[str] = None   # "closeA_fixed" | "openA_empty"
    meta: dict = field(default_factory=dict)

    @property
    def n_options(self):
        return len(self.option_keys)

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, d):
        return cls(**d)


# ---------------------------------------------------------------- Doc JSON goc
VIEW_PATTERN = re.compile(r"([A-Za-z][A-Za-z \-]*?) lens at ([\d.]+)x speed")


def load_raw(task_id):
    with open(QA_DIR / f"{task_id}.json", encoding="utf-8") as f:
        return json.load(f)


def parse_option_keys(record):
    # q14 co mau chi co O1, O3 nen khong duoc gia dinh O1..O4
    keys = [k for k in record if re.fullmatch(OPTION_KEY_PATTERN, k)]
    return sorted(keys, key=lambda k: int(k[1:]))


def canon(label):
    s = " ".join(label.strip().lower().rstrip(".").split())
    return LABEL_ALIASES.get(s, s)


def parse_views(question):
    """Tach (camera, speed) tu phan dau cau hoi q14."""
    return [{"camera": cam.strip(), "speed": float(speed)}
            for cam, speed in VIEW_PATTERN.findall(question)]


# ---------------------------------------------------------------- Kiem toan nhan
OK = "OK"
OK_NONCONTIGUOUS = "OK_NONCONTIGUOUS"
RECOVERABLE = "RECOVERABLE"
UNRECOVERABLE = "UNRECOVERABLE"
INDEX_ERROR = "INDEX_ERROR"
OPENA_EMPTY = "OPENA_EMPTY"
OPENA_IS_LIST = "OPENA_IS_LIST"
NO_MATCH = "NO_MATCH"   # closeA hop le nhung openA khong khop option nao (chua gap)

CATEGORIES = [OK, OK_NONCONTIGUOUS, RECOVERABLE, UNRECOVERABLE,
              INDEX_ERROR, OPENA_EMPTY, OPENA_IS_LIST, NO_MATCH]


@dataclass
class AuditRecord:
    index: int
    category: str
    closeA: str
    fixed: str = None
    note: str = ""


@dataclass
class AuditReport:
    task_id: str
    n_total: int
    counts: dict = field(default_factory=dict)
    records: list = field(default_factory=list)   # chi luu cac mau khong phai OK
    drop_indices: list = field(default_factory=list)
    fix_map: dict = field(default_factory=dict)

    def by_category(self, cat):
        return [r for r in self.records if r.category == cat]


def q9_gold_key(rec):
    # openA cua q9 la duong dan tren may tac gia, phai so ten file voi 4 ung vien
    gold = Path(rec["openA"][0]).name
    candidates = [Path(m).name for m in rec["materials"][1:5]]
    return f"O{candidates.index(gold) + 1}"


def classify(rec):
    """Tra ve (category, closeA sau khi sua hoac None)."""
    keys = parse_option_keys(rec)
    close_a, open_a = rec["closeA"], rec["openA"]

    if isinstance(open_a, list):
        return (OPENA_IS_LIST, None) if q9_gold_key(rec) == close_a else (NO_MATCH, None)
    if open_a.strip() == "":
        return OPENA_EMPTY, None

    matched = [k for k in keys if canon(rec[k]) == canon(open_a)]

    if close_a in keys:
        if close_a in matched:
            contiguous = [int(k[1:]) for k in keys] == list(range(1, len(keys) + 1))
            return (OK if contiguous else OK_NONCONTIGUOUS), None
        return (INDEX_ERROR, matched[0]) if matched else (NO_MATCH, None)

    # closeA tro toi option khong ton tai
    return (RECOVERABLE, matched[0]) if matched else (UNRECOVERABLE, None)


def audit_task(task_id):
    raw = load_raw(task_id)
    report = AuditReport(task_id, len(raw))
    counts = Counter()

    for idx, rec in enumerate(raw):
        cat, fixed = classify(rec)
        counts[cat] += 1
        if cat in (OK, OPENA_IS_LIST):
            continue
        options = {k: rec[k] for k in parse_option_keys(rec)}
        note = f"openA={rec['openA']!r}, options={options}"
        report.records.append(AuditRecord(idx, cat, rec["closeA"], fixed, note))
        if fixed:
            report.fix_map[idx] = fixed
        if cat == UNRECOVERABLE:
            report.drop_indices.append(idx)

    report.counts = dict(counts)
    return report


def audit_all():
    return [audit_task(t) for t in TASK_IDS]


def write_audit_reports(reports, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "audit_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({r.task_id: asdict(r) for r in reports}, f, indent=2, ensure_ascii=False)

    lines = ["# Audit report", "",
             "| task | paper | n | " + " | ".join(CATEGORIES) + " |",
             "|---|---|---|" + "---|" * len(CATEGORIES)]
    for r in reports:
        cells = [r.task_id, TASK_META[r.task_id]["paper_id"], str(r.n_total)]
        cells += [str(r.counts.get(c, 0)) for c in CATEGORIES]
        lines.append("| " + " | ".join(cells) + " |")

    for r in reports:
        if not r.records:
            continue
        lines += ["", f"## {r.task_id}", "",
                  "| index | category | closeA | fixed | note |", "|---|---|---|---|---|"]
        for rec in r.records:
            lines.append(f"| {rec.index} | {rec.category} | {rec.closeA} | {rec.fixed or '-'} | {rec.note} |")

    md_path = out_dir / "audit_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


# ---------------------------------------------------------------- Lam sach
MATCH_NAME_PATTERN = r"(\d{4}-\d{2}-\d{2}[ _-][^/]*)"


class DataDriftError(RuntimeError):
    pass


def check_against_constants(report):
    """Ket qua audit phai trung voi danh sach loi da ghi trong config.py."""
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


# ---------------------------------------------------------------- Chia dev / test
def group_key(sample):
    """Don vi chia split: q14 theo action_id, q9 theo tran, con lai theo file video."""
    if sample.task_id == "q14":
        return sample.meta["action_id"]
    if sample.task_id == "q9":
        return sample.meta["match_name"]
    return "|".join(sorted(sample.materials))


def build_index(samples):
    index = defaultdict(list)
    for s in samples:
        index[group_key(s)].append(s.index)
    return dict(index)


def build_splits(dev_ratio=0.3, seed=42, clean_dir=CLEAN_DIR, task_ids=None):
    rng = random.Random(seed)
    splits = {}
    for task_id in task_ids or TASK_IDS:
        samples = load_clean(task_id, clean_dir)
        groups = build_index(samples)
        keys = list(groups)
        rng.shuffle(keys)

        # ca nhom vao cung mot phia nen khong bi ro ri
        n_dev = round(dev_ratio * len(samples))
        dev, test = [], []
        for k in keys:
            (dev if len(dev) < n_dev else test).extend(groups[k])
        splits[task_id] = {"dev": sorted(dev), "test": sorted(test)}
    return splits


def verify_no_leakage(samples, split):
    by_index = {s.index: s for s in samples}
    dev = {group_key(by_index[i]) for i in split["dev"]}
    test = {group_key(by_index[i]) for i in split["test"]}
    return dev.isdisjoint(test)


def write_index_files(splits, out_dir=INDEX_DIR, clean_dir=CLEAN_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "splits.json": splits,
        "action_index.json": build_index(load_clean("q14", clean_dir)),
        "match_index.json": build_index(load_clean("q9", clean_dir)),
    }
    for name, data in files.items():
        with open(out_dir / name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    return [out_dir / name for name in files]
