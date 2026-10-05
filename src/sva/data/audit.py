import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sva.constants import TASK_IDS, TASK_META
from sva.data.loader import canon, load_raw, parse_option_keys

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
