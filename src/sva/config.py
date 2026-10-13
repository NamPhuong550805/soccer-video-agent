import logging
import subprocess
import sys
from datetime import datetime
from functools import lru_cache

import yaml

from sva.constants import CONFIG_DIR, LOGS_DIR, PROJECT_ROOT, RESULTS_DIR

RUNS_FILE = RESULTS_DIR / "RUNS.md"
_run = {}


@lru_cache(maxsize=None)
def load_config(name="default"):
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def git_version():
    """Commit hien tai; them * neu code (src, scripts, configs) dang sua chua commit."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True)
    try:
        head = git("rev-parse", "--short", "HEAD").stdout.strip()
    except OSError:
        head = ""
    if not head:                          # server: khong co git, doc phien ban ghi luc dong goi code
        f = PROJECT_ROOT / "VERSION"
        return f.read_text(encoding="utf-8").strip() if f.exists() else "?"
    dirty = git("diff", "--quiet", "HEAD", "--", "src", "scripts", "configs").returncode != 0
    return head + ("*" if dirty else "")


def setup_logging(name):
    """Log chi tiet ra man hinh va logs/<name>_<timestamp>.log (khong dua len git)."""
    LOGS_DIR.mkdir(exist_ok=True)
    path = LOGS_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _run.update(path=path, start=datetime.now(), cmd=" ".join(["python", *sys.argv]), version=git_version())
    logging.getLogger(__name__).info("$ %s  [commit %s, %s]", _run["cmd"], _run["version"], f"{_run['start']:%Y-%m-%d %H:%M}")
    return path


def record_run(summary):
    """Them 1 dong vao results/RUNS.md: so tay thi nghiem gon (lenh, commit, ket qua, thoi gian, file log)."""
    if not RUNS_FILE.exists():
        RUNS_FILE.write_text("# Sổ tay chạy thí nghiệm\n\nMỗi dòng là một lần chạy script, do `record_run` tự ghi. "
                             "Log chi tiết nằm ở `logs/` (không đưa lên git). Commit có `*` là lúc chạy code đang sửa "
                             "chưa commit.\n\n| bắt đầu | lệnh | commit | kết quả | thời gian | log |\n"
                             "|---|---|---|---|---|---|\n", encoding="utf-8")
    minutes = (datetime.now() - _run["start"]).total_seconds() / 60
    row = (f"| {_run['start']:%Y-%m-%d %H:%M} | `{_run['cmd']}` | {_run['version']} | {summary.replace('|', '/')} | "
           f"{minutes:.0f} phút | {_run['path'].name} |\n")
    with open(RUNS_FILE, "a", encoding="utf-8") as f:
        f.write(row)
