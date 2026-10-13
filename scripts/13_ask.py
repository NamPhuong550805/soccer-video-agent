"""Hoi SVA-Agent ve video that (cau hoi tieng Anh, co hoac khong co phuong an).

    python scripts/13_ask.py --video highlight.mp4 --question "What happens at 1:10?"
    python scripts/13_ask.py --video highlight.mp4 --question "Which earlier play is shown again in the replay at 2:40?"
    python scripts/13_ask.py --video main.mp4 closeup.mp4 --question "Did the foul involve a handball?" --options No Yes

Ket qua (cau tra loi, do tin, doan video, bang chung, frame) ghi vao results/agent/ask/<thoi diem>/.
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sva.config import record_run, setup_logging
from sva.constants import RESULTS_DIR

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", nargs="+", required=True, help="1 video (co the dai) hoac 2-5 video")
    parser.add_argument("--question", required=True)
    parser.add_argument("--options", nargs="*", default=None, help="2-4 phuong an; bo trong = che do Open")
    parser.add_argument("--backend", default="qwen_local_3b")
    args = parser.parse_args()
    setup_logging("13_ask")
    from sva.agent.app import SVAAgent
    out_dir = RESULTS_DIR / "agent" / "ask" / f"{datetime.now():%Y%m%d_%H%M%S}"
    options = {f"O{i + 1}": t for i, t in enumerate(args.options)} if args.options else None
    r = SVAAgent(args.backend).ask(args.video, args.question, options, out_dir)
    log.info("\n%s\n", r["answer"])
    log.info("trang thai %s | task %s (%s) | do tin %s | doan %s", r["status"], r["task"], r["route"],
             f"{r['confidence']:.2f}" if "confidence" in r else "-", r.get("segments"))
    for e in r.get("evidence", []):
        log.info("  %-18s %s", e["tool_name"], {k: round(v, 2) for k, v in e["dist"].items()})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "answer.json").write_text(json.dumps({"question": args.question, "videos": args.video, **r},
                                                    ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    record_run(f"{r['status']}: {r['answer'][:120]}")


if __name__ == "__main__":
    main()
