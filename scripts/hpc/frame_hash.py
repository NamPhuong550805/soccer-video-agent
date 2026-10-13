"""Kiem tra frame tren server giong het frame da dung tren may local (B0-api da xem dung cac frame nay).

    python scripts/hpc/frame_hash.py --out logs/frame_hashes.json      # may local: doc tu cache
    python scripts/hpc/frame_hash.py --check logs/frame_hashes.json    # server: trich lai tu video, so hash
"""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sva.constants import TASK_IDS
from sva.eval.runner import load_split
from sva.perception.cache import PerceptionCache, rel_key


def hashes(cache, per_task):
    out = {}
    for t in TASK_IDS:
        for s in load_split(t, "test")[:per_task]:
            frames, _, timestamps = cache.get_clips(s.materials, 8, 24, 336)
            for p, f, ts in zip(s.materials, frames, timestamps):
                out[f"{rel_key(p)}|{len(ts)}"] = hashlib.sha1(f.tobytes()).hexdigest()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--check")
    parser.add_argument("--per-task", type=int, default=5)
    args = parser.parse_args()

    if args.out:
        h = hashes(PerceptionCache(), args.per_task)
        Path(args.out).write_text(json.dumps(h, indent=1), encoding="utf-8")
        print(f"da ghi {len(h)} hash -> {args.out}")
        return 0

    ref = json.loads(Path(args.check).read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as d:     # cache trong: bat buoc trich lai tu video
        h = hashes(PerceptionCache(Path(d) / "f", Path(d) / "e"), args.per_task)
    diff = [k for k in ref if h.get(k) != ref[k]]
    print(f"{len(ref) - len(diff)}/{len(ref)} clip giong het")
    for k in diff[:10]:
        print("  KHAC:", k)
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())
