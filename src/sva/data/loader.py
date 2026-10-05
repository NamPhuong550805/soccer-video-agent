import json
import re

from sva.constants import LABEL_ALIASES, OPTION_KEY_PATTERN, QA_DIR

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
