import logging
import sys
from datetime import datetime
from functools import lru_cache

import yaml

from sva.constants import CONFIG_DIR, LOGS_DIR


@lru_cache(maxsize=None)
def load_config(name="default"):
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(name):
    """Log ra man hinh va ra file logs/<name>_<timestamp>.log"""
    LOGS_DIR.mkdir(exist_ok=True)
    path = LOGS_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return path
