import json
import random
from collections import defaultdict

from sva.constants import CLEAN_DIR, INDEX_DIR, TASK_IDS
from sva.data.clean import load_clean


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
