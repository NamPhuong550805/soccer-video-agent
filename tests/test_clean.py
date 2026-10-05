import pytest

from sva.constants import (CLEAN_DIR, Q13_CLOSEA_FIX, Q13_OPENA_EMPTY_INDICES, Q14_CLOSEA_FIX,
                           Q14_DROP_INDICES, TASK_IDS)
from sva.data.clean import load_clean
from sva.data.loader import canon, parse_option_keys, parse_views
from sva.data.split import build_splits, verify_no_leakage

# can chay scripts/02_clean.py truoc
pytestmark = pytest.mark.skipif(not (CLEAN_DIR / "q14_clean.json").is_file(), reason="chua co data clean")


def test_parse_option_keys():
    assert parse_option_keys({"Q": "?", "O1": "no", "O3": "yes", "closeA": "O3"}) == ["O1", "O3"]
    assert parse_option_keys({"O1": "a", "O10": "b", "O2": "c"}) == ["O1", "O2", "O10"]


def test_canon():
    assert canon("St. tackling") == canon("Standing tackling")
    assert canon("  Yellow Card.  ") == "yellow card"


def test_parse_views():
    q = ("There are 2 video replays of the same foul, including: Main camera center lens at 1.0x speed, "
         "Close-up player or field referee lens at 1.8x speed. What category does this offence fall under?")
    assert parse_views(q) == [{"camera": "Main camera center", "speed": 1.0},
                              {"camera": "Close-up player or field referee", "speed": 1.8}]


def test_clean_counts():
    assert len(load_clean("q14")) == 276
    assert len(load_clean("q13")) == 700


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_gold_key_in_options(task_id):
    for s in load_clean(task_id):
        assert s.gold_key in s.option_keys


def test_q14_views_match_clips():
    for s in load_clean("q14"):
        assert len(s.meta["views"]) == len(s.materials)


def test_q14_drop_and_fix():
    samples = {s.index: s for s in load_clean("q14")}
    assert not set(Q14_DROP_INDICES) & set(samples)
    for idx, key in Q14_CLOSEA_FIX.items():
        assert samples[idx].gold_key == key
        assert samples[idx].cleaning_note == "closeA_fixed"


def test_q13_fix_and_flag():
    samples = {s.index: s for s in load_clean("q13")}
    for idx, key in Q13_CLOSEA_FIX.items():
        assert samples[idx].gold_key == key
    for idx in Q13_OPENA_EMPTY_INDICES:
        assert samples[idx].cleaning_note == "openA_empty"
        assert samples[idx].meta["low_confidence"]


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_split(task_id):
    samples = load_clean(task_id)
    split = build_splits(task_ids=[task_id])[task_id]
    dev, test = set(split["dev"]), set(split["test"])
    assert not dev & test
    assert dev | test == {s.index for s in samples}
    assert verify_no_leakage(samples, split)
