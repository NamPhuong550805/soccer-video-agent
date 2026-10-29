import numpy as np
import pytest

from sva.config import (CLEAN_DIR, Q13_CLOSEA_FIX, Q13_OPENA_EMPTY_INDICES, Q14_CLOSEA_FIX, Q14_DROP_INDICES,
                        SOCCERBENCH_ROOT, TASK_IDS)
from sva.data import build_splits, canon, load_clean, parse_option_keys, parse_views, verify_no_leakage
from sva.perception import PerceptionCache, cache_key, letterbox, probe, sample_frames, timestamps_for


# ---------------------------------------------------------------- Doc JSON, lam sach, chia dev / test
# can chay scripts/01_data.py all truoc
pytestmark = pytest.mark.skipif(not SOCCERBENCH_ROOT.is_dir() or not (CLEAN_DIR / "q14_clean.json").is_file(),
                                reason="chua co dataset / data clean")


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


# ---------------------------------------------------------------- Doc video, frame, cache
Q8 = SOCCERBENCH_ROOT / "materials" / "q8"
CLIP_25FPS = Q8 / "england_epl/2015-2016/2015-11-29 - 15-00 Tottenham 0 - 0 Chelsea/1_2656.mp4"
CLIP_18FPS = Q8 / "italy_serie-a/2016-2017/2016-08-28 - 21-45 Cagliari 2 - 2 AS Roma/1_887.mp4"


def test_timestamps_for():
    assert timestamps_for(4, 8.0) == pytest.approx([1.0, 3.0, 5.0, 7.0])


def test_same_duration_different_fps():
    # 2 clip cung dai ~6.72s nhung fps 25 va 18 -> timestamp phai gan nhu trung nhau
    a, b = probe(CLIP_25FPS), probe(CLIP_18FPS)
    assert abs(a.fps - b.fps) > 5
    assert abs(a.duration - b.duration) < 0.05

    _, ts_a = sample_frames(CLIP_25FPS, 8)
    _, ts_b = sample_frames(CLIP_18FPS, 8)
    assert max(abs(x - y) for x, y in zip(ts_a, ts_b)) < 0.05


@pytest.mark.parametrize("shape", [(224, 398, 3), (398, 224, 3)])
def test_letterbox_keeps_ratio(shape):
    img = np.full(shape, 200, dtype=np.uint8)
    out = letterbox(img, 336)
    assert out.shape == (336, 336, 3)

    rows = np.where(out.any(axis=(1, 2)))[0]
    cols = np.where(out.any(axis=(0, 2)))[0]
    h, w = len(rows), len(cols)
    assert max(h, w) == 336
    assert min(h, w) == round(224 * 336 / 398)


def test_squeezed_clip_is_restored():
    # toan bo clip q13 bi luu 224x398
    path = load_clean("q13")[0].materials[0]
    m = probe(path)
    assert (m.width, m.height) == (224, 398)
    frames, _ = sample_frames(path, 2)
    assert frames.shape == (2, 224, 398, 3)


def test_cache_key():
    base = cache_key("a/b.mp4", 8, 336)
    assert base == cache_key("a/b.mp4", 8, 336)
    assert base != cache_key("a/b.mp4", 12, 336)
    assert base != cache_key("a/b.mp4", 8, 224)
    assert base != cache_key("a/b.mp4", 8, 336, "clip_vit_l")


def test_cache_hit(tmp_path):
    path = load_clean("q8")[0].materials[0]
    cache = PerceptionCache(tmp_path / "frames", tmp_path / "emb")

    frames1, ts1 = cache.get_frames(path, n=4, size=224)
    frames2, ts2 = cache.get_frames(path, n=4, size=224)
    assert (cache.misses, cache.hits) == (1, 1)
    assert frames1.shape == (4, 224, 224, 3)
    assert np.array_equal(frames1, frames2)
    assert ts1 == ts2


def test_cache_key_is_relative():
    from sva.config import SOCCERBENCH_ROOT
    from sva.perception import rel_key
    clip = SOCCERBENCH_ROOT / "materials" / "q8" / "x.mp4"
    assert rel_key(clip) == "materials/q8/x.mp4"                       # giong nhau tren moi may
    assert cache_key(clip, 4, 32) != cache_key(clip, 8, 32) != cache_key(clip, 8, 32, "clip_vit_l")


def test_frames_in_memory_only(tmp_path, monkeypatch):
    monkeypatch.setenv("SVA_FRAME_CACHE", "0")                          # server: khong ghi cache frame ra dia
    path = load_clean("q8")[0].materials[0]
    cache = PerceptionCache(tmp_path / "frames", tmp_path / "emb", memory=1)
    frames1, _ = cache.get_frames(path, n=4, size=224)
    frames2, _ = cache.get_frames(path, n=4, size=224)
    cache.get_frames(path, n=2, size=224)                               # day frame n=4 ra khoi bo nho
    cache.get_frames(path, n=4, size=224)
    assert (cache.misses, cache.hits) == (3, 1) and np.array_equal(frames1, frames2)
    assert not list((tmp_path / "frames").iterdir())


def test_virtual_segment():
    from sva.perception import segment, split_segment
    path = load_clean("q8")[0].materials[0]
    seg = segment(path, 1.0, 3.0)
    assert split_segment(seg) == (str(path), 1.0, 3.0) and split_segment(path) == (str(path), None, None)
    frames, ts = sample_frames(seg, 4)
    ref, ts_ref = sample_frames(path, 4, 1.0, 3.0)
    assert np.array_equal(frames, ref) and ts == ts_ref and 1.0 < ts[0] < ts[-1] < 3.0


def test_unsqueeze_only_squeezed_clips():
    from sva.perception import unsqueeze
    squeezed = np.zeros((398, 224, 3), np.uint8)                        # clip SoccerReplay bi bop
    portrait = np.zeros((640, 360, 3), np.uint8)                        # video doc that
    assert unsqueeze(squeezed).shape == (224, 398, 3) and unsqueeze(portrait).shape == (640, 360, 3)
