import numpy as np
import pytest

from sva.constants import SOCCERBENCH_ROOT
from sva.data.clean import load_clean
from sva.perception.cache import PerceptionCache, cache_key
from sva.perception.video import letterbox, probe, sample_frames, timestamps_for

pytestmark = pytest.mark.skipif(not SOCCERBENCH_ROOT.is_dir(), reason="chua co dataset")

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
