from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("google.genai")       # B0-api chi chay o may local (server khong cai google-genai)

from sva.baselines.b0_api import B0Api
from sva.schema import Sample


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    # bo nhip gian cach 4s giua cac lan goi API khi test
    init = B0Api.__init__

    def fast_init(self, *a, **k):
        init(self, *a, **k)
        self.min_interval = 0

    monkeypatch.setattr(B0Api, "__init__", fast_init)


class FakeCache:
    def get_clips(self, paths, n_single, budget, size):
        frames = [np.zeros((n_single, 32, 32, 3), np.uint8) for _ in paths]
        return frames, [8.0] * len(paths), [[0.5] * n_single] * len(paths)


class FakeClient:
    def __init__(self, answers, fail=0):
        self.answers, self.fail, self.calls = answers, fail, []
        self.models = SimpleNamespace(generate_content=self.generate_content)

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        if self.fail:
            self.fail -= 1
            raise RuntimeError("400 INVALID_ARGUMENT bad image")
        part = lambda t: SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=t)]))
        return SimpleNamespace(candidates=[part(a) for a in self.answers])


def sample():
    return Sample("q14", 0, "Was it a handball?", ["a.mp4", "b.mp4"], ["O1", "O3"], {"O1": "No", "O3": "Yes"}, "O1")


def test_sampling_distribution():
    api = B0Api(client=FakeClient(["O1", "O1", "No.", "O3", "???"]), cache=FakeCache())
    out = api(sample())
    assert out["pred"] == "O1"
    assert out["dist"] == pytest.approx({"O1": 4 / 6, "O3": 2 / 6})
    assert out["unparsed"] == 1


def test_request_has_clip_labels_images_and_prompt():
    client = FakeClient(["O1"])
    B0Api(client=client, cache=FakeCache())(sample())
    _, contents, config = client.calls[0]
    assert contents[0] == "Clip 1:" and "Clip 2:" in contents
    assert "O3. Yes" in contents[-1]
    assert config.candidate_count == 5


def test_retry_then_failure_evidence(monkeypatch):
    monkeypatch.setattr("sva.baselines.b0_api.time.sleep", lambda s: None)
    out = B0Api(client=FakeClient(["O1"], fail=10), cache=FakeCache())(sample())
    assert out["pred"] is None and out["evidence"][0]["failed"]


def test_missing_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEYS", raising=False)
    monkeypatch.setattr("sva.baselines.b0_api.load_dotenv", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        B0Api(cache=FakeCache())


def test_falls_back_to_single_candidate_calls():
    class NoMulti(FakeClient):
        def generate_content(self, model, contents, config):
            if config.candidate_count > 1:
                raise ValueError("candidateCount is not supported")
            return super().generate_content(model, contents, config)

    client = NoMulti(["O3"])
    out = B0Api(client=client, cache=FakeCache())(sample())
    assert len(client.calls) == 5 and out["pred"] == "O3"     # lan goi bi tu choi khong duoc ghi


def test_quota_error_is_skipped_not_wrong(monkeypatch):
    class Quota(FakeClient):
        def generate_content(self, model, contents, config):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
    monkeypatch.setattr("sva.baselines.b0_api.time.sleep", lambda s: None)
    assert B0Api(client=Quota([]), cache=FakeCache())(sample())["skip"]


def test_per_minute_quota_waits_then_succeeds(monkeypatch):
    slept = []
    monkeypatch.setattr("sva.baselines.b0_api.time.sleep", lambda s: slept.append(s))

    class Minute(FakeClient):
        def generate_content(self, model, contents, config):
            if not slept:
                raise RuntimeError("429 RESOURCE_EXHAUSTED ... Please retry in 12.5s.")
            return super().generate_content(model, contents, config)

    api = B0Api(client=Minute(["O1"]), cache=FakeCache())
    out = api(sample())
    assert out["pred"] == "O1" and 13.5 in slept


def test_quota_is_retried_then_skipped(monkeypatch):
    monkeypatch.setattr("sva.baselines.b0_api.time.sleep", lambda s: None)

    class Day(FakeClient):
        def generate_content(self, model, contents, config):
            self.calls.append(1)
            raise RuntimeError("429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    client = Day([])
    out = B0Api(client=client, cache=FakeCache())(sample())
    assert out["skip"] and "PerDay" in out["reason"] and len(client.calls) == 1   # 1 key het quota ngay: bo qua


def test_overloaded_is_skipped(monkeypatch):
    monkeypatch.setattr("sva.baselines.b0_api.time.sleep", lambda s: None)

    class Busy(FakeClient):
        def generate_content(self, model, contents, config):
            raise RuntimeError("503 UNAVAILABLE high demand")

    assert B0Api(client=Busy([]), cache=FakeCache())(sample())["skip"]


def test_rotates_to_next_key_on_daily_quota(monkeypatch):
    monkeypatch.setattr("sva.baselines.b0_api.time.sleep", lambda s: None)

    class Day(FakeClient):
        def generate_content(self, model, contents, config):
            self.calls.append(1)
            raise RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    dead, alive = Day([]), FakeClient(["O3"])
    api = B0Api(client=[dead, alive], cache=FakeCache())
    out = api(sample())
    assert out["pred"] == "O3" and api.cur == 1 and len(dead.calls) == 1
    assert api(sample())["pred"] == "O3" and len(dead.calls) == 1     # khong quay lai key da het


def test_next_reset_is_midnight_pacific():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from sva.baselines.b0_api import next_reset
    la = ZoneInfo("America/Los_Angeles")
    r = next_reset(datetime(2026, 10, 5, 20, 30, tzinfo=la))
    assert (r.year, r.month, r.day, r.hour, r.minute) == (2026, 10, 6, 0, 5)
    vn = r.astimezone(ZoneInfo("Asia/Ho_Chi_Minh"))
    assert (vn.hour, vn.minute) == (14, 5)


def test_denied_key_is_dropped_and_quota_keys_come_back(monkeypatch):
    from datetime import datetime, timedelta
    monkeypatch.setattr("sva.baselines.b0_api.time.sleep", lambda s: None)

    class Err(FakeClient):
        def __init__(self, msg):
            super().__init__([])
            self.msg = msg

        def generate_content(self, model, contents, config):
            self.calls.append(1)
            raise RuntimeError(self.msg)

    denied = Err("403 PERMISSION_DENIED Your project has been denied access")
    api = B0Api(client=[denied, FakeClient(["O1"]), FakeClient(["O3"])], cache=FakeCache())
    assert api(sample())["pred"] == "O1" and api.dead == {0}
    api.exhausted.add(1)
    api.reset_at = datetime.now().astimezone() - timedelta(minutes=1)   # da qua gio reset
    api.cur = 2
    api._rotate("429 PerDay")
    assert api.cur == 1 and not api.exhausted                          # key 1 dung lai, khong quay ve key bi khoa
    only_denied = B0Api(client=[denied], cache=FakeCache())
    assert only_denied(sample())["skip"]                               # khong ghi thanh cau sai
    banned = B0Api(client=[Err("401 UNAUTHENTICATED service account is deleted or disabled"), FakeClient(["O3"])],
                   cache=FakeCache())
    assert banned(sample())["pred"] == "O3" and banned.dead == {0}
