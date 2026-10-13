import numpy as np
import pytest

from sva.reasoner import Pairwise, Reasoner, Reorder
from sva.schema import Evidence, Sample

KEYS = ["O1", "O2", "O3", "O4"]


def sample(keys=KEYS):
    return Sample("q8", 0, "?", ["a.mp4"], keys, {k: f"text {k}" for k in keys}, keys[0])


def ev(probs, name="t", keys=KEYS):
    p = np.array(probs, dtype=float) / sum(probs)
    dist = dict(zip(keys, p.tolist()))
    s = sorted(p)[::-1]
    return Evidence(name, "categorical", keys, dist, float(s[0] - s[1]))


class Fake:
    def __init__(self, name, kind, probs, ok=True):
        self.name, self.kind, self.probs, self.ok, self.calls = name, kind, probs, ok, 0

    def applicable(self, sample, state):
        return self.ok

    def __call__(self, sample, state):
        self.calls += 1
        return ev(self.probs, self.name)


def test_stops_when_confident():
    act = Fake("more", "more_info", [1, 1, 1, 1])
    out = Reasoner([act], theta=0.15).run(sample(), [ev([0.7, 0.1, 0.1, 0.1])])
    assert out["stop"] == "du tin" and out["pred"] == "O1" and act.calls == 0


def test_more_info_when_spread():
    act = Fake("more", "more_info", [0.8, 0.1, 0.05, 0.05])
    out = Reasoner([act], theta=0.15).run(sample(), [ev([0.3, 0.25, 0.25, 0.2])])
    assert act.calls == 1
    assert out["steps"][0]["action"] == "more"
    assert out["pred"] == "O1" and out["confidence"] >= 0.15


def test_narrow_when_two_options_close():
    narrow, more = Fake("pair", "narrow", [0.05, 0.85, 0.05, 0.05]), Fake("more", "more_info", [1, 1, 1, 1])
    out = Reasoner([more, narrow], theta=0.15).run(sample(), [ev([0.42, 0.40, 0.1, 0.08])])
    assert out["steps"][0]["action"] == "pair"
    assert "O1" in out["steps"][0]["reason"] and "O2" in out["steps"][0]["reason"]


def test_reframe_when_sources_conflict():
    re_, more = Fake("reorder", "reframe", [0.1, 0.6, 0.2, 0.1]), Fake("more", "more_info", [1, 1, 1, 1])
    first = [ev([0.5, 0.3, 0.1, 0.1], "a"), ev([0.3, 0.5, 0.1, 0.1], "b")]
    out = Reasoner([more, re_], theta=0.15).run(sample(), first)
    assert out["steps"][0]["action"] == "reorder"


def test_max_steps_and_no_repeat():
    acts = [Fake(f"a{i}", "more_info", [1, 1, 1, 1]) for i in range(5)]
    out = Reasoner(acts, theta=0.15, max_steps=2).run(sample(), [ev([1, 1, 1, 1])])
    assert out["stop"] == "het luot"
    assert [a.calls for a in acts] == [1, 1, 0, 0, 0]


def test_runs_out_of_actions():
    out = Reasoner([Fake("x", "more_info", [1, 1, 1, 1], ok=False)], theta=0.15).run(sample(), [ev([1, 1, 1, 1])])
    assert out["stop"] == "het hanh dong"


def test_failed_action_is_ignored():
    class Broken(Fake):
        def __call__(self, sample, state):
            raise RuntimeError("video loi")
    out = Reasoner([Broken("bad", "more_info", None)], theta=0.15).run(sample(), [ev([0.35, 0.3, 0.2, 0.15])])
    assert out["evidence"][-1]["failed"]
    assert out["dist"]["O1"] == pytest.approx(0.35)


class FakeScorer:
    """Gia lap B0Local.score: tra ve phan phoi theo thu tu option_keys cua mau duoc dua vao."""
    def __init__(self, probs):
        self.probs, self.calls = probs, []

    def score(self, s, prompt=None, labels=None, tool=None, **kw):
        self.calls.append((s.option_keys, labels, prompt))
        return ev(self.probs[:s.n_options], tool, keys=s.option_keys)


def test_reorder_reverses_options():
    scorer = FakeScorer([0.7, 0.1, 0.1, 0.1])
    e = Reorder(scorer)(sample(), None)
    keys, labels, prompt = scorer.calls[0]
    assert keys == ["O4", "O3", "O2", "O1"] and labels == ["A", "B", "C", "D"]
    assert "A. text O4" in prompt
    assert e.pred == "O4"      # model chon "A" = phuong an dau tien sau khi dao


def test_pairwise_redistributes_top2_mass():
    st = Reasoner([], 0.15).state(sample(), [ev([0.40, 0.38, 0.12, 0.10])], [])
    scorer = FakeScorer([0.8, 0.2])           # chon "A" = phuong an xep thu 2 (O2)
    e = Pairwise(scorer)(sample(), st)
    keys, labels, _ = scorer.calls[0]
    assert keys == ["O2", "O1"] and labels == ["A", "B"]
    assert e.dist["O2"] == pytest.approx(0.78 * 0.8) and e.dist["O1"] == pytest.approx(0.78 * 0.2)
    assert e.dist["O3"] == pytest.approx(0.12)
    assert sum(e.dist.values()) == pytest.approx(1)


def test_pairwise_not_for_two_options():
    assert not Pairwise(None).applicable(sample(["O1", "O3"]), None)


def trace_rec(gold, steps):
    return {"gold": gold, "steps": [{"dist": d, "confidence": c} for d, c in steps]}


def test_replay_stops_at_first_confident_step():
    from sva.reasoner import replay
    r = trace_rec("O2", [({"O1": 0.5, "O2": 0.4}, 0.10), ({"O1": 0.3, "O2": 0.7}, 0.40), ({"O1": 0.6, "O2": 0.4}, 0.2)])
    assert replay(r, 0.05) == ("O1", 0)
    assert replay(r, 0.3) == ("O2", 1)
    assert replay(r, 1.01) == ("O1", 2)


def test_choose_theta():
    from sva.reasoner import choose_theta
    # buoc 0 sai (O1), buoc 1 dung (O2): can theta > 0.10 de di tiep
    recs = {"q8": [trace_rec("O2", [({"O1": 0.55, "O2": 0.45}, 0.10), ({"O1": 0.2, "O2": 0.8}, 0.6)])]}
    best, scores = choose_theta(recs, grid=(0, 0.1, 0.2, 1.01))
    assert best == 0.2 and scores[0] == 0 and scores[0.2] == 1
