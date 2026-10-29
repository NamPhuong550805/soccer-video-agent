import numpy as np
import pytest

from sva.branches import CAMERA_PROMPT, HINTS, BranchB, parse_switch
from sva.config import CAMERA_LABELS, TASK_IDS
from sva.data import Sample, load_clean
from sva.evidence import Evidence, build_categorical
from sva.pipeline import route, router_accuracy
from sva.reasoner import Pairwise, Reasoner, Reorder


# ---------------------------------------------------------------- Router, cac nhanh
def q8(texts):
    keys = [f"O{i + 1}" for i in range(len(texts))]
    return Sample("q8", 0, "How does the camera position change?", ["a.mp4"], keys, dict(zip(keys, texts)), "O1")


class FakeBackend:
    """Kenh camera: frame dau la camera i, frame cuoi la camera j (danh dau bang gia tri pixel)."""

    def __init__(self):
        self.prompts = []

    def score_labels(self, frames, prompt, labels, duration=None):
        self.prompts.append(prompt)
        lp = np.full(len(labels), -10.0)
        lp[int(frames[0, 0, 0, 0])] = 0.0
        return lp


class FrameScorer:
    def __init__(self, start, end):
        self.backend = FakeBackend()
        f = np.zeros((8, 4, 4, 3), np.uint8)
        f[:4], f[4:] = start, end
        self.frames = f
        self.prompts = []

    def clips(self, sample):
        return [self.frames], [8.0], [[0.5 + i for i in range(8)]]

    def score(self, sample, prompt=None, tool=None, **kw):
        self.prompts.append(prompt)
        return build_categorical(tool, sample, np.zeros(sample.n_options))


def test_router_rules_match_task_table():
    for t in TASK_IDS:
        assert router_accuracy(load_clean(t)) == 1.0
    assert route(q8(["From Public to Public."])) == "B"
    multi = Sample("q8", 0, "q", ["a.mp4", "b.mp4"], ["O1"], {"O1": "x"}, "O1")
    assert route(multi) == "C"                                           # nhieu clip -> nhanh C


def test_parse_switch():
    s = q8(["From Main camera center to Public.", "From Public to Public.", "From Other to Spider camera."])
    i = {c: k for k, c in enumerate(CAMERA_LABELS)}
    assert parse_switch(s) == [(i["Main camera center"], i["Public"]), (i["Public"], i["Public"]),
                               (i["Other"], i["Spider camera"])]
    assert parse_switch(q8(["From Main camera center to somewhere."])) is None


def test_camera_channel_picks_start_and_end():
    i = {c: k for k, c in enumerate(CAMERA_LABELS)}
    s = q8(["From Public to Public.", "From Main camera left to Public.", "From Public to Main camera left."])
    scorer = FrameScorer(start=i["Public"], end=i["Main camera left"])
    evs = BranchB(scorer, edge_frames=3).run(s)
    assert [e.tool_name for e in evs] == ["holistic", "camera_switch"]
    assert evs[1].pred == "O3"                                           # dau Public, cuoi Main camera left
    assert all(p == CAMERA_PROMPT for p in scorer.backend.prompts)


def test_q13_gets_hint_and_no_camera_channel():
    s = Sample("q13", 0, "Which color jersey takes the corner?", ["a.mp4"], ["O1", "O2"],
               {"O1": "red", "O2": "blue"}, "O1")
    scorer = FrameScorer(0, 0)
    evs = BranchB(scorer).run(s)
    assert len(evs) == 1 and scorer.prompts[0].startswith(HINTS["q13"])


def test_strip_bars_keeps_content_only():
    from sva.branches import strip_bars
    f = np.zeros((2, 336, 336, 3), np.uint8)
    f[:, 73:262, :, :] = 200                                            # noi dung 336x189 giua 2 vien den
    out = strip_bars(f, size=32)
    assert out.shape == (2, 32, 32, 3) and out.min() == 200


def test_candidate_order_and_chamfer_penalty():
    from sva.branches import candidate_order, chamfer
    s = Sample("q9", 0, "q", ["r", "a", "b", "c", "d"], ["O1", "O2", "O3", "O4"],
               {"O1": "The 1st one.", "O2": "The 2nd one.", "O3": "The 3rd one.", "O4": "The 4th one."}, "O2")
    assert candidate_order(s) == [1, 2, 3, 4]
    e = np.eye(4)
    forward, _, omega_f, _ = chamfer(e, e, lam=0.5)                      # khop dung thu tu thoi gian
    backward, _, omega_b, _ = chamfer(e, e[::-1], lam=0.5)               # cung cac frame nhung dao nguoc
    assert omega_f == 0 and omega_b == 1 and forward > backward


def test_replay_matcher_maps_candidates_to_options():
    from sva.branches import BranchC

    class FakeCache:                                                     # clip c co embedding e_c, replay giong clip "c3"
        def get_embedding(self, path, n, size, encoder):
            e = np.zeros((2, 4), np.float32)
            e[:, {"r": 2, "c1": 0, "c2": 1, "c3": 2, "c4": 3}[path]] = 1
            return e

    s = Sample("q9", 0, "q", ["r", "c1", "c2", "c3", "c4"], ["O1", "O2", "O3", "O4"],
               {"O1": "The 4th one.", "O2": "The 3rd one.", "O3": "The 1st one.", "O4": "The 2nd one."}, "O2")
    evs = BranchC(None, FakeCache(), {"encoder": "clip_vit_l", "n_frames": 2}).run(s)
    assert len(evs) == 1 and evs[0].pred == "O2" and evs[0].dist["O2"] > 0.99
    assert evs[0].material_ref["max_cos"] == [0.0, 1.0, 0.0, 0.0]


def test_fusion_two_views_disagree():
    from sva.evidence import hard_vote, linear_pool, log_pool, ordinal_fusion
    v1, v2 = {"O1": 0.5, "O2": 0.4, "O4": 0.1}, {"O1": 0.1, "O2": 0.55, "O4": 0.35}
    assert hard_vote([v1, v2]) == {"O1": 0.5, "O2": 0.5, "O4": 0.0}           # bo phieu hoa 1-1
    assert max(linear_pool([v1, v2]).items(), key=lambda x: x[1])[0] == "O2"
    lp = log_pool([v1, v2], [1, 0])
    assert abs(lp["O1"] - 0.5) < 1e-9                                        # trong so 0 -> chi con goc 1
    scale = {"O1": 0.0, "O2": 1.0, "O4": 1.5}                                # no card / yellow / yellow or red
    d = ordinal_fusion([v1, v2], scale)
    assert max(d, key=d.get) == "O2"                                         # mu = 0.81 -> gan yellow nhat


def test_dimension_and_question_part():
    from sva.branches import classify_dimension, question_part
    q = ("There are 2 video replays of the same foul, including: Main camera center lens at 1.0x speed, "
         "Close-up player or field referee lens at 1.8x speed. Was the player's hand/arm involved in the foul?")
    assert question_part(q) == "Was the player's hand/arm involved in the foul?"

    def s(q, *texts):
        keys = [f"O{i + 1}" for i in range(len(texts))]
        return Sample("q14", 0, q, ["a", "b"], keys, dict(zip(keys, texts)), "O1")
    assert classify_dimension(s(q, "No", "Yes")) == "handball"
    assert classify_dimension(s("Was the ball touched during the foul?", "No", "Yes")) == "touch_ball"
    assert classify_dimension(s("Is this a multi-foul situation?", "No", "Yes")) == "multiple"
    assert classify_dimension(s("Was there an attempt to play the ball?", "No", "Yes")) == "try_to_play"
    assert classify_dimension(s("How severe?", "No card", "Yellow card or red card")) == "severity"
    assert classify_dimension(s("Category?", "St. tackling", "Elbowing", "Dive")) == "action"
    assert classify_dimension(s("Where?", "Arms", "Use of shoulder")) == "upper_bodypart"


def test_multiview_weights_clear_view():
    from sva.branches import BranchC
    from sva.evidence import build_categorical

    class Backend:                                  # cau hoi do ro: goc 0 ro (P(A)=0.9), goc 1 mo (0.1)
        def score_labels(self, frames, prompt, labels, duration=None):
            p = 0.9 if frames[0][0, 0, 0, 0] == 0 else 0.1
            return np.log([p, 1 - p])

    class Scorer:
        backend, questions = Backend(), []

        def clips(self, sample, n_single=None, budget=None):
            v = int(sample.materials[0][-1])
            return [np.full((n_single, 2, 2, 3), v, np.uint8)], [8.0], [[0.5]]

        def score(self, sample, n_single=None, tool=None, **kw):
            self.questions.append(sample.question)
            lp = [0.0, -3.0] if sample.materials[0] == "v0" else [-0.5, 0.0]   # goc 0 chac O1, goc 1 hoi nghieng O2
            return build_categorical(tool or "b0", sample, np.array(lp))

    q = ("There are 2 video replays of the same foul, including: Main camera center lens at 1.0x speed, "
         "Close-up player or field referee lens at 1.8x speed. Did the player attempt to play the ball?")
    s = Sample("q14", 0, q, ["v0", "v1"], ["O1", "O2"], {"O1": "No", "O2": "Yes"}, "O1")
    scorer = Scorer()
    evs = BranchC(scorer, None, {"encoder": "clip_vit_l"}, {"with_holistic": False}).run(s)
    assert len(evs) == 1 and evs[0].pred == "O1" and evs[0].material_ref["quality"] == [0.9, 0.1]
    assert scorer.questions[1] == "This video shows the foul from the Close-up player or field referee view. " \
                                  "Did the player attempt to play the ball?"


def test_branch_a_q10_and_q11(tmp_path):
    from sva.branches import BranchA
    from sva.models import CLASS_NAMES

    class Classifier:                                   # video la mot pha phat goc
        def probs(self, path):
            return {c: (0.9 if c == "corner" else 0.1 / 22) for c in CLASS_NAMES}

    class Backend:                                      # binh luan co chu "corner" -> su kien corner
        name, calls = "fake", 0

        def score_chat(self, messages, labels):
            Backend.calls += 1
            assert len(labels) == len(CLASS_NAMES)                     # 23 nhan A..W
            text = messages[0]["content"][0]["text"].split('"')[1]
            target = "corner" if "corner" in text else "goal"
            return np.array([0.0 if c == target else -5.0 for c in CLASS_NAMES])

    class Scorer:
        backend = Backend()

    a = BranchA(Scorer(), Classifier())
    a.cache_dir = tmp_path
    q10 = Sample("q10", 0, "What event?", ["v"], ["O1", "O2"], {"O1": "Goal", "O2": "corner"}, "O2")
    assert a.run(q10)[0].pred == "O2" and a.run(q10)[0].tool_name == "action_classifier"
    q11 = Sample("q11", 0, "Commentary?", ["v"], ["O1", "O2"],
                 {"O1": "Goal! [PLAYER] scores.", "O2": "[TEAM] win a corner."}, "O2")
    ev = a.run(q11)[0]
    assert ev.pred == "O2" and ev.dist["O2"] > 0.9
    a.run(q11)
    assert Backend.calls == 2                                            # lan 2 doc cache


# ---------------------------------------------------------------- Reasoner
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


class ProbScorer:
    """Gia lap B0Local.score: tra ve phan phoi theo thu tu option_keys cua mau duoc dua vao."""
    def __init__(self, probs):
        self.probs, self.calls = probs, []

    def score(self, s, prompt=None, labels=None, tool=None, **kw):
        self.calls.append((s.option_keys, labels, prompt))
        return ev(self.probs[:s.n_options], tool, keys=s.option_keys)


def test_reorder_reverses_options():
    scorer = ProbScorer([0.7, 0.1, 0.1, 0.1])
    e = Reorder(scorer)(sample(), None)
    keys, labels, prompt = scorer.calls[0]
    assert keys == ["O4", "O3", "O2", "O1"] and labels == ["A", "B", "C", "D"]
    assert "A. text O4" in prompt
    assert e.pred == "O4"      # model chon "A" = phuong an dau tien sau khi dao


def test_pairwise_redistributes_top2_mass():
    st = Reasoner([], 0.15).state(sample(), [ev([0.40, 0.38, 0.12, 0.10])], [])
    scorer = ProbScorer([0.8, 0.2])           # chon "A" = phuong an xep thu 2 (O2)
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
