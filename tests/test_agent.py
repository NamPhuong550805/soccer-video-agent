import numpy as np

from sva.agent.grounding import DIM_LABELS, Grounding, dimension_from_question, rouge_l
from sva.agent.question import QuestionAgent, parse_time, rule_task
from sva.perception.video import segment
from sva.schema import Sample


def q(text, n=1, options=None, task="q8"):
    opts = options or {"O1": "a", "O2": "b"}
    return Sample(task, 0, text, [f"v{i}.mp4" for i in range(n)], list(opts), opts, "O1")


def test_rules_route_by_clips_keywords_and_option_length():
    assert rule_task("Which clip is replayed?", 5)[0] == "q9"
    assert rule_task("anything", 3)[0] == "q14"
    assert rule_task("In the video showing a yellow card offense, which color jersey commits the foul?")[0] == "q13"
    assert rule_task("There are 2 videos of the same foul: Main camera center lens at 1.0x speed. Was it a handball?")[0] == "q14"
    assert rule_task("How does the camera position change?")[0] == "q8"
    assert rule_task("Write a comment for this football video.")[0] == "q11"
    assert rule_task("What kind of football event is depicted?")[0] == "q10"
    long = {f"O{i}": "[PLAYER] ([TEAM]) sends a lofted cross into the box which is headed clear." for i in range(4)}
    assert rule_task("Describe the content of this football video.", 1, long) == ("q11", "options")
    assert rule_task("Describe the content of this football video.") == (None, None)


def test_llm_fallback_and_scope_check():
    class Backend:                       # LLM: cau ve thong ke -> "ngoai pham vi" (nhan G)
        def score_chat(self, messages, labels):
            return np.array([0.0 if l == "G" else -4.0 for l in labels])
    agent = QuestionAgent(Backend())
    assert agent("Who scored the most goals this season?").task == "out"                 # luat khong khop -> LLM
    assert agent("What happens at 1:10?").task == "q10"                                  # luat khop, khong hoi LLM
    assert agent("Which camera shows the season's top scorer?", check_scope=True).task == "out"
    assert agent("What happens at 1:10?").time == (70, 70) and agent("What happens at 1:10?").open


def test_parse_time():
    assert parse_time("between 0:15 and 0:22") == (15, 22) and parse_time("around minute 23") == (1380, 1380)
    assert parse_time("at 40 seconds") == (40.0, 40.0) and parse_time("lens at 1.8x speed") is None


def test_dimension_from_question():
    cases = {"How severe was this foul?": "severity", "Was this an offensive foul?": "offence",
             "Where did the foul make contact on the body?": "bodypart", "Did the offence involve deliberate hand contact?": "handball",
             "Was the ball involved in the physical contact?": "touch_ball", "Did the players make contact in this foul?": "contact",
             "Which upper body part was the focus of the offence?": "upper_bodypart", "Is this a multi-foul situation?": "multiple",
             "Was there an attempt to play the ball?": "try_to_play", "Which foul type is displayed in the videos?": "action"}
    for text, dim in cases.items():
        assert dimension_from_question(text) == dim, text
    assert DIM_LABELS["upper_bodypart"] == ["Use of arms", "Use of shoulder"]


def test_open_q9_maps_position_and_expand():
    s = Sample("q9", 0, "q", ["r.mp4", "a.mp4", "b.mp4", "c.mp4", "d.mp4"], ["O1", "O2", "O3", "O4"],
               {"O1": "The 1st one.", "O2": "The 2nd one.", "O3": "The 3rd one.", "O4": "The 4th one."}, "O3")

    class FakeSVA:
        def __call__(self, sample):
            assert sample.gold_key is None and sample.option_texts["O3"] == "The 3rd one."
            return {"pred": "O3", "dist": {k: 0.25 for k in sample.option_keys}, "confidence": 0.0,
                    "evidence": [], "steps": [], "branch": "C"}
    g = Grounding()
    g.attach(FakeSVA())
    out = g.solve_open(s)
    assert out["pred"] == "O3" and out["open_pred"] == "c.mp4" and out["select"] == "position"
    real = Sample("q10", 0, "q", [segment("v.mp4", 10, 20)], ["O1"], {"O1": "a"}, None)   # doan ao cua video that
    assert not g.can_expand(s) and g.can_expand(real)                                    # clip benchmark khong mo rong


def test_rouge_l():
    assert rouge_l("a b c", "a b c") == 1.0 and rouge_l("a b", "c d") == 0.0
    assert 0.5 < rouge_l("[PLAYER] ([TEAM]) fouls an opponent.", "[PLAYER] ([TEAM]) commits a foul on an opponent.") < 0.7


def test_from_video_windows_and_clarify():
    from sva.agent.question import Intent
    from sva.data.clean import load_clean
    from sva.perception.video import split_segment
    clip = load_clean("q10")[0].materials[0]                            # clip ~30 giay
    g = Grounding()
    s, _ = g.from_video(Intent("q10", "rule", time=(10, 10)), [clip], "What happens at 0:10?", None)
    assert split_segment(s.materials[0])[1:] == (0.0, 25.0) and s.gold_key is None   # +-15 s, cat o dau video
    s, _ = g.from_video(Intent("q10", "rule"), [clip], "What happens?", {"O1": "goal", "O2": "corner"})
    assert s.materials == [clip] and s.option_texts == {"O1": "goal", "O2": "corner"}   # video ngan: lay ca video
    s, _ = g.from_video(Intent("q8", "rule", time=(3, 3)), [clip], "How does the camera change at 0:03?", None)
    assert split_segment(s.materials[0])[1:] == (0.0, 7.0)                              # q8: +-4 s


def test_from_video_q9_uses_replay_time_and_earlier_shots():
    from sva.agent.question import Intent
    vmap = {"duration": 60, "shots": [{"start": 0, "end": 8}, {"start": 8, "end": 9}, {"start": 9, "end": 20},
                                      {"start": 20, "end": 35}, {"start": 35, "end": 60}]}
    s, _ = Grounding().from_video(Intent("q9", "rule", time=(40, 40)), ["m.mp4"], "the replay at 0:40?", None, vmap)
    assert s.materials[0] == "m.mp4#t=34.00,46.00"                       # replay = +-6 s quanh moc thoi gian
    assert s.materials[1:] == ["m.mp4#t=0.00,8.00", "m.mp4#t=9.00,20.00"]   # ket thuc truoc 40 - 11 s, bo canh < 1.5 s
    assert s.option_texts == {"O1": "The 1st one.", "O2": "The 2nd one."}
    s, msg = Grounding().from_video(Intent("q9", "rule"), ["m.mp4"], "Which play does the replay show?", None, vmap)
    assert s is None and "time of the replay" in msg                     # khong co moc thoi gian -> hoi lai
    s, _ = Grounding().from_video(Intent("q9", "rule", time=(5, 5)), ["m.mp4"], "replay at 0:05?", None, vmap)
    assert s is None                                                     # khong co canh nao truoc replay -> hoi lai


def test_video_map_helpers_and_answer():
    from sva.agent.answer import compose
    from sva.agent.video import VideoAgent
    windows = VideoAgent(cache_dir=__import__("pathlib").Path(__import__("tempfile").mkdtemp())).windows(100)
    assert windows[0] == (0.0, 30.0) and windows[-1][1] == 100 and len(windows) == 8
    s = Sample("q10", 0, "q", ["v.mp4#t=55.00,85.00"], ["O1", "O2"], {"O1": "corner", "O2": "goal"}, None)
    assert compose(s, "corner", 0.74, [("corner", 0.74), ("goal", 0.2)]) == \
        ("At 0:55-1:25: corner (confidence 0.74).", "answered")
    text, status = compose(s, "corner", 0.4, [("corner", 0.4), ("goal", 0.35)], abstain_below=0.5)
    assert status == "abstained" and text.startswith("Not sure. Most likely: corner (40%), then goal (35%)")
