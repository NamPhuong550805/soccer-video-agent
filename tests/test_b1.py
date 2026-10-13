import ast
import re
from pathlib import Path

import numpy as np
import pytest

from sva.baselines import b1_vanilla as b1
from sva.baselines.b1_tools import (CAMERA_ASK, CAMERA_LEARN, FOUL_CATEGORY_PROMPT, FOUL_OPTIONS, Scene, Toolbox,
                                    camera_position, foul_vlm_prompt)
from sva.models.unisoccer import CLASS_NAMES
from sva.schema import Sample


class FakeBackend:
    """Bo thuc thi tra ve <Call> co san; ghi lai moi prompt de kiem tra luong thong tin."""

    def __init__(self, replies=None, scores=None):
        self.replies = list(replies or [])
        self.scores = scores
        self.chats, self.gens, self.scored = [], [], []

    def chat(self, messages, **kw):
        self.chats.append(messages)
        return self.replies.pop(0) if self.replies else "Answer: O1"

    def generate(self, frames, prompt, duration=None, **kw):
        self.gens.append(prompt)
        return "Yes"

    def score_chat(self, messages, labels):
        self.scored.append(messages)
        return np.array(self.scores if self.scores is not None else [0.0] * len(labels))


class FakeCache:
    def get_frames(self, path, n, size):
        return np.zeros((n, 8, 8, 3), np.uint8), [0.5 / n * 4.0 * (2 * k + 1) / 2 for k in range(n)]


def call(query, tool="X", end=False):
    tag = "EndCall" if end else "Call"
    return (f"<{tag}>\n<Purpose>p</Purpose>\n<Query>{query}</Query>\n<Material>['a.mp4']</Material>\n"
            f"<Tool>{tool}</Tool>\n</{tag}>")


def make(task, n_clips=1, keys=("O1", "O2", "O3", "O4")):
    return Sample(task, 0, "Which color jersey intercepts the set piece?", [f"c{i}.mp4" for i in range(n_clips)],
                  list(keys), {k: f"text {k}" for k in keys}, keys[1])


def system(backend, monkeypatch, tools=None):
    sys_ = b1.B1Vanilla.__new__(b1.B1Vanilla)
    sys_.cfg = {"exec_tokens": 8, "answer_tokens": 8, "prompt_lookup": 10}
    sys_.vlm = sys_.llm = backend
    sys_.tools = tools
    sys_.toolbox = "TOOLBOX"
    monkeypatch.setattr(b1, "rel_path", lambda p: str(p))
    return sys_


class RecTools:
    def __init__(self, out="obs"):
        self.calls, self.out = [], out

    def __call__(self, name, query, material):
        self.calls.append((name, query, material))
        return self.out


def test_parse_call():
    assert b1.parse_call(call("what color?", "Vision Language Model")) == (
        "Vision Language Model", "what color?", "['a.mp4']")
    assert b1.parse_call("no tags") == (None, None, None)


def test_tool_gets_executor_query_and_options_go_to_executor(monkeypatch):
    be = FakeBackend([call("Describe the jersey colors"), call("final", "LLM", end=True), "Answer: green"],
                     scores=[0.0, 2.0, 0.0, 0.0])
    tools = RecTools("The defender in green intercepts.")
    s = make("q13")
    out = system(be, monkeypatch, tools)(s)
    # tool chi nhan query do bo thuc thi viet, khong nhan cau hoi goc
    assert tools.calls == [("Vision Language Model", "Describe the jersey colors", ["c0.mp4"])]
    exec_prompt = be.chats[0][1]["content"]
    assert s.question in exec_prompt and "O3: text O3" in exec_prompt       # ban MCQ: bo thuc thi thay option
    choice = be.scored[0][1]["content"]
    assert "The defender in green intercepts." in choice and "Answer: green" in choice
    assert out["pred"] == "O2" and [st["tool"] for st in out["steps"]] == ["Vision Language Model", "LLM"]


def test_missing_query_falls_back_to_question(monkeypatch):
    be = FakeBackend(["garbage", "garbage", "ans"])
    tools = RecTools()
    s = make("q13")
    system(be, monkeypatch, tools)(s)
    assert tools.calls[0][1] == s.question


def test_final_llm_prompt_is_assistant_turn(monkeypatch):
    be = FakeBackend([call("q"), call("Pick the answer", "LLM", end=True), "ans"])
    system(be, monkeypatch, RecTools())(make("q10"))
    last = be.chats[-1]
    assert last[-1]["role"] == "assistant" and "'Pick the answer'" in last[-1]["content"]


def test_routing_q9_q14_q8(monkeypatch):
    sys_ = system(FakeBackend(), monkeypatch)
    q9 = make("q9", 5)
    assert [sys_.route("Commentary Generation", k, q9, [])[0] for k in range(5)] == q9.materials
    q14 = make("q14", 3, ("O1", "O3"))
    assert sys_.route("Foul Recognition", 0, q14, []) == q14.materials
    scenes = [Scene(f"scene_{i}.mp4", "c0.mp4", i, i + 1, 25) for i in range(1, 4)]
    q8 = make("q8")
    assert sys_.route("Camera Detection", 0, q8, scenes) == [scenes[0]]
    assert sys_.route("Camera Detection", 1, q8, scenes) == [scenes[-1]]
    assert sys_.route("Camera Detection", 1, q8, []) == ["c0.mp4"]


def test_tool_error_is_reported_as_text(monkeypatch):
    def boom(name, query, material):
        raise RuntimeError("bad file")
    be = FakeBackend([call("q"), call("f", "LLM", end=True), "ans"])
    out = system(be, monkeypatch, boom)(make("q10"))
    assert out["steps"][0]["result"] == "Error occurred: bad file"


def test_camera_position_regex():
    assert camera_position("It is the Main camera left view.") == "Main camera left"
    assert camera_position("unknown") == "None"


def test_foul_recognition_vote_and_category(monkeypatch):
    be = FakeBackend(["[CLASS]: Handball"])
    answers = iter(["No", "Yes", "Yes"])
    be.generate = lambda frames, prompt, duration=None, **kw: (be.gens.append(prompt), next(answers))[1]
    tb = Toolbox.__new__(Toolbox)
    tb.vlm = tb.llm = be
    tb.cache, tb.n_frames, tb.size, tb.b1 = FakeCache(), 8, 336, {"vlm_tokens": 8}
    out = tb.foul_recognition("3 views ... at 1.0x speed. Did this foul involve a handball?", ["a", "b", "c"])
    assert out.endswith("The inferred answer should be Yes.") and "{No, Yes, Yes}" in out
    # chi phan sau "speed." duoc dua vao prompt
    assert be.chats[0][1]["content"].endswith("Did this foul involve a handball?")
    assert '"Yes", "No"' in be.gens[0] and "1th video" in be.gens[0]


def test_foul_tie_keeps_first_and_multiword_category_has_no_options():
    assert '"Yes"' not in foul_vlm_prompt("q", "Upper", 1)          # \w+ chi bat "Upper" -> khong co option
    assert "Upper body part" in FOUL_OPTIONS


def test_action_classifier_format():
    tb = Toolbox.__new__(Toolbox)
    tb.classifier = type("C", (), {"probs": lambda self, p: {"corner": 0.604, "free kick": 0.249, "goal": 0.04}})()
    out = tb.action_classifier("", ["a.mp4"])
    assert out == ("The classification probabilities of this soccer video clip is: corner - 60.4%, "
                   "free kick - 24.9% (only above 5% mentioned).")


def test_chains_match_recommended():
    assert b1.CHAINS["q9"].count("Commentary Generation") == 5
    assert b1.CHAINS["q8"] == ["Shot Change", "Camera Detection", "Camera Detection", "LLM"]
    assert all(c[-1] == "LLM" for c in b1.CHAINS.values())


ORIG = Path(__file__).resolve().parents[1] / "external" / "SoccerAgent"


def original(path, names):
    """Lay ham trong code goc ma khong import ca file (file goc nap model va API khi import)."""
    tree = ast.parse((ORIG / path).read_text(encoding="utf-8"))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"re": re}
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), ns)
    return ns


@pytest.mark.skipif(not ORIG.exists(), reason="chua clone code goc vao external/SoccerAgent")
def test_prompts_match_original_code():
    g = original("multiagent_platform.py", {"parse_input", "generate_prompt_execution", "generate_LLM_prompt"})
    s = make("q14", 3, ("O1", "O3"))
    chain = b1.CHAINS["q14"]
    plan = "Known Info: [$VideoClip$]\nTool Chain: [" + " -> ".join(f"*{c}*" for c in chain) + "]"
    toolbox = b1.load_toolbox()
    assert g["generate_prompt_execution"](s.question, s.materials, plan, toolbox, b1.options_str(s)) == \
        b1.EXEC_PROMPT.format(query=s.question, material=s.materials, options=b1.options_str(s),
                              known=["VideoClip"], chain=chain, toolbox=toolbox)
    assert g["generate_LLM_prompt"]("q") == b1.LLM_PROMPT.format(query="q")

    src = (ORIG / "platform_full_version.py").read_text(encoding="utf-8")
    assert re.search(r'INSTRUCTION = f"""(.*?)"""', src, re.S).group(1) == b1.CHOICE_SYSTEM
    tmpl = re.search(r'prompt = f"""(.*?)"""', src, re.S).group(1)
    assert tmpl.replace("{options_str}", "{options}").replace("{openA_process}", "{process}") == b1.CHOICE_PROMPT

    f = original("toolbox/foul_recognition.py", {"generate_prompt", "generate_vlm_prompt"})
    assert f["generate_prompt"]("Q?") == FOUL_CATEGORY_PROMPT + "Q?"
    for cat in ["Offence", "Severity", "Upper", "Handball"]:
        assert f["generate_vlm_prompt"]("Q?", cat, 2) == foul_vlm_prompt("Q?", cat, 2)

    cam = (ORIG / "toolbox/camera_detection.py").read_text(encoding="utf-8")
    assert re.search(r'learn_prompt = "(.*?)"\n', cam).group(1) == CAMERA_LEARN
    assert set(re.findall(r'ask_prompt = "(.*?)"\n', cam)) == {CAMERA_ASK}
    labels = ast.literal_eval(re.search(r"camera_position = (\[.*?\])", cam).group(1))
    assert labels == sorted(p.stem for p in (ORIG / "toolbox/utils/example_tiny").glob("*.png"))

    cls = (ORIG / "toolbox/unisoccer_com_cls.py").read_text(encoding="utf-8")
    assert ast.literal_eval(re.search(r"CLASS_NAMES = (\[.*?\])", cls).group(1)) == CLASS_NAMES


def test_every_chain_tool_dispatches_to_a_tool_method():
    be = FakeBackend(["[CLASS]: Handball"])
    clf = type("C", (), {"probs": lambda self, p: {"corner": 0.9, "goal": 0.1}})()
    tb = Toolbox(be, be, FakeCache(), clf)
    names = {t for chain in b1.CHAINS.values() for t in chain if t != "LLM"}
    assert names == set(tb.tools)
    assert all(getattr(f, "__self__", None) is tb for f in tb.tools.values())     # khong bi thuoc tinh che mat
    assert tb("Vision Language Model", "which color?", ["a.mp4"]) == "Yes"
    assert tb("Commentary Generation", "", ["a.mp4"]).startswith("The according commentary to this video is:")
    assert tb("Action Classifier", "", ["a.mp4"]).startswith("The classification probabilities")
    assert "The inferred answer should be Yes." in tb("Foul Recognition", "x speed. Handball?", ["a.mp4", "b.mp4"])
