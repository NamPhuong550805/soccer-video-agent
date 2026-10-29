"""B1: dung lai SoccerAgent (bo cong cu va vong thuc thi cua code goc) tren cung VLM tai cho."""
import csv
import re
from collections import Counter
from dataclasses import dataclass

import cv2

from sva.config import PROJECT_ROOT, load_config
from sva.evidence import build_categorical, rel_path
from sva.models import ActionClassifier, build_backend
from sva.perception import PerceptionCache, letterbox, probe, sample_frames


# ---------------------------------------------------------------- Cong cu cua SoccerAgent
CAMERA_OPTIONS = ["Main camera center", "Close-up player or field referee", "Close-up side staff",
                  "Main camera left", "Main behind the goal", "Close-up behind the goal", "Spider camera",
                  "Main camera right", "Public", "Goal line technology camera", "Close-up corner",
                  "Inside the goal", "Other"]
CAMERA_LEARN = ("I want you to help me identify the camera position of a football game photo. Now I will give "
                "you some example images, each of which corresponds to a specific camera position. Please learn "
                "the characteristics of these images for classification of new photos.")
CAMERA_ASK = ("What is the camera position in this picture? The answer should be chosen from the following "
              "options: [" + ", ".join(CAMERA_OPTIONS) + "].")

# MatchVoice (LLaMA-3 8B) khong chay duoc tren GPU 4GB -> dung backbone sinh binh luan cung kieu an danh.
# Prompt chon tren 8 clip dev q11 (ban "TV commentator" sinh "And that's a goal" cho gan nhu moi clip)
COMMENTARY_PROMPT = ("You are a soccer commentator. Watch the video clip and write the commentary for exactly what "
                     "happens in it (for example a foul, a corner, a shot, a substitution, a goal, a save, a "
                     "throw-in). One sentence. Use [PLAYER], [TEAM], [REFEREE] and [COACH] instead of names.")

FOUL_CATEGORY_PROMPT = (
    "You are tasked with classifying a question into one of the following 11 categories:\n"
    "1. Offence - Classifies whether the foul occurred during an offensive play.\n"
    "2. Contact - Determines if there was contact between players during the foul.\n"
    "3. Bodypart - Identifies the body part involved in the foul.\n"
    "4. Upper body part - Specifies the exact upper body part involved in the foul.\n"
    "5. Action class - Categorizes the type of foul action.\n"
    "6. Severity - Evaluates the severity of the foul.\n"
    "7. Multiple fouls - Checks if multiple fouls occurred.\n"
    "8. Try to play - Determines if the player attempted to continue playing after the foul.\n"
    "9. Touch ball - Verifies if the offending player touched the ball.\n"
    "10. Handball - Checks if the foul involved handball.\n"
    "11. Handball offence - Determines if the foul was caused by handball.\n\n"
    "Your task is to analyze the given question and return the category it belongs to.\n"
    "The output format must strictly be:\n"
    "[CLASS]: <Category>\n"
    "Where <Category> is one of the 11 categories listed above (e.g., Offence, Contact, etc.).\n"
    "Do not include any additional text or explanations.\n\n"
    "Question: ")
FOUL_OPTIONS = {
    "Offence": ["Offence", "No offence", "Between"],
    "Contact": ["With", "Without"],
    "Bodypart": ["Upperbody", "Underbody"],
    "Upper body part": ["Arms", "Shoulder"],
    "Action class": ["St. tackling", "Tackling", "Challenge", "Holding", "Elbowing", "High leg", "Pushing", "Dive"],
    "Severity": ["No card", "No card or yellow card", "Yellow card", "Yellow card or red card", "Red card"],
    "Multiple fouls": ["Yes", "No"],
    "Try to play": ["Yes", "No"],
    "Touch ball": ["Yes", "No"],
    "Handball": ["Yes", "No"],
    "Handball offence": ["Yes", "No"],
}


@dataclass
class Scene:
    """1 canh do Shot Change cat ra; khong ghi file, chi nho doan thoi gian trong clip."""
    name: str
    path: str
    start: float
    end: float
    n_frames: int


def camera_position(reply):
    m = re.search("|".join(re.escape(o) for o in CAMERA_OPTIONS), reply)
    return m.group(0) if m else "None"


def foul_vlm_prompt(question, category, index):
    options_str = ", ".join(f'"{o}"' for o in FOUL_OPTIONS.get(category, []))
    return (f"You are a professional football referee tasked with analyzing a video and answering the following "
            f"question: \"{question}\".\n"
            f"The video you are analyzing is the {index}th video related to this question.\n"
            f"Please provide your answer based solely on the content of the video. Your response MUST(!!!) be "
            f"strictly one of the following options: {options_str}.\n"
            f"Do not include any additional text or explanations. Provide your answer now, without any other "
            f"words, only one of the above option itself:")


class Toolbox:
    def __init__(self, vlm, llm, cache, classifier=None):
        cfg = load_config()
        self.vlm = vlm
        self.llm = llm
        self.cache = cache
        self.classifier = classifier
        self.n_frames = cfg["perception"]["n_frames_default"]
        self.size = cfg["perception"]["image_size"]
        self.b1 = cfg["b1"]
        self._examples = None
        self.tools = {"Shot Change": self.shot_change, "Camera Detection": self.camera_detection,
                      "Commentary Generation": self.commentary, "Action Classifier": self.action_classifier,
                      "Vision Language Model": self.vision_language_model, "Foul Recognition": self.foul_recognition}

    def __call__(self, name, query, material):
        return self.tools[name](query, material)

    def clip(self, path):
        frames, ts = self.cache.get_frames(path, self.n_frames, self.size)
        return frames, 2 * len(ts) * ts[0]

    # ---- Shot Change: PySceneDetect ContentDetector, mac dinh nhu code goc
    def shot_change(self, query, material):
        if len(material) == 0:
            return "No video material provided.", []
        if len(material) > 1:
            return "Only one video material is supported at a time. But you provided more than one.", []
        from scenedetect import ContentDetector, detect
        path = material[0]
        found = detect(str(path), ContentDetector())
        if not found:
            return "No scene changes detected in the video.", []
        scenes = [Scene(f"scene_{i + 1}.mp4", path, a.get_seconds(), b.get_seconds(), b.frame_num - a.frame_num)
                  for i, (a, b) in enumerate(found)]
        change = [s.end for s in scenes]
        return (f"Shot change detection completed. {len(scenes)} scenes detected. The clips are saved in "
                f"{[s.name for s in scenes]}. Change occurred at {change[:-1]} seconds."), scenes

    # ---- Camera Detection: 13 anh mau lam vi du, hoi tung frame roi lay nhan xuat hien nhieu nhat
    def examples(self):
        if self._examples is None:
            d = PROJECT_ROOT / load_config("models")["unisoccer"]["code_dir"] / "toolbox" / "utils" / "example_tiny"
            self._examples = [(p.stem, letterbox(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), self.size))
                              for p in sorted(d.glob("*.png"))]
        return self._examples

    def camera_detection(self, query, material):
        m = material[0]
        if isinstance(m, Scene):
            path, start, end, n_src = m.path, m.start, m.end, m.n_frames
        else:
            meta = probe(m)
            path, start, end, n_src = m, 0.0, None, meta.n_frames
        # code goc lay 1/10 so frame; gioi han de tong so frame 2 lan goi bang ngan sach cua B0
        n = min(self.b1["camera_max_frames"], max(1, n_src // 10))
        frames, _ = sample_frames(path, n, start, end)
        history = [{"role": "system", "content": CAMERA_LEARN}]
        for label, img in self.examples():
            history.append({"role": "user", "content": [
                {"type": "text", "text": f"The camera position corresponding to this photo is: {label}"},
                {"type": "image", "image": img}]})
        answers = []
        for f in frames:
            ask = {"role": "user", "content": [{"type": "text", "text": CAMERA_ASK},
                                               {"type": "image", "image": letterbox(f, self.size)}]}
            answers.append(camera_position(self.vlm.chat(history + [ask], max_new_tokens=32)))
        return f"The camera position in the video is: {Counter(answers).most_common(1)[0][0]}."

    def commentary(self, query, material):
        frames, dur = self.clip(material[0])
        text = self.vlm.generate(frames, COMMENTARY_PROMPT, dur, max_new_tokens=self.b1["commentary_tokens"])
        return f"The according commentary to this video is: {text}"

    def action_classifier(self, query, material):
        p = self.classifier.probs(material[0])
        top = ", ".join(f"{c} - {v * 100:.1f}%" for c, v in p.items() if v >= 0.05)
        return f"The classification probabilities of this soccer video clip is: {top} (only above 5% mentioned)."

    def vision_language_model(self, query, material):
        frames, dur = self.clip(material[0])
        return self.vlm.generate(frames, query, dur, max_new_tokens=self.b1["vlm_tokens"])

    # ---- Foul Recognition: LLM doan loai cau hoi, VLM tra loi tung goc quay, bo phieu nhan cung
    def foul_recognition(self, query, material):
        question = query.split("speed.")[-1] if "speed." in query else query
        reply = self.llm.chat([{"role": "system", "content": "You are an expert of soccer referee."},
                               {"role": "user", "content": FOUL_CATEGORY_PROMPT + question}], max_new_tokens=32)
        m = re.search(r"\[CLASS\]:\s*(\w+)", reply)
        category = m.group(1) if m else "Unknown"     # \w+ chi lay 1 tu, giu nguyen nhu code goc
        answers = []
        for i, path in enumerate(material):
            frames, dur = self.clip(path)
            answers.append(self.vlm.generate(frames, foul_vlm_prompt(question, category, i + 1), dur,
                                             max_new_tokens=self.b1["vlm_tokens"]))
        if not answers:
            return "No answers found"
        counts = Counter(answers)
        top = max(counts.values())
        best = next(a for a in answers if counts[a] == top)
        return (f"For these videos, the tool's sequential answers are: {{{', '.join(answers)}}}. "
                f"The inferred answer should be {best}.")


# ---------------------------------------------------------------- Vong thuc thi cua SoccerAgent
CHAINS = {
    "q8": ["Shot Change", "Camera Detection", "Camera Detection", "LLM"],
    "q9": ["Commentary Generation"] * 5 + ["LLM"],
    "q10": ["Action Classifier", "LLM"],
    "q11": ["Commentary Generation", "LLM"],
    "q13": ["Vision Language Model", "LLM"],
    "q14": ["Foul Recognition", "LLM"],
}

CALL_END = ["</Call>", "</EndCall>"]     # moi luot chi 1 lenh, dung sinh khi dong the
EXEC_SYSTEM = "You are a helpful multi-agent assistant that can answer questions about soccer."

EXEC_PROMPT = """As a multi-agent core in the Soccer Question Answering Assistant, you are required to execute the following tool chain to answer the question:

"{query}"

with the following additional material:

{material}

There are several options:

{options}

with the known info as:

{known}

and you should execute the following tool chain to solve the question:

{chain}

As for the usage of the tools, you should follow the following references:

{toolbox}

For every tool above, we would input queries and materials into the tool for execution, the queries are in **text** form and the materials are in list with **file paths**. If no file path is suitable, you just write in 'None' You should determine the contents of materials and queries based on the context of the question, known info and tool descriptions.

For every steps of excution, you should return me with a clear statement of the goal of this step in the context of the overall analysis, the specific tool you are using, and the input variables you are using.

<Call>
    <Purpose>Brief, clear statement of this step’s goal in context of overall analysis</Purpose>\x20
    <Query>[Query/question here(string). IMPORTANT!!: Such query is highly relevant to the toolbox descriptions. you need to think carefully about your purpose this step and generate appropriate query.]</Query>\x20
    <Material>[Material list here(a string showing list form). Here as well, you need to think carefully considering the purpose and toolbox.]</Material>
    <Tool>[Tool name here(string)]</Tool>
</Call>

If it is the last step of the execution, you should return me with the following format:

<EndCall>
    <Purpose>Brief, clear statement of this step’s goal in context of overall analysis</Purpose>\x20
    <Query>[Query/question here(string)]</Query>\x20
    <Material>[Material list with file paths here(a string showing list form)]</Material>
    <Tool>[Tool name here(string)]</Tool>
</EndCall>

Every time you return me with the instruction as above, I will execute it and return you with the feedback of the execution in this format:

<StepResult>
    <Answer>[The results of this time's execution here(string)]</Answer>
</StepResult>

For every time of generation, you should follow the following rules:
    1. You should be clear about the tool name (must be chosen from toolbox), file path and query/question in the instruction. This part is important for me to understand the context of the execution. You cannot change any of the information in the instruction.
    2. If I have given you the feedback of the execution, you should analyze what you should write in the next call based on the feedback considering the tool chain I gave you and the task descriptions and tool descriptions. You should not repeat the same instruction again.
    3. If my prompt leaves you to generate the first call, you should directly return me with the call in the form from <> to </>. You should not add any other information in the instruction.
    4. Otherwise, if in the prompt I have given you some <StepResult>, you should consider the total process of the execution and continue to return me exactly with the form from <> to </>. You should not add any other information in the instruction.

Once again, I repreat that the question is:

"{query}"

with the following additional material:

{material}

with the known info as:

{known}

and you should execute the following tool chain to solve the question:

{chain}

The following is all our execution history, now you can start with your call of first step:
    """

LLM_PROMPT = """According to the total process of above conversation, now give me your answer to the question '{query}'. You should retrun me with the following form:
\x20\x20\x20\x20
Answer: [Your answer here]
Reasoning and Explanation: [Your reasoning and explanation here]

Please make sure that your answer is consistent with the total process of the execution and the tool chain I gave you. You should not change the tool chain and the process of the execution. You should only give me the answer and the reasoning and explanation of the answer without any other words.
    """

CHOICE_SYSTEM = """
You are a football expert. You are provided with a question 'Q' and four options 'O1', 'O2', 'O3', and 'O4'.
Before I have used a helpful soccer multi-agent system to solve this process, I will tell you the total process of how agent deal with this problem.\x20
Please answer the question with one option that best matches the question (replay with 'O1', 'O2', 'O3', or 'O4').\x20
Do not include any other text or explanations!!!
"""

CHOICE_PROMPT = """
This football question is "{question}". The four corresponding options are:
{options}

The processing through the multi-agent platform is as follows:
{process}

Please provide your answer:
"""


def load_toolbox():
    """Mo ta ca 19 tool tu toolbox.csv cua code goc, dung dinh dang cua ham load_toolbox goc."""
    path = PROJECT_ROOT / load_config("models")["unisoccer"]["code_dir"] / "toolbox.csv"
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f, quotechar='"', skipinitialspace=True), start=1):
            out.append(f"=== Tool Description for TOOL{i} ===\nName: {row['name']}\nAbility: {row['ability']}\n"
                       f"Query Input: {row['query input']}\nmaterial Input: {row['material input']}\n"
                       f"Output: {row['output']}\nRemark: {row['remark']}\n")
    return "".join(f"{d}\n" for d in out)


def options_str(sample):
    return "\n".join(f"{k}: {sample.option_texts[k]}" for k in sample.option_keys)


def parse_call(reply):
    out = []
    for tag in ("Tool", "Query", "Material"):
        m = re.search(f"<{tag}>(.*?)</{tag}>", reply, re.DOTALL)
        out.append(m.group(1).strip() if m else None)
    return tuple(out)


class B1Vanilla:
    name = "b1_vanilla"

    def __init__(self, backend="qwen_local_3b", llm=None, cache=None):
        """backend: VLM cho cac tool nhin hinh; llm: bo thuc thi, LLM, Choice Selection (mac dinh cung backend)."""
        self.cfg = load_config()["b1"]
        self.vlm = build_backend(backend)
        self.llm = build_backend(llm) if llm else self.vlm
        self.tools = Toolbox(self.vlm, self.llm, cache or PerceptionCache(), ActionClassifier())
        self.toolbox = load_toolbox()

    def route(self, tool, k, sample, scenes):
        """File dua vao tool thu k cung loai trong chuoi (bo thuc thi goc tu chon, o day co dinh)."""
        if tool == "Foul Recognition":
            return list(sample.materials)
        if tool == "Camera Detection":
            if scenes:
                return [scenes[0] if k == 0 else scenes[-1]]
            return [sample.materials[0]]
        if tool == "Commentary Generation":
            return [sample.materials[k]]
        return [sample.materials[0]]

    def chat(self, messages, n, stop=None, **kw):
        return self.llm.chat(messages, max_new_tokens=n, stop_strings=stop, **kw)

    def exec_call(self, messages):
        # prompt lookup: nhanh ~2 lan, ket qua giong het giai ma tham lam (da kiem 8/8 luot)
        return self.chat(messages, self.cfg["exec_tokens"], CALL_END, prompt_lookup_num_tokens=self.cfg["prompt_lookup"])

    def run(self, sample):
        chain = CHAINS[sample.task_id]
        material = [rel_path(p) for p in sample.materials]
        prompt = EXEC_PROMPT.format(query=sample.question, material=material, options=options_str(sample),
                                    known=["VideoClip"], chain=chain, toolbox=self.toolbox)
        messages = [{"role": "system", "content": EXEC_SYSTEM}, {"role": "user", "content": prompt}]
        process, steps, scenes, seen = "", [], [], {}

        for tool in chain[:-1]:
            reply = self.exec_call(messages)
            messages.append({"role": "assistant", "content": reply})
            process += reply
            exec_tool, query, exec_mat = parse_call(reply)
            k = seen.get(tool, 0)
            seen[tool] = k + 1
            mats = self.route(tool, k, sample, scenes)
            try:
                result = self.tools(tool, query if query else sample.question, mats)
                if isinstance(result, tuple):
                    result, scenes = result
            except Exception as e:
                result = f"Error occurred: {e}"
            step = f"<StepResult>\n    <Answer>{result}</Answer>\n</StepResult>"
            messages.append({"role": "user", "content": step})
            process += step
            steps.append({"tool": tool, "query": query, "exec_tool": exec_tool, "exec_material": exec_mat,
                          "material": [m.name if isinstance(m, Scene) else rel_path(m) for m in mats],
                          "result": result})

        # buoc cuoi: bo thuc thi goi LLM (<EndCall>), LLM tra loi dua tren toan bo hoi thoai
        reply = self.exec_call(messages)
        messages.append({"role": "assistant", "content": reply})
        process += reply
        _, query, _ = parse_call(reply)
        messages.append({"role": "assistant", "content": LLM_PROMPT.format(query=query or sample.question)})
        answer = self.chat(messages, self.cfg["answer_tokens"])
        process += answer
        steps.append({"tool": "LLM", "query": query, "result": answer})
        return process, steps

    def __call__(self, sample):
        process, steps = self.run(sample)
        prompt = CHOICE_PROMPT.format(question=sample.question, options=options_str(sample), process=process)
        msgs = [{"role": "system", "content": CHOICE_SYSTEM}, {"role": "user", "content": prompt}]
        lp = self.llm.score_chat(msgs, sample.option_keys)
        ev = build_categorical(self.name, sample, lp, raw_obs=steps[-1]["result"])
        return {"pred": ev.pred, "dist": ev.dist, "confidence": ev.confidence, "evidence": [ev.to_json()],
                "steps": steps, "process": process}
