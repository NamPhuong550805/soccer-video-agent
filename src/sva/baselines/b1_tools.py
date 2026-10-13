"""Cac tool cua SoccerAgent dung cho 6 task video, giu dung prompt va dinh dang dau ra cua code goc.

Tool nhan (query, material) va tra ve chuoi. VLM/LLM ben trong deu la cung backbone voi B0.
"""
import re
from collections import Counter
from dataclasses import dataclass

import cv2

from sva.config import load_config
from sva.constants import PROJECT_ROOT
from sva.perception.video import letterbox, probe, sample_frames

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
