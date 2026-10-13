"""B1-vanilla: tai hien SoccerAgent (MCQ) tren 6 task video.

Chuoi tool lay theo chuoi khuyen nghi (tasks.csv); bo thuc thi (LLM) viet <Query> cho tung tool va chi
thay cau hoi qua prompt cua no; tool chi nhan <Query> + file; buoc cuoi LLM tra loi roi Choice Selection
chon option tu toan bo qua trinh. Prompt giu nguyen van code goc, ca loi chinh ta; dau cach cuoi dong
viet la \x20 de editor khong tu xoa.
"""
import csv
import re

from sva.baselines.b1_tools import Scene, Toolbox
from sva.config import load_config
from sva.constants import PROJECT_ROOT
from sva.evidence.builder import build_categorical, rel_path
from sva.models.base import build_backend
from sva.perception.cache import PerceptionCache

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
        from sva.models.unisoccer import ActionClassifier
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
        # bo thuc thi chep lai nhieu doan trong prompt (duong dan, cau hoi): prompt lookup doan truoc cac doan do,
        # ket qua van giong het giai ma tham lam (da kiem 8/8 luot), nhanh hon ~2 lan
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
