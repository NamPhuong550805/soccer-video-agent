"""Reasoner: gop cac Evidence, chua du tin thi tu chon hanh dong hoi them; chinh sach bang luat hoac bang LLM."""
from dataclasses import dataclass, field, replace

import numpy as np

from sva.baselines import B0Local
from sva.config import load_config
from sva.evidence import Evidence
from sva.models import LETTERS, margin_confidence, mcq_prompt, softmax


# ---------------------------------------------------------------- Reasoner (chinh sach bang luat)
@dataclass
class State:
    evidences: list
    dist: dict
    confidence: float
    used: list = field(default_factory=list)

    @property
    def top2(self):
        return sorted(self.dist, key=self.dist.get, reverse=True)[:2]

    @property
    def conflict(self):
        return len({e.pred for e in self.evidences if not e.failed}) > 1


class Reasoner:
    def __init__(self, actions, theta=0.15, max_steps=3, pair_mass=0.7):
        self.actions = actions
        self.theta = theta
        self.max_steps = max_steps
        self.pair_mass = pair_mass

    @staticmethod
    def aggregate(evidences, keys):
        """Trung binh cac phan phoi, Evidence loi (failed) co trong so 0."""
        ok = [e for e in evidences if not e.failed]
        if not ok:
            return {k: 1 / len(keys) for k in keys}
        return {k: float(np.mean([e.dist[k] for e in ok])) for k in keys}

    def state(self, sample, evidences, used):
        dist = self.aggregate(evidences, sample.option_keys)
        return State(evidences, dist, margin_confidence(dist), used)

    def choose(self, sample, state):
        """Chon hanh dong theo trang thai; tra ve (action, ly do) hoac (None, ly do dung)."""
        free = [a for a in self.actions if a.name not in state.used and a.applicable(sample, state)]
        if not free:
            return None, "het hanh dong"

        def pick(kind):
            return next((a for a in free if a.kind == kind), None)

        a, b = state.top2
        if state.conflict and pick("reframe"):
            return pick("reframe"), "cac nguon chon khac nhau -> hoi lai theo cach khac"
        if state.dist[a] + state.dist[b] >= self.pair_mass and pick("narrow"):
            return pick("narrow"), f"phan van giua {a} va {b} -> hoi rieng 2 phuong an"
        if pick("more_info"):
            return pick("more_info"), "phan van dan trai -> lay them thong tin"
        return free[0], "thu hanh dong con lai"

    def run(self, sample, evidences):
        evidences, used = list(evidences), []
        steps = []
        while True:
            st = self.state(sample, evidences, used)
            steps.append({"dist": st.dist, "confidence": st.confidence})
            if st.confidence >= self.theta:
                stop = "du tin"
                break
            if len(used) >= self.max_steps:
                stop = "het luot"
                break
            action, reason = self.choose(sample, st)
            if action is None:
                stop = reason
                break
            try:
                ev = action(sample, st)
            except Exception as e:
                ev = Evidence.failure(action.name, sample.option_keys, f"{type(e).__name__}: {e}")
            evidences.append(ev)
            used.append(action.name)
            steps[-1].update({"action": action.name, "reason": reason, "action_pred": ev.pred})

        return {"pred": max(st.dist, key=st.dist.get), "dist": st.dist, "confidence": st.confidence,
                "stop": stop, "steps": steps, "evidence": [e.to_json() for e in evidences]}


def replay(rec, theta):
    """Tu ket qua chay voi theta = 1 (luu dist sau tung buoc), suy ra ket qua neu dung theta khac.
    Dung duoc vi chinh sach chon hanh dong khong phu thuoc theta. Tra ve (pred, so hanh dong da dung)."""
    for i, st in enumerate(rec["steps"]):
        if st["confidence"] >= theta or i == len(rec["steps"]) - 1:
            return max(st["dist"], key=st["dist"].get), i


def choose_theta(records_by_task, grid=(0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.01)):
    """Chon theta cho accuracy trung binh (deu theo task) cao nhat tren dev; hoa thi lay theta nho hon (it buoc hon)."""
    def acc(theta):
        per_task = [np.mean([replay(r, theta)[0] == r["gold"] for r in recs])
                    for recs in records_by_task.values() if recs]
        return float(np.mean(per_task))
    scores = {t: acc(t) for t in grid}
    best = max(grid, key=lambda t: (round(scores[t], 6), -t))
    return best, scores


# ---------------------------------------------------------------- Hanh dong dung VLM
class MoreFrames:
    """Lay gap doi so frame roi cham lai."""
    name, kind = "more_frames", "more_info"

    def __init__(self, scorer, n_single=16, budget=32):
        self.scorer, self.n_single, self.budget = scorer, n_single, budget

    def applicable(self, sample, state):
        return True

    def __call__(self, sample, state):
        return self.scorer.score(sample, n_single=self.n_single, budget=self.budget, tool=self.name)


class Reorder:
    """Hoi lai voi thu tu phuong an dao nguoc va nhan A-D, de giam thien vi vi tri."""
    name, kind = "reorder", "reframe"

    def __init__(self, scorer):
        self.scorer = scorer

    def applicable(self, sample, state):
        return True

    def __call__(self, sample, state):
        rev = replace(sample, option_keys=sample.option_keys[::-1])
        labels = list(LETTERS[:rev.n_options])
        return self.scorer.score(rev, prompt=mcq_prompt(rev, labels), labels=labels, tool=self.name)


class Pairwise:
    """Chi hoi 2 phuong an dang dan dau (dat phuong an xep sau len truoc), roi chia lai xac suat cua 2 phuong an do."""
    name, kind = "pairwise", "narrow"

    def __init__(self, scorer):
        self.scorer = scorer

    def applicable(self, sample, state):
        return sample.n_options > 2

    def __call__(self, sample, state):
        a, b = state.top2
        sub = replace(sample, option_keys=[b, a], option_texts={k: sample.option_texts[k] for k in (b, a)})
        q = self.scorer.score(sub, prompt=mcq_prompt(sub, ["A", "B"]), labels=["A", "B"], tool=self.name).dist
        mass = state.dist[a] + state.dist[b]
        dist = dict(state.dist)
        dist[a], dist[b] = mass * q[a], mass * q[b]
        return Evidence(self.name, "categorical", sample.option_keys, dist, margin_confidence(dist),
                        f"{a} vs {b}: {q[a]:.2f} / {q[b]:.2f}")


def vlm_actions(scorer):
    return [Reorder(scorer), Pairwise(scorer), MoreFrames(scorer)]


class B0Loop:
    """B0 + vong lap agent: buoc dau giong B0-local, sau do Reasoner tu quyet dinh co hoi them khong."""

    def __init__(self, backend="qwen_local_3b", theta=None, max_steps=None, name="b0_loop"):
        cfg = load_config()["reasoner"]
        self.name = name
        self.b0 = B0Local(backend)
        self.reasoner = Reasoner(vlm_actions(self.b0), theta if theta is not None else cfg["theta_fallback"],
                                 max_steps or cfg["max_steps"])

    def __call__(self, sample):
        return self.reasoner.run(sample, [self.b0.score(sample)])


# ---------------------------------------------------------------- Controller: LLM chon buoc, mo rong doan video
ACTION_TEXT = {"reorder": "ask the video model again with the options in reverse order",
               "pairwise": "ask the video model to compare only the two leading options",
               "more_frames": "ask the video model again with twice as many frames",
               "expand": "look at a wider part of the video (longer segment or more candidate clips)"}


def top(dist, k=2):
    return sorted(dist, key=dist.get, reverse=True)[:k]


def describe(sample, state):
    """Bang trang thai bang chu cho LLM: cau hoi, phuong an, tung bang chung, phan phoi gop."""
    lines = [f"Question: {sample.question.strip()}", "Options:"]
    lines += [f"{k}. {sample.option_texts[k]}" for k in sample.option_keys]
    lines.append("Evidence so far:")
    for e in state.evidences:
        a, b = top(e.dist)
        lines.append(f"- {e.tool_name}: prefers {a} ({e.dist[a]:.2f}), then {b} ({e.dist[b]:.2f})")
    a, b = state.top2
    lines.append(f"Combined: {a} {state.dist[a]:.2f}, {b} {state.dist[b]:.2f} (margin {state.confidence:.2f})")
    return "\n".join(lines)


class LLMPolicyReasoner(Reasoner):
    def __init__(self, actions, backend, max_steps=3, pair_mass=0.7):
        super().__init__(actions, theta=1.01, max_steps=max_steps, pair_mass=pair_mass)   # LLM tu quyet khi nao dung
        self.backend = backend

    def choose(self, sample, state):
        free = [a for a in self.actions if a.name not in state.used and a.applicable(sample, state)]
        if not free:
            return None, "het hanh dong"
        choices = [f"stop and answer {state.top2[0]}"] + [f"{a.name}: {ACTION_TEXT.get(a.name, a.name)}" for a in free]
        prompt = (describe(sample, state) + "\nWhat should be done next to answer the question correctly?\n"
                  + "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, choices)) + "\nAnswer with the letter only.")
        p = softmax(self.backend.score_chat([{"role": "user", "content": prompt}], list(LETTERS[:len(choices)])))
        i = int(p.argmax())
        if i == 0:
            return None, f"llm: dung ({p[0]:.2f})"
        return free[i - 1], f"llm: {free[i - 1].name} ({p[i]:.2f})"


class Expand:
    """Chay lai nhanh tren doan rong hon / them ung vien (chi khi Grounding biet mo rong, tuc la video that)."""
    name, kind = "expand", "more_info"

    def __init__(self, grounding, run_branch):
        self.grounding, self.run_branch = grounding, run_branch

    def applicable(self, sample, state):
        return self.grounding is not None and self.grounding.can_expand(sample)

    def __call__(self, sample, state):
        wider = self.grounding.expand(sample)
        ev = self.run_branch(wider)[-1]
        ev.tool_name = self.name
        return ev
