"""Controller: Reasoner voi chinh sach do LLM chon (thay luat), va hanh dong "mo rong" cho video that.

Chinh sach luat (Reasoner goc): du tin (>= theta) thi dung; mau thuan -> hoi lai; phan van 2 phuong an -> hoi cap.
Chinh sach LLM: moi vong, LLM doc bang bang chung (nguon nao chon gi, chac bao nhieu) roi tu chon
"dung" hoac mot hanh dong con dung duoc. Khong co nguong theta.
"""
from sva.models.scoring import softmax
from sva.reasoner import Reasoner

LETTERS = "ABCDEFGH"
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
