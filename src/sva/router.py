"""Chia mau vao 3 nhanh bang luat: tang lap ke hoach khong phai nut that (paper B.3), khong can LLM."""
from sva.constants import TASK_META

BRANCH_A_TASKS = {"q10", "q11"}     # don clip, da co cong cu chuyen dung tot


def route(sample):
    if sample.task_id in BRANCH_A_TASKS:
        return "A"
    if len(sample.materials) > 1:   # nhieu clip: dap an nam o quan he giua cac clip
        return "C"
    return "B"


def router_accuracy(samples):
    """Ty le mau duoc chia dung nhanh theo bang task (metric kiem soat)."""
    return sum(route(s) == TASK_META[s.task_id]["branch"] for s in samples) / len(samples)
