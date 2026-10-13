"""Nhanh C (nhieu clip): q9 so khop replay bang hinh anh; q14 gop cac goc quay."""
from sva.branches.matching import ReplayMatcher
from sva.branches.multiview import MultiView


class BranchC:
    name = "branch_c"
    TASKS = {"q9", "q14"}

    def __init__(self, scorer, cache, matching, multiview=None, budget=24):
        mv = dict(multiview or {})
        self.scorer, self.with_holistic = scorer, mv.pop("with_holistic", False)
        self.matcher = ReplayMatcher(cache, **matching)
        self.multiview = MultiView(scorer, budget, **mv)

    def supports(self, task_id):
        return task_id in self.TASKS

    def run(self, sample):
        if sample.task_id == "q9":
            return [self.matcher(sample)]
        if sample.task_id == "q14":
            return ([self.scorer.score(sample)] if self.with_holistic else []) + [self.multiview(sample)]
        raise NotImplementedError(f"nhanh C chua co cho {sample.task_id}")
