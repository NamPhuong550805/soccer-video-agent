"""SVA: Router -> nhanh -> cac Evidence -> Reasoner (gop, chua du tin thi tu hoi them)."""
from sva.config import load_config
from sva.reasoner import Reasoner, vlm_actions
from sva.router import route


class SVA:
    def __init__(self, backend="qwen_local_3b", theta=None, name="sva", policy=None, grounding=None):
        """policy: "rule" (Reasoner goc) hoac "llm" (LLM chon buoc tiep theo); mac dinh theo config.
        grounding: co thi them hanh dong "expand" (chi dung duoc voi doan video that, khong voi clip benchmark)."""
        from sva.agent.controller import Expand, LLMPolicyReasoner
        from sva.baselines.b0_local import B0Local
        from sva.branches.branch_a import BranchA
        from sva.branches.branch_b import BranchB
        from sva.branches.branch_c import BranchC
        cfg = load_config()
        self.name = name
        self.scorer = B0Local(backend)
        self.branches = {"A": BranchA(self.scorer, **cfg["branch_a"]),
                         "B": BranchB(self.scorer, **cfg["branch_b"]),
                         "C": BranchC(self.scorer, self.scorer.cache, cfg["matching"], cfg["multiview"],
                                      cfg["perception"]["n_frames_total_multiview"])}
        r = cfg["reasoner"]
        theta = r["theta_fallback"] if theta is None else theta
        self.policy = policy or r["policy"]
        extra = [Expand(grounding, self.run_branch)] if grounding is not None else []

        def make(actions):
            if self.policy == "llm":
                return LLMPolicyReasoner(actions, self.scorer.backend, r["max_steps"])
            return Reasoner(actions, theta, r["max_steps"])
        self.reasoner = make(vlm_actions(self.scorer) + extra)
        self.no_actions = make(extra)
        self.no_vlm_actions = set(r["no_vlm_actions"])

    def supports(self, task_id):
        from sva.data.clean import load_clean
        branch = self.branches.get(route(load_clean(task_id)[0]))
        return branch is not None and getattr(branch, "supports", lambda t: True)(task_id)

    def run_branch(self, sample):
        return self.branches[route(sample)].run(sample)

    def __call__(self, sample):
        branch = route(sample)
        reasoner = self.no_actions if sample.task_id in self.no_vlm_actions else self.reasoner
        out = reasoner.run(sample, self.branches[branch].run(sample))
        return {"branch": branch, **out}
