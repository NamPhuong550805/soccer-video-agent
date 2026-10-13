"""Chay mot he thong tren cac task, ghi results/predictions/<system>_<task>.jsonl

    python scripts/05_run_system.py --system b0_local --task all
    python scripts/05_run_system.py --system b0_local --task q8 --split dev --limit 20
    python scripts/05_run_system.py --system b0_api --task all --workers 4
    python scripts/05_run_system.py --system b0_loop_trace --theta 1.0 --split dev --task all
    python scripts/05_run_system.py --system b1_vanilla --task q8 --split dev --limit 10
    python scripts/05_run_system.py --system b1_vanilla --task all --backend qwen_local   # 7B -> b1_vanilla_7b_*
    python scripts/05_run_system.py --system b0_loop --task all --backend qwen_local --theta auto
    python scripts/05_run_system.py --system sva_trace --theta 1.0 --split dev --task all   # roi:
    python scripts/05_run_system.py --system sva --theta auto --task all
    python scripts/05_run_system.py --system sva_agent --theta 0.2 --task all        # an task_id, Question Agent doan
    python scripts/05_run_system.py --system sva_llm --policy llm --split dev --task q8 q13
    python scripts/05_run_system.py --system sva_agent_open --theta 0.2 --task all   # an ca phuong an (kieu Open)
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sva.config import load_config, record_run, setup_logging
from sva.constants import TASK_IDS
from sva.eval.metrics import accuracy
from sva.eval.runner import load_predictions, result_name, run_system

log = logging.getLogger(__name__)


def theta_from_dev(system, backend):
    """Chon theta tren dev tu lan chay <system>_trace (theta = 1.0) cung backend, tren cac task da co trace."""
    from sva.eval.runner import load_predictions
    from sva.reasoner import choose_theta
    trace = {t: load_predictions(result_name(f"{system}_trace", backend), t, "dev") for t in TASK_IDS}
    trace = {t: r for t, r in trace.items() if r}
    if not trace:
        raise RuntimeError(f"chua co {system}_trace tren dev, chay voi --theta 1.0 --split dev truoc")
    theta, scores = choose_theta(trace)
    log.info("theta chon tren dev: %s (accuracy dev %s)", theta, {k: round(v, 4) for k, v in scores.items()})
    return theta


def build_system(name, backend, theta=None, llm=None, policy=None):
    if name == "b0_local":
        from sva.baselines.b0_local import B0Local
        system = B0Local(backend)
    elif name == "b0_api":
        from sva.baselines.b0_api import B0Api
        return B0Api()
    elif name == "b1_vanilla":
        from sva.baselines.b1_vanilla import B1Vanilla
        system = B1Vanilla(backend, llm)
    elif name.startswith("b0_loop"):
        from sva.reasoner import B0Loop
        system = B0Loop(backend, theta=theta, name=name)
    elif name.startswith("sva_agent"):          # Question Agent tu doan task (an task_id that)
        from sva.agent.app import SVAAgent
        system = SVAAgent(backend, theta=theta, name=name, policy=policy, open_mode="open" in name)
    elif name.startswith("sva"):
        from sva.pipeline import SVA
        system = SVA(backend, theta=theta, name=name, policy=policy)
    else:
        raise ValueError(f"khong co system {name!r}")
    system.name = result_name(system.name, backend, llm)
    return system


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument("--task", nargs="+", default=["all"])
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", default="qwen_local_3b", help="ten cau hinh trong configs/models.yaml")
    parser.add_argument("--llm", default=None, help="b1_vanilla: model cho vai tro LLM (mac dinh = backend)")
    parser.add_argument("--workers", type=int, default=1, help="so mau chay song song (chi dung cho API)")
    parser.add_argument("--policy", default=None, choices=["rule", "llm"],
                        help="sva*: chinh sach cua Reasoner (mac dinh theo config)")
    parser.add_argument("--theta", default=None,
                        help="nguong do tin cua b0_loop; 1.0 = luon chay het max_steps (de chon theta tren dev); "
                             "auto = chon tren dev tu <system>_trace cung backend")
    args = parser.parse_args()

    setup_logging(f"05_run_{args.system}")
    tasks = TASK_IDS if args.task == ["all"] else args.task
    theta = theta_from_dev(args.system, args.backend) if args.theta == "auto" else (
        float(args.theta) if args.theta else None)
    system = build_system(args.system, args.backend, theta, args.llm, args.policy)
    if hasattr(system, "supports"):          # SVA: chi chay task co nhanh da cai
        skipped = [t for t in tasks if not system.supports(t)]
        tasks = [t for t in tasks if t not in skipped]
        if skipped:
            log.info("bo qua %s: nhanh chua cai", skipped)
    summary = []
    for task_id in tasks:
        path = run_system(system, task_id, args.split, args.limit, args.workers)
        recs = load_predictions(system.name, task_id, args.split)
        log.info("%s %s -> %s (%d mau, accuracy %.1f)", args.system, task_id, path, len(recs), 100 * accuracy(recs))
        summary.append(f"{task_id} {100 * accuracy(recs):.1f} (n={len(recs)})")
    record_run(f"{system.name} {args.split}" + (f" theta={theta}" if theta is not None else "") + ": " + "; ".join(summary))


if __name__ == "__main__":
    main()
