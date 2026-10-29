"""SVA-Agent: hoi video that (ask) va cac danh gia E0-E6, U2 (moi phan ghi results/agent/<phan>.json).

    python scripts/07_agent.py ask --video highlight.mp4 --question "What happens at 1:10?"
    python scripts/07_agent.py ask --video main.mp4 closeup.mp4 --question "Did the foul involve a handball?" --options No Yes
    python scripts/07_agent.py router       # E0/E1: dinh tuyen (luat / LLM / ket hop) + cau ngoai pham vi
    python scripts/07_agent.py paraphrase   # E1: cau hoi viet lai (LLM viet lai, chay lai SVA tren q8, q13, q14)
    python scripts/07_agent.py open         # E2: tu sva_agent_open (kieu paper + kieu dap an mo)
    python scripts/07_agent.py long         # E3: video dai ghep tu clip test, hoi kem moc thoi gian (--task q14: chay lai q14)
    python scripts/07_agent.py format       # E4: doi fps / do phan giai / them logo / cat doc
    python scripts/07_agent.py abstain      # E6: nguong tu choi chon tren dev, do tren test
    python scripts/07_agent.py policy       # U2: Reasoner luat vs LLM (dev, test)
    python scripts/07_agent.py table        # gop -> results/tables/agent_test.md

ask ghi cau tra loi, do tin, doan video, bang chung, frame vao results/agent/ask/<thoi diem>/.
E0 (an task_id, chay lai ca test): python scripts/03_run.py --system sva_agent --theta 0.2 --task all
"""
import argparse
import json
import logging
import random
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sva.config import CACHE_DIR, RESULTS_DIR, TASK_IDS, record_run, setup_logging
from sva.data import canon
from sva.eval import accuracy, correct, load_predictions, load_split, mcnemar

log = logging.getLogger(__name__)
OUT = RESULTS_DIR / "agent"
LONG_DIR = CACHE_DIR / "agent" / "long"
THETA = 0.2                                     # theta cua SVA chon tren dev (sva_trace)
OUT_OF_SCOPE = [
    "Who scored the most goals in this season?", "What is the final score of this match?",
    "How many players are on the pitch in this image?", "Who is the referee of this game?",
    "What is the name of the player wearing number 10?", "Which stadium is this match played in?",
    "When was this football club founded?", "How old is the goalkeeper?", "Which team won the league last year?",
    "What is the offside rule in football?", "How many yellow cards did this team get this season?",
    "What time does the second half start?", "Who is the coach of the home team?",
    "What is the weather like during this match?", "How much is this player's transfer fee?",
    "Which country does the striker play for?", "What is the current minute shown on the scoreboard?",
    "How many goals has this player scored in the Champions League?", "Is this match broadcast on television?",
    "What brand makes the team's kit?", "Who won the Ballon d'Or in 2018?", "How long is a football match?",
    "What is the capacity of this stadium?", "Which league is this match from?",
    "What is the jersey number of the captain?", "How many substitutions are allowed in a match?",
    "Who was the man of the match?", "What was the attendance at this game?",
    "Which player has the most assists?", "What formation is the away team using?"]


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load(name):
    f = OUT / f"{name}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def backend():
    from sva.models import build_backend
    return build_backend("qwen_local_3b")


# ---------------------------------------------------------------- ask: hoi video that
def ask(args):
    """Hoi SVA-Agent ve video that (cau hoi tieng Anh, co hoac khong co phuong an)."""
    from sva.agent import SVAAgent
    out_dir = RESULTS_DIR / "agent" / "ask" / f"{datetime.now():%Y%m%d_%H%M%S}"
    options = {f"O{i + 1}": t for i, t in enumerate(args.options)} if args.options else None
    r = SVAAgent(args.backend).ask(args.video, args.question, options, out_dir)
    log.info("\n%s\n", r["answer"])
    log.info("trang thai %s | task %s (%s) | do tin %s | doan %s", r["status"], r["task"], r["route"],
             f"{r['confidence']:.2f}" if "confidence" in r else "-", r.get("segments"))
    for e in r.get("evidence", []):
        log.info("  %-18s %s", e["tool_name"], {k: round(v, 2) for k, v in e["dist"].items()})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "answer.json").write_text(json.dumps({"question": args.question, "videos": args.video, **r},
                                                    ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return f"{r['status']}: {r['answer'][:120]}"


# ---------------------------------------------------------------- E0 / E1: dinh tuyen
def part_router():
    from sva.agent import QuestionAgent, rule_task
    qa = QuestionAgent(backend())
    res = {}
    for split in ("dev", "test"):
        rows = {"luat (so clip that)": [], "luat (1 video)": [], "chi LLM": [], "luat + LLM": []}
        for t in TASK_IDS:
            for s in load_split(t, split):
                rows["luat (so clip that)"].append(rule_task(s.question, len(s.materials), s.option_texts)[0] == t)
                rows["luat (1 video)"].append(rule_task(s.question, 1, s.option_texts)[0] == t)
                d = qa.llm_dist(s.question, s.option_texts)
                rows["chi LLM"].append(max(d, key=d.get) == t)
                rows["luat + LLM"].append(qa(s.question, len(s.materials), s.option_texts).task == t)
        res[split] = {k: [int(sum(v)), len(v)] for k, v in rows.items()}
        log.info("%s %s", split, {k: f"{100 * a / n:.2f}%" for k, (a, n) in res[split].items()})
    out_rows = []
    for text in OUT_OF_SCOPE:
        rule = rule_task(text)[0]
        intent = qa(text, check_scope=True)
        out_rows.append({"question": text, "rule": rule, "agent": intent.task, "p_out": round(intent.dist.get("out", 0), 3)})
    res["out_of_scope"] = out_rows
    n_ok = sum(r["agent"] == "out" for r in out_rows)
    log.info("cau ngoai pham vi: agent tu choi dung %d/%d", n_ok, len(out_rows))
    save("router", res)
    return f"dinh tuyen test: luat {res['test']['luat (so clip that)']}, chi LLM {res['test']['chi LLM']}; ngoai pham vi {n_ok}/{len(out_rows)}"


# ---------------------------------------------------------------- E1: cau hoi viet lai
PARA_PROMPT = ("Rewrite the following question about a football video so that it uses different words but asks exactly "
               "the same thing. Do not answer it and do not add options. Question: {q}\nRewritten question:")


def part_paraphrase(per_task=40, seed=0):
    from sva.agent import QuestionAgent
    from sva.branches import question_part
    from sva.pipeline import SVA
    rng = random.Random(seed)
    sva = SVA("qwen_local_3b", theta=THETA, name="sva_para")
    qa = QuestionAgent(sva.scorer.backend)
    old = {t: {r["index"]: r for r in load_predictions("sva", t, "test")} for t in TASK_IDS}
    rows = []
    for t in TASK_IDS:
        for s in rng.sample(load_split(t, "test"), per_task):
            main = question_part(s.question) if t == "q14" else s.question
            new = sva.scorer.backend.chat([{"role": "user", "content": PARA_PROMPT.format(q=main)}], max_new_tokens=60)
            new = new.strip().split("\n")[0].strip('" ')
            text = s.question.replace(main, new) if t == "q14" else new
            intent = qa(text, len(s.materials), s.option_texts)
            row = {"task": t, "index": s.index, "old": s.question, "new": text, "routed": intent.task,
                   "route": intent.source, "old_correct": correct(old[t][s.index])}
            if t in ("q8", "q13", "q14") and intent.task == t:     # cac task con lai khong dung chu cua cau hoi
                row["new_correct"] = correct({"pred": sva(replace(s, question=text))["pred"], "gold": s.gold_key})
            elif intent.task == t:
                row["new_correct"] = row["old_correct"]
            else:
                row["new_correct"] = False
            rows.append(row)
        log.info("%s xong", t)
    save("paraphrase", rows)
    r = [x["routed"] == x["task"] for x in rows]
    return f"viet lai {len(rows)} cau: dinh tuyen {100 * np.mean(r):.1f}%"


# ---------------------------------------------------------------- E2: Open
def strict_ok(t, r, s):
    """Kieu chat: so nhan tra loi voi dap an mo cua SoccerBench."""
    if not s.open_answer:
        return None
    gray = lambda x: canon(x).replace("grey", "gray")
    return gray(r["open_pred"]) == gray(s.open_answer)


def part_open():
    from sva.agent import rouge_l
    res = {}
    for t in TASK_IDS:
        recs = load_predictions("sva_agent_open", t, "test")
        if not recs:
            continue
        S = {s.index: s for s in load_split(t, "test")}
        mcq = {r["index"]: r for r in load_predictions("sva", t, "test")}
        same = [mcq[r["index"]] for r in recs]
        row = {"n": len(recs), "mcq": 100 * accuracy(same), "open_paper": 100 * accuracy(recs),
               "mcnemar": mcnemar(same, recs), "select": dict(zip(*np.unique([r.get("select") for r in recs], return_counts=True)))}
        if t == "q11":
            row["rouge_l"] = float(np.mean([rouge_l(r["open_pred"], S[r["index"]].open_answer) for r in recs]))
        else:
            ok = [strict_ok(t, r, S[r["index"]]) for r in recs]
            ok = [o for o in ok if o is not None]
            row["open_strict"] = 100 * float(np.mean(ok))
        row["select"] = {k: int(v) for k, v in row["select"].items()}
        res[t] = row
        log.info("%s %s", t, {k: v for k, v in row.items() if k != "mcnemar"})
    if "q11" in res:
        res["q11"]["event_match"] = q11_event_agreement()
    save("open", res)
    return "open: " + "; ".join(f"{t} paper {r['open_paper']:.1f}" for t, r in res.items())


def q11_event_agreement():
    """q11 kieu chat: loai su kien (LLM doc cau) cua cau sinh ra co trung voi cau that khong."""
    from sva.branches import BranchA
    from sva.models import ActionClassifier

    class Stub:
        class backend:
            name = "qwen_local_3b"
    a = BranchA(Stub(), ActionClassifier())
    S = {s.index: s for s in load_split("q11", "test")}
    recs = load_predictions("sva_agent_open", "q11", "test")
    try:
        same = [int(np.argmax(a.caption_events(r["open_pred"]))) == int(np.argmax(a.caption_events(S[r["index"]].open_answer)))
                for r in recs]
    except AttributeError:                       # cau chua co trong cache: can backend that
        a.scorer.backend = backend()
        same = [int(np.argmax(a.caption_events(r["open_pred"]))) == int(np.argmax(a.caption_events(S[r["index"]].open_answer)))
                for r in recs]
    return 100 * float(np.mean(same))


# ---------------------------------------------------------------- E3: video dai
def write_video(clips, out, size):
    """Noi cac clip thanh 1 video (25 fps, kich thuoc size); tra ve [(bat dau, ket thuc)] cua tung clip."""
    w, h = size
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))
    spans, t = [], 0.0
    for path in clips:
        cap = cv2.VideoCapture(str(path))
        n = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            vw.write(cv2.resize(f, (w, h)) if (f.shape[1], f.shape[0]) != (w, h) else f)
            n += 1
        cap.release()
        spans.append((t, t + n / 25.0))
        t += n / 25.0
    vw.release()
    return spans


def size_of(path):
    cap = cv2.VideoCapture(str(path))
    s = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return s


def build_long(seed=0):
    """Video ghep tu clip test (dap an da biet): q8/q10/q11/q13 moi video 5 clip; q9: 4 ung vien (xao) + replay o cuoi;
    q14: cac goc cua 1 pha noi tiep nhau."""
    from sva.agent import clock
    rng = random.Random(seed)
    LONG_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for t, n_videos, per in (("q8", 20, 5), ("q10", 20, 5), ("q11", 12, 5), ("q13", 20, 5)):
        pool = load_split(t, "test")
        pool = [s for s in pool if size_of(s.materials[0]) == size_of(pool[0].materials[0])]   # cung kich thuoc
        rng.shuffle(pool)
        for v in range(n_videos):
            group = pool[v * per:(v + 1) * per]
            out = LONG_DIR / f"{t}_{v}.mp4"
            spans = write_video([s.materials[0] for s in group], out, size_of(group[0].materials[0]))
            for s, (a, b) in zip(group, spans):
                mid = (a + b) / 2
                q = f"{s.question.rstrip()} Look at {clock(mid)} of the video."
                items.append({"task": t, "index": s.index, "video": str(out), "question": q, "span": [a, b],
                              "options": s.option_texts, "gold": s.gold_key})
    for v, s in enumerate(rng.sample(load_split("q9", "test"), 25)):
        cands = list(enumerate(s.materials[1:]))
        rng.shuffle(cands)
        out = LONG_DIR / f"q9_{v}.mp4"
        spans = write_video([p for _, p in cands] + [s.materials[0]], out, size_of(s.materials[0]))
        gold_pos = int(s.gold_key[1:]) - 1                     # vi tri ung vien dung trong danh sach goc
        gold_span = spans[[i for i, _ in cands].index(gold_pos)]
        q = f"Which earlier play is shown again in the replay at {clock(sum(spans[-1]) / 2)}?"
        items.append({"task": "q9", "index": s.index, "video": str(out), "question": q, "span": list(gold_span),
                      "options": None, "gold": s.gold_key})
    for v, s in enumerate(rng.sample(load_split("q14", "test"), 30)):
        out = LONG_DIR / f"q14_{v}.mp4"
        write_video(s.materials, out, size_of(s.materials[0]))
        from sva.branches import question_part
        items.append({"task": "q14", "index": s.index, "video": str(out), "question": f"{question_part(s.question)} Look at 0:00.",
                      "options": s.option_texts, "gold": s.gold_key, "span": None})
    save("long_items", items)
    return items


def part_long(tasks=None):
    """tasks: chi chay lai cac task nay, giu nguyen dong cua task khac trong long.json."""
    from sva.agent import SVAAgent
    from sva.perception import split_segment
    items = load("long_items") or build_long()
    agent = SVAAgent("qwen_local_3b", theta=THETA, name="sva_agent_long")
    old = {t: {r["index"]: r for r in load_predictions("sva", t, "test")} for t in TASK_IDS}
    keep = {(r["task"], r["index"]): r for r in (load("long") or [])} if tasks else {}
    rows = []
    for it in items:
        if tasks and it["task"] not in tasks:
            rows.append(keep[(it["task"], it["index"])])
            continue
        try:
            r = agent.ask([it["video"]], it["question"], it["options"])
        except Exception as e:                         # 1 cau loi khong lam hong ca phan
            r = {"status": "error", "answer": f"{type(e).__name__}: {e}"}
        row = {"task": it["task"], "index": it["index"], "status": r["status"], "routed": r.get("task"),
               "old_correct": correct(old[it["task"]][it["index"]])}
        if it["task"] == "q9":                          # dung neu giua doan duoc chon nam trong clip goc that
            _, a, b = split_segment(r["chosen"]) if r.get("chosen") else (None, None, None)
            row["correct"] = a is not None and it["span"][0] <= (a + b) / 2 <= it["span"][1]
        else:
            row["correct"] = r.get("key") == it["gold"]
            if it["span"] and r.get("segments"):
                _, a, b = split_segment(r["segments"][0])
                if a is not None:
                    inter = max(0.0, min(b, it["span"][1]) - max(a, it["span"][0]))
                    row["iou"] = inter / (max(b, it["span"][1]) - min(a, it["span"][0]))
        rows.append(row)
    save("long", rows)
    return "long: " + "; ".join(f"{t} {100 * np.mean([r['correct'] for r in rows if r['task'] == t]):.1f}"
                                for t in ("q8", "q9", "q10", "q11", "q13", "q14"))


# ---------------------------------------------------------------- E4: dinh dang khac
def variant(path, kind, out):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    h, w = frames[0].shape[:2]
    if kind == "fps50":
        frames, fps, size = [f for f in frames for _ in (0, 1)], fps * 2, (w, h)
    elif kind == "hd720":
        frames, size = [cv2.resize(f, (1280, 720)) for f in frames], (1280, 720)
    elif kind == "overlay":
        def draw(f):
            g = f.copy()
            cv2.rectangle(g, (5, 5), (130, 30), (20, 20, 20), -1)
            cv2.putText(g, "HOME 1-0 AWAY 63:12", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.putText(g, "LIVE", (w - 45, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            return g
        frames, size = [draw(f) for f in frames], (w, h)
    else:                                                    # cat doc 9:16 o giua (kieu video ngan tren mang xa hoi)
        cw = int(h * 9 / 16)
        x = (w - cw) // 2
        frames, size = [np.ascontiguousarray(f[:, x:x + cw]) for f in frames], (cw, h)   # VideoWriter can mang lien
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for f in frames:
        vw.write(f)
    vw.release()


def part_format(per_task=40, seed=0):
    from sva.pipeline import SVA
    rng = random.Random(seed)
    sva = SVA("qwen_local_3b", theta=THETA, name="sva_format")
    old = {t: {r["index"]: r for r in load_predictions("sva", t, "test")} for t in ("q8", "q10")}
    rows = []
    for t in ("q8", "q10"):                                  # clip ngang 398x224 (khong dinh dang bop)
        pool = [s for s in load_split(t, "test") if size_of(s.materials[0]) == (398, 224)]
        for s in rng.sample(pool, per_task):
            row = {"task": t, "index": s.index, "orig": correct(old[t][s.index])}
            for kind in ("fps50", "hd720", "overlay", "portrait"):
                out = CACHE_DIR / "agent" / "format" / kind / f"{t}_{s.index}.mp4"
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists():
                    variant(s.materials[0], kind, out)
                row[kind] = correct({"pred": sva(replace(s, materials=[str(out)]))["pred"], "gold": s.gold_key})
            rows.append(row)
        log.info("%s xong", t)
    save("format", rows)
    return "format: " + "; ".join(f"{k} {100 * np.mean([r[k] for r in rows]):.1f}" for k in ("orig", "fps50", "hd720", "overlay", "portrait"))


# ---------------------------------------------------------------- E6: tu choi
def part_abstain(target=0.85):
    """Nguong tren xac suat cao nhat: chon tren dev (gop 6 task) de cau duoc tra loi dung >= target, do tren test."""
    def pool(split):
        return [(max(r["dist"].values()), correct(r)) for t in TASK_IDS for r in load_predictions("sva_agent", t, split)]
    dev, test = pool("dev"), pool("test")
    if not dev:
        raise RuntimeError("chua co sva_agent tren dev: chay 03_run.py --system sva_agent --split dev --theta 0.2")
    best = 0.0
    for th in sorted({round(p, 3) for p, _ in dev}):
        kept = [c for p, c in dev if p >= th]
        if kept and np.mean(kept) >= target:
            best = th
            break
    res = {"threshold": best, "target": target}
    for name, data in (("dev", dev), ("test", test)):
        kept = [c for p, c in data if p >= best]
        res[name] = {"coverage": 100 * len(kept) / len(data), "acc_answered": 100 * float(np.mean(kept)),
                     "acc_all": 100 * float(np.mean([c for _, c in data]))}
    log.info("%s", res)
    save("abstain", res)
    return f"tu choi: nguong {best} (dev); test tra loi {res['test']['coverage']:.1f}% cau, dung {res['test']['acc_answered']:.1f}%"


# ---------------------------------------------------------------- U2: luat vs LLM
def part_policy():
    from sva.reasoner import replay
    res = {}
    for t in ("q8", "q13"):
        trace = {r["index"]: r for r in load_predictions("sva_trace", t, "dev")}
        llm_dev = load_predictions("sva_llm", t, "dev")
        rule_dev = [{"index": i, "gold": r["gold"], "pred": replay(r, THETA)[0]} for i, r in trace.items()]
        row = {"dev_rule": 100 * accuracy(rule_dev), "dev_llm": 100 * accuracy(llm_dev) if llm_dev else None,
               "dev_steps_rule": float(np.mean([replay(r, THETA)[1] for r in trace.values()])),
               "dev_steps_llm": float(np.mean([len(r["steps"]) - 1 for r in llm_dev])) if llm_dev else None}
        llm_test = load_predictions("sva_llm", t, "test")
        if llm_test:
            rule_test = load_predictions("sva", t, "test")
            row.update(test_rule=100 * accuracy(rule_test), test_llm=100 * accuracy(llm_test),
                       test_mcnemar=mcnemar(rule_test, llm_test),
                       test_steps_llm=float(np.mean([len(r["steps"]) - 1 for r in llm_test])))
        res[t] = row
        log.info("%s %s", t, row)
    save("policy", res)
    return "policy: " + "; ".join(f"{t} dev luat {r['dev_rule']:.1f} / LLM {r['dev_llm']}" for t, r in res.items())


# ---------------------------------------------------------------- bang
def part_table():
    f = lambda x: "—" if x is None else f"{x:.1f}".replace(".", ",")
    L = ["# SVA-Agent: kết quả đánh giá (bản 3B, tập test)", ""]
    e0 = {t: (load_predictions("sva_agent", t, "test"), load_predictions("sva", t, "test")) for t in TASK_IDS}
    if all(a for a, _ in e0.values()):
        L += ["## E0 · Chế độ benchmark, giấu tên task", "",
              "| Task | n | task đoán đúng | SVA (biết task) | SVA-Agent (tự đoán) | dự đoán trùng | b / c |", "|---|---|---|---|---|---|---|"]
        for t, (a, b) in e0.items():
            B = {r["index"]: r for r in b}
            same = np.mean([r["pred"] == B[r["index"]]["pred"] for r in a])
            m = mcnemar(b, a)
            L.append(f"| {t} | {len(a)} | {100 * np.mean([r['task_pred'] == t for r in a]):.1f}% | {f(100 * accuracy(b))} | "
                     f"{f(100 * accuracy(a))} | {100 * same:.1f}% | {m['b']} / {m['c']} |")
        L.append("")
    r = load("router")
    if r:
        L += ["## E0 / E1 · Định tuyến", "", "| cách | dev | test |", "|---|---|---|"]
        for k in r["dev"]:
            (a1, n1), (a2, n2) = r["dev"][k], r["test"][k]
            L.append(f"| {k} | {100 * a1 / n1:.2f}% ({a1}/{n1}) | {100 * a2 / n2:.2f}% ({a2}/{n2}) |")
        oos = r["out_of_scope"]
        L += ["", f"Câu ngoài phạm vi ({len(oos)} câu tự viết): Question Agent từ chối đúng **{sum(x['agent'] == 'out' for x in oos)}/{len(oos)}**; "
              f"chỉ luật thì {sum(x['rule'] is None for x in oos)}/{len(oos)} câu không khớp luật nào.", ""]
    p = load("paraphrase")
    if p:
        L += ["## E1 · Câu hỏi viết lại (LLM viết lại, 40 câu test mỗi task)", "",
              "| Task | định tuyến đúng | đúng (câu gốc) | đúng (câu viết lại) |", "|---|---|---|---|"]
        for t in TASK_IDS:
            rows = [x for x in p if x["task"] == t]
            L.append(f"| {t} | {100 * np.mean([x['routed'] == t for x in rows]):.1f}% | "
                     f"{f(100 * np.mean([x['old_correct'] for x in rows]))} | {f(100 * np.mean([x['new_correct'] for x in rows]))} |")
        L.append("")
    o = load("open")
    if o:
        L += ["## E2 · Chế độ Open (không cho thấy phương án)", "",
              "| Task | n | MCQ (cùng câu) | Open kiểu paper (ánh xạ vào phương án) | Open kiểu chặt (khớp đáp án mở) | b / c (MCQ vs Open) |",
              "|---|---|---|---|---|---|"]
        for t, x in o.items():
            strict = f(x.get("open_strict")) if "open_strict" in x else f"sự kiện khớp {f(x.get('event_match'))}, ROUGE-L {x.get('rouge_l', 0):.2f}"
            L.append(f"| {t} | {x['n']} | {f(x['mcq'])} | {f(x['open_paper'])} | {strict} | {x['mcnemar']['b']} / {x['mcnemar']['c']} |")
        L.append("")
    lg = load("long")
    if lg:
        L += ["## E3 · Video dài (ghép từ clip test, câu hỏi kèm mốc thời gian)", "",
              "| Task | n | đúng (clip gốc, cắt sẵn) | đúng (video dài, agent tự tìm đoạn) | IoU đoạn | trạng thái |", "|---|---|---|---|---|---|"]
        for t in ("q8", "q9", "q10", "q11", "q13", "q14"):
            rows = [x for x in lg if x["task"] == t]
            if not rows:
                continue
            iou = [x["iou"] for x in rows if "iou" in x]
            st = dict(zip(*np.unique([x["status"] for x in rows], return_counts=True)))
            L.append(f"| {t} | {len(rows)} | {f(100 * np.mean([x['old_correct'] for x in rows]))} | "
                     f"{f(100 * np.mean([x['correct'] for x in rows]))} | {np.mean(iou):.2f} | {', '.join(f'{k} {v}' for k, v in st.items())} |"
                     if iou else f"| {t} | {len(rows)} | {f(100 * np.mean([x['old_correct'] for x in rows]))} | "
                     f"{f(100 * np.mean([x['correct'] for x in rows]))} | — | {', '.join(f'{k} {v}' for k, v in st.items())} |")
        L.append("")
    fm = load("format")
    if fm:
        L += ["## E4 · Định dạng video khác (40 clip test mỗi task)", "",
              "| Task | gốc | fps ×2 | 1280×720 | thêm bảng tỉ số / logo | cắt dọc 9:16 |", "|---|---|---|---|---|---|"]
        for t in ("q8", "q10"):
            rows = [x for x in fm if x["task"] == t]
            L.append(f"| {t} | " + " | ".join(f(100 * np.mean([x[k] for x in rows])) for k in ("orig", "fps50", "hd720", "overlay", "portrait")) + " |")
        L.append("")
    a = load("abstain")
    if a:
        L += ["## E6 · Biết từ chối", "", f"Ngưỡng xác suất **{a['threshold']}** chọn trên dev (mục tiêu đúng ≥ {100 * a['target']:.0f}% trên câu trả lời).", "",
              "| | trả lời (% câu) | đúng trên câu trả lời | đúng nếu trả lời hết |", "|---|---|---|---|"]
        for k in ("dev", "test"):
            L.append(f"| {k} | {f(a[k]['coverage'])} | {f(a[k]['acc_answered'])} | {f(a[k]['acc_all'])} |")
        L.append("")
    pl = load("policy")
    if pl:
        L += ["## U2 · Reasoner: luật hay LLM ra quyết định", "",
              "| Task | dev luật | dev LLM | test luật | test LLM | b / c (test) | số bước TB (LLM, test) |", "|---|---|---|---|---|---|---|"]
        for t, x in pl.items():
            m = x.get("test_mcnemar")
            L.append(f"| {t} | {f(x['dev_rule'])} | {f(x['dev_llm'])} | {f(x.get('test_rule'))} | {f(x.get('test_llm'))} | "
                     f"{(str(m['b']) + ' / ' + str(m['c'])) if m else '—'} | {f(x.get('test_steps_llm'))} |")
        L.append("")
    out = RESULTS_DIR / "tables" / "agent_test.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log.info("\n".join(L))
    return f"bang {out.relative_to(RESULTS_DIR).as_posix()}"


def main():
    parts = {"ask": ask, "router": part_router, "paraphrase": part_paraphrase, "open": part_open, "long": part_long,
             "format": part_format, "abstain": part_abstain, "policy": part_policy, "table": part_table}
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=list(parts))
    parser.add_argument("--video", nargs="+", help="ask: 1 video (co the dai) hoac 2-5 video")
    parser.add_argument("--question", help="ask: cau hoi tieng Anh")
    parser.add_argument("--options", nargs="*", default=None, help="ask: 2-4 phuong an; bo trong = che do Open")
    parser.add_argument("--backend", default="qwen_local_3b", help="ask: ten cau hinh trong configs/models.yaml")
    parser.add_argument("--task", nargs="+", default=None, help="long: chi chay lai cac task nay")
    args = parser.parse_args()
    setup_logging(f"07_agent_{args.part}")
    if args.part == "ask":
        record_run(ask(args))
    else:
        record_run(parts[args.part](args.task) if args.part == "long" else parts[args.part]())


if __name__ == "__main__":
    main()
