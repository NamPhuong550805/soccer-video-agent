"""Bang tong ket ban 3B tu results/predictions/ (khong chay model):

    python scripts/11_report.py                          # ban 3B
    python scripts/11_report.py --backend qwen_local     # ban 7B -> cac bang co hau to _7b

results/tables/main_results.md   4 moc tren 6 task, McNemar SVA vs B1, so cua paper (khac giao thuc)
results/tables/cost_time.md      giay / cau, so luot goi model, thoi gian tinh truoc
results/tables/risk_coverage.md  chi tra loi x% cau chac nhat thi dung bao nhieu; ECE
results/tables/sva_gemini.md     Gemini hoi thang vs Gemini + cac nhanh cua SVA (ban lai, khong goi API them)
results/tables/case_studies.md   ~30 ca: B1 sai SVA dung, nguoc lai, ca hai sai, kem chuoi bang chung
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sva.config import load_config, record_run, setup_logging
from sva.constants import RESULTS_DIR, TASK_IDS, TASK_META
from sva.eval.metrics import accuracy, correct, ece
from sva.eval.runner import load_predictions, load_split, result_name
from sva.eval.stats import mcnemar

log = logging.getLogger(__name__)
TABLES = RESULTS_DIR / "tables"
RUN = {"backend": "qwen_local_3b", "suffix": "", "model": "Qwen2.5-VL-3B 4-bit"}


def res(name):
    """Ten file ket qua cua he theo backend dang bao cao (B0-api la Gemini, khong doi)."""
    return name if name == "b0_api" else result_name(name, RUN["backend"])
SYSTEMS = {"B0-local": "b0_local", "B0 + vòng lặp": "b0_loop", "B0-api (Gemini)": "b0_api", "B1-vanilla": "b1_vanilla",
           "SVA": "sva"}
PAPER = {"Qwen2.5-VL-7B": [56.8, 31.6, 52.2, 51.6, 46.9, 50.7], "SoccerAgent": [51.1, 35.7, 85.0, 72.9, 46.0, 55.5]}
PRECOMPUTE = {"q9": "embedding CLIP 5 clip × 24 frame ≈ 7 s", "q10": "UniSoccer 1 clip ≈ 0,2 s",
              "q11": "UniSoccer 1 clip ≈ 0,2 s"}       # tu results/RUNS.md (04_warm_cache)


def fmt(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def main_table(P):
    head = ["Task", "n"] + list(SYSTEMS) + ["Δ SVA − B1", "SVA đúng & B1 sai / ngược lại", "p (McNemar)"] + \
           [f"paper: {k}" for k in PAPER]
    lines = [f"# Kết quả chính (tập test, {RUN['model']})", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, t in enumerate(TASK_IDS):
        acc = {n: 100 * accuracy(P[n][t]) for n in SYSTEMS}
        m = mcnemar(P["B1-vanilla"][t], P["SVA"][t])
        lines.append("| " + " | ".join([f"{t} ({TASK_META[t]['paper_id']})", str(len(P["SVA"][t]))]
                                       + [(f"**{fmt(acc[n])}**" if acc[n] == max(acc.values()) else fmt(acc[n])) for n in SYSTEMS]
                                       + [fmt(acc["SVA"] - acc["B1-vanilla"]), f"{m['b']} / {m['c']}",
                                          "< 0,001" if m["p_value"] < 0.001 else fmt(m["p_value"], 3)]
                                       + [fmt(PAPER[k][i]) for k in PAPER]) + " |")
    for label, tasks in (("Trung bình 6 task", TASK_IDS), ("Trung bình bỏ q10, q11", [t for t in TASK_IDS if t not in ("q10", "q11")])):
        means = {n: np.mean([100 * accuracy(P[n][t]) for t in tasks]) for n in SYSTEMS}
        lines.append("| " + " | ".join([f"**{label}**", ""] + [fmt(means[n]) for n in SYSTEMS]
                                       + [fmt(means["SVA"] - means["B1-vanilla"]), "", ""]
                                       + [fmt(np.mean([PAPER[k][TASK_IDS.index(t)] for t in tasks])) for k in PAPER]) + " |")
    lines += ["", "- n: số câu test (chia dev/test theo nhóm nội dung; q14 còn 276 câu sau khi bỏ 24 câu lỗi, test 192).",
              f"- B0-local, B1, SVA dùng cùng {RUN['model']} và cùng frame (8 frame một clip, tối đa 24 frame một câu).",
              "- Cột paper: số công bố của SoccerAgent (Qwen2.5-VL-7B, cả video, toàn bộ 10 nghìn câu): **khác giao thức, "
              "chỉ để tham khảo, không so trực tiếp**.",
              "- q10, q11 dùng Action Classifier của UniSoccer, được huấn luyện trên SoccerReplay-1988 — nguồn của clip q10: "
              "có thể trùng dữ liệu huấn luyện công cụ; so với B1 vẫn công bằng (cùng công cụ)."]
    return lines


def calls(name, t, r):
    """Uoc luong so luot goi model cua 1 cau."""
    if name == "b0_local":
        return 1
    if name == "b0_api":
        return 1                                    # 1 yeu cau API, 5 cau tra loi (candidate_count)
    if name == "b0_loop":
        return len(r["steps"])
    if name == "b1_vanilla":
        return 2 * len(r.get("steps") or []) + 2     # moi buoc: bo thuc thi viet lenh + cong cu; them lap ke hoach, tra loi
    n = 0
    for e in r["evidence"]:
        n += {"holistic": 1, "camera_switch": 2, "replay_match": 0, "action_classifier": 1,
              "event_match": 1 + 4, "multiview": 2 * len(e["material_ref"].get("views", []))}.get(e["tool_name"], 1)
    return n


def cost_table(P):
    lines = [f"# Chi phí và thời gian mỗi câu (tập test, {RUN['model']})", "",
             "Thời gian = độ trễ đo khi chạy (giây/câu), **chưa gồm** giải mã frame (cache dùng chung cho mọi hệ); "
             "SVA lấy lần chạy đầu (chưa có cache phân phối sự kiện của câu bình luận). "
             "Lượt gọi model là ước lượng theo cấu trúc của từng hệ.", "",
             "| Task | " + " | ".join(f"{n}: giây / lượt gọi" for n in SYSTEMS) + " | SVA: tính trước |", "|" + "---|" * (len(SYSTEMS) + 2)]
    for t in TASK_IDS:
        cells = []
        first = {r["index"]: r["latency_ms"] for r in load_predictions(res("sva_noloop"), t, "test")}
        for n, s in SYSTEMS.items():
            recs = P[n][t]
            lat = [max(r["latency_ms"], first.get(r["index"], 0)) if s == "sva" else r.get("latency_ms", 0) for r in recs]
            sec = np.mean(lat) / 1000
            cells.append(f"{fmt(sec)} / {fmt(np.mean([calls(s, t, r) for r in recs]))}")
        lines.append(f"| {t} | " + " | ".join(cells) + f" | {PRECOMPUTE.get(t, '—')} |")
    lines += ["", "- B0-api: gọi Gemini 3.5 Flash-Lite qua gói miễn phí (độ trễ gồm cả mạng và giới hạn tốc độ); chi phí tiền thực tế 0.",
              "- B0-local, B1, SVA chạy tại chỗ, không tốn tiền API; dữ liệu không rời máy.",
              "- SVA ở q9/q10/q11 gần như không gọi VLM: phần nặng là tính trước embedding / Action Classifier (cột cuối), "
              "làm một lần cho mỗi clip."]
    return lines


def risk_coverage(recs):
    """Sap theo do tin (xac suat cao nhat) giam dan; accuracy khi chi tra loi x% cau dau."""
    conf = np.array([max(r["dist"].values()) for r in recs])
    ok = np.array([correct(r) for r in recs], float)
    order = np.argsort(-conf, kind="stable")
    acc_at = np.cumsum(ok[order]) / np.arange(1, len(ok) + 1)
    cov = np.arange(1, len(ok) + 1) / len(ok)
    aurc = float(np.mean(1 - acc_at))                 # dien tich duoi duong rui ro - do phu (thap la tot)
    at = lambda c: float(acc_at[max(0, int(np.ceil(c * len(ok))) - 1)])
    return {"coverage": cov.tolist(), "accuracy": acc_at.tolist(), "aurc": aurc, "acc@50": at(0.5), "acc@80": at(0.8),
            "acc@100": at(1.0), "ece": ece(recs)}


def risk_table(P):
    curves, lines = {}, ["# Độ chính xác theo độ tin cậy (tập test)", "",
                         "Chỉ trả lời x% câu mà hệ tin nhất (độ tin = xác suất cao nhất của phân phối cuối). "
                         "AURC: rủi ro trung bình trên mọi mức phủ (thấp là tốt). ECE: lệch giữa độ tin và tỷ lệ đúng (thấp là tốt).", "",
                         "| Task | hệ | acc @ 50% | acc @ 80% | acc @ 100% | AURC | ECE |", "|---|---|---|---|---|---|---|"]
    for t in TASK_IDS + ["tất cả"]:
        for n in SYSTEMS:
            recs = [r for tt in TASK_IDS for r in P[n][tt]] if t == "tất cả" else P[n][t]
            rc = risk_coverage(recs)
            curves[f"{t}|{n}"] = rc
            lines.append(f"| {t} | {n} | {fmt(100 * rc['acc@50'])} | {fmt(100 * rc['acc@80'])} | {fmt(100 * rc['acc@100'])} | "
                         f"{fmt(rc['aurc'], 3)} | {fmt(rc['ece'], 3)} |")
    return lines, curves


def gemini_hybrid(P):
    """Gemini + nhanh cua SVA, khong goi API them: q9/q10/q11 dung nhanh chuyen dung (khong can VLM);
    q8 = trung binh(Gemini, kenh camera dau/cuoi) dung dung thiet ke da chon tren dev; q13, q14 giu Gemini
    (ban day du can hoi Gemini tung goc quay o q14 - chua chay)."""
    lines = ["# Gemini hỏi thẳng vs Gemini + các nhánh của SVA (bản lai, không gọi API thêm)", "",
             "| Task | Gemini hỏi thẳng | Gemini + SVA | thay đổi | b / c | p | phần SVA thêm vào |", "|---|---|---|---|---|---|---|"]
    note = {"q8": "kênh camera đầu/cuối (Qwen 3B), gộp trung bình với Gemini", "q9": "so khớp CLIP (không cần VLM)",
            "q10": "Action Classifier (không cần VLM)", "q11": "khớp loại sự kiện (UniSoccer + đọc chữ)",
            "q13": "— (giữ Gemini)", "q14": "— (giữ Gemini; bản đầy đủ: hỏi Gemini từng góc)"}
    hyb, accs = {}, []
    api = {t: {r["index"]: r for r in P["B0-api (Gemini)"][t]} for t in TASK_IDS}
    for t in TASK_IDS:
        if t in ("q9", "q10", "q11"):
            hyb[t] = P["SVA"][t]
        elif t == "q8":
            hyb[t] = []
            for r in load_predictions(res("sva_noloop"), "q8", "test"):
                cam = next((e["dist"] for e in r["evidence"] if e["tool_name"] == "camera_switch"), None)
                g = api["q8"][r["index"]]["dist"]
                d = {k: (g[k] + cam[k]) / 2 for k in g} if cam else g
                hyb[t].append({"index": r["index"], "gold": r["gold"], "pred": max(d, key=d.get), "dist": d})
        else:
            hyb[t] = P["B0-api (Gemini)"][t]
        a, b = 100 * accuracy(P["B0-api (Gemini)"][t]), 100 * accuracy(hyb[t])
        m = mcnemar(P["B0-api (Gemini)"][t], hyb[t])
        accs.append((a, b))
        lines.append(f"| {t} | {fmt(a)} | {fmt(b)} | {fmt(b - a)} | {m['b']} / {m['c']} | "
                     f"{'< 0,001' if m['p_value'] < 0.001 else fmt(m['p_value'], 3)} | {note[t]} |")
    a, b = np.mean(accs, axis=0)
    lines += [f"| **trung bình** | {fmt(a)} | {fmt(b)} | {fmt(b - a)} | | | |", "",
              "Thiết kế gộp ở q8 lấy nguyên từ lựa chọn trên dev của SVA (Gemini không có số trên dev nên không chỉnh gì thêm)."]
    return lines


def case_studies(P, k=10, seed=0):
    rng = random.Random(seed)
    S = {t: {s.index: s for s in load_split(t, "test")} for t in TASK_IDS}
    b1 = {t: {r["index"]: r for r in P["B1-vanilla"][t]} for t in TASK_IDS}
    groups = {"B1 sai, SVA đúng": (False, True), "B1 đúng, SVA sai": (True, False), "cả hai sai": (False, False)}
    lines = ["# Phân tích ca (tập test)", "", "Chọn ngẫu nhiên (seed 0), xoay vòng qua 6 task; mỗi ca kèm chuỗi bằng chứng của SVA "
             "và chuỗi công cụ của B1.", ""]
    for title, (b1_ok, sva_ok) in groups.items():
        lines += [f"## {title}", ""]
        pools = {t: [r for r in P["SVA"][t] if correct(r) == sva_ok and correct(b1[t][r["index"]]) == b1_ok] for t in TASK_IDS}
        picked, i = [], 0
        while len(picked) < k and any(pools.values()):
            t = TASK_IDS[i % len(TASK_IDS)]
            if pools[t]:
                picked.append((t, pools[t].pop(rng.randrange(len(pools[t])))))
            i += 1
        for t, r in picked:
            s, b = S[t][r["index"]], b1[t][r["index"]]
            opts = "; ".join(f"{key}: {s.option_texts[key][:70]}" for key in s.option_keys)
            lines += [f"**{t} #{r['index']}** — {s.question.split('speed.')[-1].strip()[:160]}", "",
                      f"- phương án: {opts}", f"- đáp án: {s.gold_key} · SVA: {r['pred']} (độ tin {r['confidence']:.2f}) · B1: {b['pred']}"]
            for e in r["evidence"]:
                lines.append(f"- SVA `{e['tool_name']}`: chọn {max(e['dist'], key=e['dist'].get)} "
                             f"({max(e['dist'].values()):.2f}) — {e['raw_observation'][:110]}")
            tools = " → ".join(st.get("tool", "?") for st in (b.get("steps") or []))
            last = (b.get("steps") or [{}])[-1].get("result", "")
            lines += [f"- B1: {tools or '(không gọi công cụ)'} · kết quả cuối: {str(last)[:140]}", ""]
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="qwen_local_3b", help="ten cau hinh trong configs/models.yaml")
    args = parser.parse_args()
    cfg = load_config("models")["vlm"][args.backend]
    RUN.update(backend=args.backend, suffix=result_name("", args.backend),
               model=cfg["model_id"].split("/")[-1] + (" 4-bit" if cfg.get("quantization") else ""))
    setup_logging("11_report")
    P = {n: {t: load_predictions(res(s), t, "test") for t in TASK_IDS} for n, s in SYSTEMS.items()}
    TABLES.mkdir(parents=True, exist_ok=True)
    out = {"main_results.md": main_table(P), "cost_time.md": cost_table(P), "sva_gemini.md": gemini_hybrid(P),
           "case_studies.md": case_studies(P)}
    risk, curves = risk_table(P)
    out["risk_coverage.md"] = risk
    for name, lines in out.items():
        f = TABLES / name.replace(".md", f"{RUN['suffix']}.md")
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info("da ghi %s", f)
    (TABLES / f"risk_coverage{RUN['suffix']}.json").write_text(json.dumps(curves), encoding="utf-8")
    log.info("\n".join(out["main_results.md"]))
    sva = np.mean([100 * accuracy(P["SVA"][t]) for t in TASK_IDS])
    sig = sum(mcnemar(P["B1-vanilla"][t], P["SVA"][t])["p_value"] < 0.05 for t in TASK_IDS)
    record_run(f"bang tong ket {RUN['model']}: SVA trung binh {sva:.1f}; hon B1 co y nghia (p<0.05) o {sig}/6 task")


if __name__ == "__main__":
    main()
