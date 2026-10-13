"""Ablation tren tap test. Moi cau hinh cua SVA da chon tren dev; o day chi do, khong chon them gi.

    python scripts/10_ablation.py --part q9      # GPU: embedding CLIP/SigLIP/DINOv2/HSV + UniSoccer cho q9 test
    python scripts/10_ablation.py --part a5      # GPU (Qwen): VLM ta bang chu roi chon tu chu (q8, q13)
    python scripts/10_ablation.py --part table   # tong hop -> results/tables/ablation_test.md
    them --backend qwen_local de lam cho ban 7B (file co hau to _7b; phan q9 khong phu thuoc VLM, dung chung)

Hoa (bo phieu nhan cung) thi chon ngau nhien co dinh seed trong cac nhan hoa.
"""
import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sva.config import record_run, setup_logging
from sva.constants import CAMERA_LABELS, RESULTS_DIR
from sva.eval.metrics import accuracy
from sva.eval.runner import load_predictions, load_split, result_name
from sva.eval.stats import mcnemar

log = logging.getLogger(__name__)
OUT = RESULTS_DIR / "ablation"
RUN = {"backend": "qwen_local_3b", "suffix": ""}


def res(name):
    return name if name == "b0_api" else result_name(name, RUN["backend"])
TEXT_DECISION = ("Question: {q}\nA video analysis tool watched the video and reported: \"{text}\"\n"
                 "Based only on this report, choose the answer.\nOptions:\n{opts}\nAnswer with the option key only.")


# ------------------------------------------------------------------ q9: encoder, so frame, thu tu, UniSoccer
def part_q9():
    from sva.branches.matching import FrameEncoder, candidate_order, chamfer
    from sva.models.unisoccer import ActionClassifier
    from sva.perception.cache import PerceptionCache
    from sva.perception.encoder import HSVEncoder
    samples, cache = load_split("q9", "test"), PerceptionCache()
    out = {"index": [s.index for s in samples], "gold": [s.option_keys.index(s.gold_key) for s in samples],
           "keys": [s.option_keys for s in samples], "variants": {}}
    for enc_name, n, square in [("clip_vit_l", 24, False), ("clip_vit_l", 4, False), ("siglip", 24, True),
                                ("dinov2_vitb14", 24, True), ("hsv", 24, False)]:
        enc = HSVEncoder() if enc_name == "hsv" else FrameEncoder(enc_name, square)
        rows, votes = [], []
        for s in samples:
            rep = cache.get_embedding(s.materials[0], n, 336, enc).astype(np.float32)
            cands = [cache.get_embedding(s.materials[c], n, 336, enc).astype(np.float32) for c in candidate_order(s)]
            rows.append([[round(v, 5) for v in chamfer(rep, c, 0)[1:]] for c in cands])
            best = np.array([(rep @ c.T).max(1) for c in cands]).argmax(0)     # moi frame replay bau 1 ung vien
            votes.append(np.bincount(best, minlength=len(cands)).tolist())
        out["variants"][f"{enc_name}|{n}|{'sq' if square else 'box'}"] = {"rows": rows, "votes": votes}
        log.info("xong %s %d", enc_name, n)
        del enc

    cls = ActionClassifier()
    vec = lambda p: np.array([p[k] for k in sorted(p)])

    def js(a, b):
        m = (a + b) / 2
        kl = lambda x, y: float((x * np.log((x + 1e-12) / (y + 1e-12))).sum())
        return (kl(a, m) + kl(b, m)) / 2 / np.log(2)

    out["unisoccer"] = [[1 - js(vec(cls.probs(s.materials[0])), vec(cls.probs(s.materials[c])))
                         for c in candidate_order(s)] for s in samples]
    (OUT / "q9_test.json").write_text(json.dumps(out), encoding="utf-8")


# ------------------------------------------------------------------ A5: quyet dinh tu chu thay vi tu video + phuong an
def part_a5():
    from sva.baselines.b0_local import B0Local
    from sva.models.scoring import softmax
    scorer = B0Local(RUN["backend"])
    out = {}
    for t in ("q8", "q13"):
        b1 = {r["index"]: r for r in load_predictions(res("b1_vanilla"), t, "test")}
        rows = []
        for i, s in enumerate(load_split(t, "test")):
            step = (b1.get(s.index, {}).get("steps") or [{}])[0]
            if step.get("tool") == "Vision Language Model" and step.get("result"):
                text, source = step["result"], "b1"        # q13: B1 da hoi VLM dung cau hoi nay, 8 frame
            else:
                frames, durations, _ = scorer.clips(s)
                text, source = scorer.backend.generate(frames[0], s.question, durations[0], max_new_tokens=128), "gen"
            opts = "\n".join(f"{k}. {s.option_texts[k]}" for k in s.option_keys)
            msg = [{"role": "user", "content": TEXT_DECISION.format(q=s.question, text=text.strip(), opts=opts)}]
            p = softmax(scorer.backend.score_chat(msg, s.option_keys))
            rows.append({"index": s.index, "gold": s.gold_key, "text": text, "source": source,
                         "dist": dict(zip(s.option_keys, p.tolist()))})
            if (i + 1) % 50 == 0:
                log.info("%s %d", t, i + 1)
        out[t] = rows
    (OUT / f"a5_test{RUN['suffix']}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------ tong hop
def rec(index, gold, keys, scores, rng):
    """Du doan = argmax; hoa thi chon ngau nhien (seed co dinh) trong cac phuong an hoa."""
    scores = np.asarray(scores, float)
    best = [k for k, v in zip(keys, scores) if v >= scores.max() - 1e-12]
    return {"index": index, "gold": gold, "pred": rng.choice(best) if len(best) > 1 else best[0]}


def hard_vote(dists, keys, rng, index, gold):
    votes = np.zeros(len(keys))
    for d in dists:
        if d is not None:
            votes[keys.index(max(d, key=d.get))] += 1
    return rec(index, gold, keys, votes, rng)


def q8_rows(rng):
    from sva.branches.branch_b import parse_switch
    S = {s.index: s for s in load_split("q8", "test")}
    soft = load_predictions(res("sva_noloop"), "q8", "test")
    hol, cam, hard, camhard = [], [], [], []
    for r in soft:
        s, keys = S[r["index"]], S[r["index"]].option_keys
        ev = {e["tool_name"]: e for e in r["evidence"]}
        hol.append(rec(r["index"], r["gold"], keys, [ev["holistic"]["dist"][k] for k in keys], rng))
        camera = ev.get("camera_switch")
        cam.append(rec(r["index"], r["gold"], keys, [camera["dist"][k] for k in keys], rng) if camera else hol[-1])
        cam_vote = None                  # nhan cung: camera dau = top-1, camera cuoi = top-1 -> 1 phuong an (neu co)
        if camera:
            m = re.match(r"dau: (.+) [\d.]+ \| cuoi: (.+) [\d.]+", camera["raw_observation"])
            x, y = CAMERA_LABELS.index(m.group(1)), CAMERA_LABELS.index(m.group(2))
            hit = [k for k, pair in zip(keys, parse_switch(s)) if pair == (x, y)]
            cam_vote = {k: float(k in hit) for k in keys} if hit else None
        hard.append(hard_vote([ev["holistic"]["dist"], cam_vote], keys, rng, r["index"], r["gold"]))
    return {"holistic": hol, "camera": cam, "soft": soft, "hard": hard}


def q14_rows(rng, split="test"):
    """Cac cach gop goc quay tren phan phoi tung goc da luu (test: tu SVA; dev: tu scripts/08)."""
    from sva.data.loader import parse_views
    from sva.evidence import fusion
    S = {s.index: s for s in load_split("q14", split)}
    if split == "test":
        src = [(r["index"], r["gold"], r["evidence"][0]["material_ref"]["views"], r["evidence"][0]["material_ref"]["quality"],
                r["evidence"][0].get("ordinal_scale")) for r in load_predictions(res("sva_noloop"), "q14", "test")]
    else:
        dev = json.loads((RESULTS_DIR / "dev" / "q14_multiview.json").read_text(encoding="utf-8"))["rows"]
        src = [(r["index"], r["gold"], r["views"], r["quality"], r["scale"]) for r in dev]
    out = {k: [] for k in ("hard vote", "linear pool", "log pool", "log pool + do ro", "uu tien can canh x2",
                           "uu tien camera chinh x2", "trong so theo toc do", "severity: thang thu tu",
                           "severity: log pool")}
    for idx, gold, views, quality, scale in src:
        keys = S[idx].option_keys
        cams = [v["camera"] for v in parse_views(S[idx].question)]
        speeds = [v["speed"] for v in parse_views(S[idx].question)]
        put = lambda name, d: out[name].append(rec(idx, gold, keys, [d[k] for k in keys], rng))
        out["hard vote"].append(hard_vote(views, keys, rng, idx, gold))
        put("linear pool", fusion.linear_pool(views))
        put("log pool", fusion.log_pool(views))
        put("log pool + do ro", fusion.log_pool(views, quality))
        put("uu tien can canh x2", fusion.log_pool(views, [2.0 if c.startswith("Close-up") else 1.0 for c in cams]))
        put("uu tien camera chinh x2", fusion.log_pool(views, [2.0 if c.startswith("Main") else 1.0 for c in cams]))
        put("trong so theo toc do", fusion.log_pool(views, speeds))
        if scale and "card" in " ".join(S[idx].option_texts.values()).lower():
            put("severity: thang thu tu", fusion.ordinal_fusion(views, scale))
            put("severity: log pool", fusion.log_pool(views))
    return out


def branch_a_rows(rng):
    from sva.branches.branch_a import BranchA, option_classes
    from sva.models.unisoccer import CLASS_NAMES, ActionClassifier

    class Stub:                          # chi doc cache phan phoi su kien cua cau binh luan (khong nap model)
        class backend:
            name = RUN["backend"]
    a = BranchA(Stub(), ActionClassifier())
    out = {}
    for t in ("q10", "q11"):
        hard = []
        for s in load_split(t, "test"):
            pv = a.video_events(s.materials[0])
            top = int(np.argmax(pv))
            if t == "q10":
                hit = [k for k, c in zip(s.option_keys, option_classes(s)) if c == top]
            else:
                hit = [k for k in s.option_keys if int(np.argmax(a.caption_events(s.option_texts[k]))) == top]
            hard.append(rec(s.index, s.gold_key, s.option_keys, [k in hit for k in s.option_keys], rng))
        out[t] = {"soft": load_predictions(res("sva_noloop"), t, "test"), "hard": hard}
    return out, CLASS_NAMES


def q9_rows(rng):
    d = json.loads((OUT / "q9_test.json").read_text(encoding="utf-8"))
    z = lambda A: (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-9)
    out = {}
    for key, v in d["variants"].items():
        R = np.array(v["rows"])
        out[key] = [rec(i, ks[g], ks, R[n, :, 0], rng) for n, (i, g, ks) in enumerate(zip(d["index"], d["gold"], d["keys"]))]
        if key == "clip_vit_l|24|box":
            lam = R[..., 0] - 0.1 * R[..., 1]
            fused = z(lam) + 0.25 * z(np.array(d["unisoccer"]))
            out["clip + lam 0.1"] = [rec(i, ks[g], ks, lam[n], rng) for n, (i, g, ks) in enumerate(zip(d["index"], d["gold"], d["keys"]))]
            out["clip + lam 0.1 + UniSoccer"] = [rec(i, ks[g], ks, fused[n], rng)
                                                 for n, (i, g, ks) in enumerate(zip(d["index"], d["gold"], d["keys"]))]
            out["clip, moi frame bau 1 phieu (nhan cung)"] = [rec(i, ks[g], ks, v["votes"][n], rng)
                                                             for n, (i, g, ks) in enumerate(zip(d["index"], d["gold"], d["keys"]))]
    out["UniSoccer 1-JS"] = [rec(i, ks[g], ks, d["unisoccer"][n], rng) for n, (i, g, ks) in enumerate(zip(d["index"], d["gold"], d["keys"]))]
    return out


def a5_rows():
    d = json.loads((OUT / f"a5_test{RUN['suffix']}.json").read_text(encoding="utf-8"))
    return {t: [{"index": r["index"], "gold": r["gold"], "pred": max(r["dist"], key=r["dist"].get)} for r in rows]
            for t, rows in d.items()}


def cleaning_rows():
    """A10: diem cua tung he neu cham bang dap an goc (truoc khi sua) tren cac cau da sua dap an."""
    from sva.data.loader import load_raw
    rows = []
    for t in ("q13", "q14"):
        fixed = [s for s in load_split(t, "test") if s.cleaning_note == "closeA_fixed"]
        raw = load_raw(t)
        for name in ("b0_local", "b1_vanilla", "b0_api", "sva"):
            P = {r["index"]: r["pred"] for r in load_predictions(res(name), t, "test")}
            rows.append({"task": t, "he": name, "so cau da sua": len(fixed),
                         "dung theo dap an da sua": sum(P[s.index] == s.gold_key for s in fixed),
                         "dung theo dap an goc": sum(P[s.index] == raw[s.index]["closeA"] for s in fixed)})
    return rows


def part_table():
    rng = random.Random(0)
    lines, summary = [f"# Ablation (tập test, backend {RUN['backend']})", "",
                      "Mọi cấu hình của SVA chọn trên dev; bảng này chỉ đo trên test. "
                      "McNemar so với dòng **in đậm** của cùng nhóm (b = biến thể đúng & mốc sai, c = ngược lại). "
                      "Hòa phiếu thì chọn ngẫu nhiên (seed 0) trong các phương án hòa.", ""], {}

    def block(title, ref_name, variants):
        nonlocal lines
        lines += [f"## {title}", "", "| biến thể | n | accuracy | b / c | p |", "|---|---|---|---|---|"]
        ref = variants[ref_name]
        for name, recs in variants.items():
            m = mcnemar(ref, recs) if name != ref_name else None
            label = f"**{name}**" if name == ref_name else name
            lines.append(f"| {label} | {len(recs)} | {100 * accuracy(recs):.1f} | ".replace(".", ",")
                         + (f"{m['b']} / {m['c']} | {m['p_value']:.4f} |".replace(".", ",") if m else "— | — |"))
            summary[f"{title} | {name}"] = round(100 * accuracy(recs), 1)
        lines.append("")

    pred = lambda name, t: load_predictions(res(name), t, "test")
    q8 = q8_rows(rng)
    ba, _ = branch_a_rows(rng)
    q9 = q9_rows(rng)
    q14 = q14_rows(rng)
    a5 = a5_rows() if (OUT / f"a5_test{RUN['suffix']}.json").exists() else {}

    # A1: bang chung la phan phoi vs nhan cung (moi cong cu chi con top-1)
    block("A1 · q8: phân phối vs nhãn cứng", "phân phối (SVA không vòng lặp)", {"phân phối (SVA không vòng lặp)": q8["soft"], "nhãn cứng": q8["hard"]})
    block("A1 · q9: phân phối vs nhãn cứng", "điểm khớp mềm (SVA)", {"điểm khớp mềm (SVA)": q9["clip_vit_l|24|box"], "mỗi frame bầu 1 phiếu": q9["clip, moi frame bau 1 phieu (nhan cung)"]})
    block("A1 · q10: phân phối vs nhãn cứng", "phân phối (SVA)", {"phân phối (SVA)": ba["q10"]["soft"], "chỉ nhãn top-1 của UniSoccer": ba["q10"]["hard"]})
    block("A1 · q11: phân phối vs nhãn cứng", "phân phối (SVA)", {"phân phối (SVA)": ba["q11"]["soft"], "top-1 video == top-1 câu bình luận": ba["q11"]["hard"]})
    block("A1 · q14: phân phối vs nhãn cứng", "log pool (SVA)", {"log pool (SVA)": q14["log pool"], "bỏ phiếu nhãn cứng": q14["hard vote"]})
    # A2: kien truc chuoi cong cu goc
    for t in ("q8", "q9", "q10", "q11", "q13", "q14"):
        block(f"A2 · {t}: chuỗi công cụ gốc (B1) vs SVA", "SVA", {"SVA": pred("sva", t), "B1 (chuỗi công cụ)": pred("b1_vanilla", t),
                                                               "B0 (VLM hỏi thẳng)": pred("b0_local", t)})
    # A3, A9: q9
    block("A3 · q9: các cách giải", "CLIP khớp mềm, 24 frame (SVA)", {
        "CLIP khớp mềm, 24 frame (SVA)": q9["clip_vit_l|24|box"], "B1: bình luận ×5 rồi so chữ": pred("b1_vanilla", "q9"),
        "B0: VLM hỏi thẳng": pred("b0_local", "q9"), "CLIP + phạt thứ tự λ = 0,1": q9["clip + lam 0.1"],
        "CLIP + λ = 0,1 + UniSoccer (đủ thành phần spec)": q9["clip + lam 0.1 + UniSoccer"],
        "CLIP, 4 frame/clip (bằng B0)": q9["clip_vit_l|4|box"], "chỉ UniSoccer (1 − JS)": q9["UniSoccer 1-JS"]})
    block("A9 · q9: encoder ảnh (24 frame/clip)", "CLIP ViT-L", {"CLIP ViT-L": q9["clip_vit_l|24|box"], "SigLIP B/16": q9["siglip|24|sq"],
                                                                 "DINOv2 B/14": q9["dinov2_vitb14|24|sq"], "HSV histogram (không học)": q9["hsv|24|box"]})
    # A4, A7, A8: q14
    block("A4 · q14: cách gộp góc quay", "log pool", {"log pool": q14["log pool"], "bỏ phiếu nhãn cứng": q14["hard vote"],
                                                     "linear pool": q14["linear pool"], "log pool + độ rõ tự đánh giá": q14["log pool + do ro"],
                                                     "B0: mọi góc trong 1 prompt": pred("b0_local", "q14")})
    block("A7 · q14 thẻ phạt: categorical vs thang thứ tự", "severity: log pool", {"severity: log pool": q14["severity: log pool"],
                                                                                   "severity: thang thứ tự": q14["severity: thang thu tu"]})
    q14dev = q14_rows(rng, "dev") if not RUN["suffix"] else None      # so dev da luu chi co cho ban 3B
    for tag, rows in (("test", q14), ("dev", q14dev)):
        if rows is None:
            continue
        block(f"A8 · q14 ({tag}): trọng số theo loại camera / tốc độ (chữ trong câu hỏi)", "log pool", {
            "log pool": rows["log pool"], "cận cảnh ×2": rows["uu tien can canh x2"], "camera chính ×2": rows["uu tien camera chinh x2"],
            "trọng số = hệ số tốc độ": rows["trong so theo toc do"]})
    if q14dev:
        block("A7 · q14 thẻ phạt (dev): categorical vs thang thứ tự", "severity: log pool", {
            "severity: log pool": q14dev["severity: log pool"], "severity: thang thứ tự": q14dev["severity: thang thu tu"]})
    # A5
    for t, rows in a5.items():
        idx = {r["index"] for r in rows}
        block(f"A5 · {t}: VLM thấy video + phương án vs quyết định từ chữ", "VLM chấm thẳng phương án (B0)", {
            "VLM chấm thẳng phương án (B0)": [r for r in pred("b0_local", t) if r["index"] in idx],
            "VLM tả bằng chữ → chọn từ chữ": rows, "SVA": [r for r in pred("sva", t) if r["index"] in idx]})
    # Reasoner
    for t in ("q8", "q13"):
        block(f"Reasoner · {t}: có / không vòng lặp", "SVA", {"SVA": pred("sva", t), "không vòng lặp": pred("sva_noloop", t)})
    block("Thành phần · q8: kênh camera", "SVA không vòng lặp", {"SVA không vòng lặp": q8["soft"], "chỉ tổng thể": q8["holistic"],
                                                               "chỉ kênh camera đầu/cuối": q8["camera"]})

    lines += ["## A6 · đổi backbone 3B → 7B", "", "Chờ server (Giai đoạn 3 của kế hoạch dài hạn).", "",
              "## A10 · trước và sau làm sạch dữ liệu", "",
              "q14 bỏ 24 câu không có đáp án hợp lệ (không chấm được). Trên các câu test được **sửa đáp án**:", "",
              "| task | hệ | số câu đã sửa | đúng theo đáp án đã sửa | đúng theo đáp án gốc |", "|---|---|---|---|---|"]
    for r in cleaning_rows():
        lines.append(f"| {r['task']} | {r['he']} | {r['so cau da sua']} | {r['dung theo dap an da sua']} | {r['dung theo dap an goc']} |")
    out = RESULTS_DIR / "tables" / f"ablation_test{RUN['suffix']}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / f"summary{RUN['suffix']}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("\n".join(lines))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", required=True, choices=["q9", "a5", "table"])
    parser.add_argument("--backend", default="qwen_local_3b", help="ten cau hinh trong configs/models.yaml")
    args = parser.parse_args()
    RUN.update(backend=args.backend, suffix=result_name("", args.backend))
    setup_logging(f"10_ablation_{args.part}")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.part == "q9":
        part_q9()
        record_run("q9 test: embedding CLIP 4/24, SigLIP, DINOv2, HSV + UniSoccer -> results/ablation/q9_test.json")
    elif args.part == "a5":
        part_a5()
        rows = a5_rows()
        record_run("A5 test: " + "; ".join(f"{t} quyet dinh tu chu {100 * accuracy(r):.1f}" for t, r in rows.items()))
    else:
        out = part_table()
        record_run(f"bang {out.relative_to(RESULTS_DIR).as_posix()}")


if __name__ == "__main__":
    main()
