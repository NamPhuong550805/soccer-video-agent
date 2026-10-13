# SVA - Soccer Video Agent

Khóa luận tốt nghiệp: hệ đa tác tử trả lời câu hỏi về video bóng đá trên 6 task video của
[SoccerBench](https://arxiv.org/abs/2505.03735), không huấn luyện lại model.

| task_id | paper | tác vụ | số mẫu | sau làm sạch | clip/mẫu |
|---|---|---|---|---|---|
| q8 | Q7 | Camera Status Switching | 400 | 400 | 1 |
| q9 | Q8 | Replay Grounding | 400 | 400 | 5 |
| q10 | Q9 | Action Classification | 1000 | 1000 | 1 |
| q11 | Q10 | Commentary Generation | 1000 | 1000 | 1 |
| q13 | Q12 | Jersey Color Relevant QA | 700 | 700 | 1 |
| q14 | Q13 | Multi-view Foul Recognition | 300 | 276 | 2-4 |

Tên file json của dataset lệch 1 so với ký hiệu trong paper (`q8.json` là Q7), trong code luôn dùng tên file.

## Cài đặt

```bash
pip install -e ".[dev]"
```

Dataset đặt ở `data/raw/datasets/SoccerBench/` (gồm `qa/` và `materials/`), không nằm trong repo.

## Chạy

```bash
python scripts/00_verify_dataset.py     # kiểm tra dataset
python scripts/01_audit.py              # phân loại lỗi annotation
python scripts/02_clean.py              # -> data/processed/clean/
python scripts/03_build_split.py        # -> data/processed/index/splits.json
python scripts/04_warm_cache.py --task all --encoder clip_vit_l
python scripts/05_run_system.py --system b0_local --task all      # baseline, tập test
python scripts/06_evaluate.py --systems b0_local
pytest tests
```

SVA (chọn tham số trên dev rồi chạy test):

```bash
python scripts/04_warm_cache.py --task q9 --encoder clip_vit_l --n-frames 24   # GPU 4GB: tinh truoc embedding q9
python scripts/07_dev_matching.py                                           # chon cau hinh so khop q9 tren dev
python scripts/08_dev_multiview.py                                          # chon cach gop goc quay q14 tren dev
python scripts/04_warm_cache.py --task q10 q11 --unisoccer                     # Action Classifier cho nhanh A
python scripts/09_dev_branch_a.py                                           # chon kenh cho q10, q11 tren dev
python scripts/05_run_system.py --system sva_trace --theta 1.0 --split dev --task all
python scripts/05_run_system.py --system sva --theta auto --task all
python scripts/06_evaluate.py --systems b0_local b1_vanilla sva --compare b1_vanilla sva
python scripts/10_ablation.py --part q9 && python scripts/10_ablation.py --part a5 && python scripts/10_ablation.py --part table
python scripts/11_report.py                                                 # bang chinh, chi phi, do tin cay, phan tich ca
```

Mỗi lần chạy script ghi log chi tiết vào `logs/` và thêm một dòng vào `results/RUNS.md` (lệnh, commit, kết quả, thời gian).

Baseline dùng API (Gemini) cần file `.env` ở thư mục gốc:

```
GOOGLE_API_KEY=...
```

Baseline B1 (tái hiện SoccerAgent) cần code gốc và checkpoint Action Classifier của UniSoccer, đặt trong
`external/` (không nằm trong repo):

```bash
git clone https://github.com/jyrao/SoccerAgent external/SoccerAgent
git -C external/SoccerAgent checkout 763c254     # ban da dung (31/10/2025)
huggingface-cli download Homie0609/UniSoccer pretrained_classification.pth --local-dir external/unisoccer
pip install scenedetect einops
python scripts/04_warm_cache.py --task q10 --unisoccer
python scripts/05_run_system.py --system b1_vanilla --task all
```

Notebook:

- `notebooks/data_overview.ipynb`: dữ liệu, làm sạch, lấy frame
- `notebooks/evidence_scoring.ipynb`: chấm phân phối bằng Qwen2.5-VL (cần GPU)
- `notebooks/baseline_b0.ipynb`: kết quả baseline B0 và vòng lặp agent
- `notebooks/replay_matching.ipynb`: so khớp replay bằng hình ảnh (q9): chọn trên dev, ví dụ, kết quả test
- `notebooks/multiview_fusion.ipynb`: gộp nhiều góc quay (q14): nhãn lệch, so cách gộp, kết quả test theo chiều
- `notebooks/sva_pipeline.ipynb`: nhánh A (q10, q11), SVA trên cả 6 task, kiểm tra chuỗi bằng chứng
- `notebooks/evaluation.ipynb`: kết quả chính, ablation, độ tin cậy, chi phí, phân tích ca
- `notebooks/demo.ipynb`: **chạy thử một mẫu bất kỳ** (chạy thật model): chuỗi bằng chứng, chế độ Open, hỏi về video dài
- `notebooks/agent_evaluation.ipynb`: kết quả đánh giá SVA-Agent (E0–E6)

Trên GPU 4GB dùng cấu hình `qwen_local_3b` (Qwen2.5-VL-3B, 4-bit). Nếu tải model bị treo: đặt `HF_HUB_DISABLE_XET=1`.

## Hỏi về video thật (SVA-Agent)

Câu hỏi tiếng Anh, có 2–4 phương án hoặc không có phương án (agent tự tạo ứng viên); video có thể dài, fps / kích thước bất kỳ:

```bash
python scripts/13_ask.py --video highlight.mp4 --question "What happens at 1:10?"
python scripts/13_ask.py --video highlight.mp4 --question "Which earlier play is shown again in the replay at 2:40?"
python scripts/13_ask.py --video main.mp4 closeup.mp4 --question "Did the foul involve a handball?" --options No Yes
```

Đánh giá agent (giấu tên task, chế độ Open, video dài, định dạng khác, từ chối khi không chắc):

```bash
python scripts/05_run_system.py --system sva_agent --theta 0.2 --task all
python scripts/05_run_system.py --system sva_agent_open --theta 0.2 --task all
python scripts/12_eval_agent.py --part router   # cũng có: paraphrase, open, long, format, abstain, policy, table
```

## Cấu trúc

```
configs/            siêu tham số, cấu hình model
src/sva/
  constants.py      đường dẫn, bảng task, danh sách lỗi annotation
  schema.py         Sample, Evidence
  data/             loader, audit, clean, split
  perception/       video (lấy frame), cache, encoder
  models/           giao diện VLM, Qwen2.5-VL, chấm phân phối, UniSoccer
  evidence/         dựng Evidence
  baselines/        B0 (hỏi thẳng VLM local / API), B1 (chuỗi công cụ của SoccerAgent)
  eval/             chạy hệ thống, metric, McNemar
  reasoner.py       vòng lặp agent: gộp bằng chứng, tự chọn hành động khi chưa đủ tin
  router.py         chọn nhánh theo task
  branches/         nhánh A (công cụ chuyên dụng), B (một clip), C (nhiều clip: so khớp replay, gộp góc quay)
  pipeline.py       SVA: router -> nhánh -> reasoner
  agent/            SVA-Agent: hiểu câu hỏi, bản đồ video, grounding (đoạn video, tự tạo ứng viên), controller (LLM chọn bước), trả lời
scripts/            chạy theo thứ tự số
notebooks/
tests/
```

## Kết quả (bản 3B, tập test)

| Task | B0 (Qwen 3B) | B1 (SoccerAgent, 3B) | Gemini hỏi thẳng | SVA (3B) |
|---|---|---|---|---|
| q8 camera | 45,4 | 45,0 | 68,9 | 58,9 |
| q9 replay | 27,5 | 25,7 | 32,5 | 78,9 |
| q10 loại sự kiện | 36,1 | 74,6 | 64,4 | 87,1 |
| q11 bình luận | 31,3 | 31,4 | 70,1 | 74,3 |
| q13 màu áo | 45,9 | 46,3 | 48,8 | 46,5 |
| q14 phạm lỗi đa góc | 56,8 | 45,3 | 79,2 | 60,4 |
| **Trung bình** | 40,5 | 44,7 | 60,7 | **67,7** |

Chi tiết: `results/tables/` (bảng chính, ablation, độ tin cậy, chi phí, phân tích ca) và `results/RUNS.md` (lịch sử chạy).

## Tiến độ

- [x] Kiểm tra và làm sạch dữ liệu, chia dev/test
- [x] Lấy frame, cache, encoder (CLIP / SigLIP / DINOv2)
- [x] Evidence schema, chấm phân phối bằng VLM
- [x] Baseline B0, vòng lặp agent cho tầng suy luận
- [x] Baseline B1 (tái hiện chuỗi công cụ gốc)
- [x] Router, nhánh B (q8, q13), nhánh C so khớp replay (q9) và gộp nhiều góc quay (q14)
- [x] Nhánh A (q10, q11), SVA chạy đủ 6 task
- [x] Đánh giá, ablation (bản 3B)
- [x] SVA-Agent: tự đoán loại câu hỏi, chế độ không phương án, video dài, demo
- [ ] Bản 7B trên server
