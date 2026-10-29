# SVA - Soccer Video Agent

Khóa luận tốt nghiệp: hệ đa tác tử trả lời câu hỏi về video bóng đá trên 6 task video của
[SoccerBench](https://arxiv.org/abs/2505.03735), không huấn luyện lại model (training-free).

Ý chính: mỗi công cụ trả về **một phân phối xác suất trên các phương án** (Evidence) thay vì một chuỗi chữ; mỗi nhóm
câu hỏi đi vào một nhánh có công cụ phù hợp (so khớp hình ảnh cho replay, Action Classifier cho loại sự kiện, hỏi riêng
từng góc quay cho lỗi đa góc...); Reasoner gộp các phân phối và chỉ hỏi thêm khi chưa đủ tin.

| task_id | paper | tác vụ | số câu | sau làm sạch | clip/câu |
|---|---|---|---|---|---|
| q8 | Q7 | Camera Status Switching | 400 | 400 | 1 |
| q9 | Q8 | Replay Grounding | 400 | 400 | 5 |
| q10 | Q9 | Action Classification | 1000 | 1000 | 1 |
| q11 | Q10 | Commentary Generation | 1000 | 1000 | 1 |
| q13 | Q12 | Jersey Color Relevant QA | 700 | 700 | 1 |
| q14 | Q13 | Multi-view Foul Recognition | 300 | 276 | 2-4 |

Tên file json của dataset lệch 1 so với ký hiệu trong paper (`q8.json` là Q7), trong code luôn dùng tên file.

## Kết quả (tập test, 2.642 câu)

| Task | B0-3B | B1-3B | **SVA-3B** | B0-7B | B1-7B | **SVA-7B** | Gemini hỏi thẳng |
|---|---|---|---|---|---|---|---|
| q8 camera | 45,4 | 45,0 | 58,9 | 54,3 | 52,5 | **69,3** | 68,9 |
| q9 replay | 27,5 | 25,7 | 78,9 | 24,3 | 23,6 | **78,9** | 32,5 |
| q10 loại sự kiện | 36,1 | 74,6 | 87,1 | 45,4 | 79,7 | **87,1** | 64,4 |
| q11 bình luận | 31,3 | 31,4 | 74,3 | 44,6 | 37,0 | **72,0** | 70,1 |
| q13 màu áo | 45,9 | 46,3 | 46,5 | 45,1 | 43,3 | 48,4 | **48,8** |
| q14 phạm lỗi đa góc | 56,8 | 45,3 | 60,4 | 63,0 | 52,6 | 64,1 | **79,2** |
| **Trung bình** | 40,5 | 44,7 | 67,7 | 46,1 | 48,1 | **70,0** | 60,7 |

B0: Qwen2.5-VL hỏi thẳng; B1: dựng lại chuỗi công cụ của SoccerAgent với cùng model. SVA hơn B1 có ý nghĩa thống kê
(McNemar p < 0,05) ở 5/6 task (3B) và 6/6 task (7B). Mọi cấu hình chọn trên dev, tập test chỉ dùng để chấm.
Chi tiết: `results/tables/` (bảng chính, ablation, độ tin cậy, chi phí, phân tích ca; hậu tố `_7b` cho bản 7B).

SVA-Agent (`results/tables/agent_test.md`, bản 3B): giấu tên task thì 2.642/2.642 câu trả lời giữ nguyên; định tuyến
bằng luật đúng 100%; video dài ghép từ clip test 62,4% so với 63,1% trên clip cắt sẵn; ngưỡng từ chối 0,527 → trả lời
65,5% câu, đúng 82,5%.

## Cấu trúc

```
configs/
  default.yaml      siêu tham số (chọn trên dev, ghi chú kèm số liệu)
  models.yaml       VLM (3B / 7B / Gemini), encoder ảnh, Action Classifier của UniSoccer
src/sva/
  config.py         đường dẫn, bảng task, lỗi nhãn đã biết; đọc YAML; log và sổ tay chạy
  data.py           Sample; đọc JSON gốc, kiểm toán nhãn, làm sạch, chia dev / test
  perception.py     đọc video, lấy frame theo thời gian, đoạn ảo, cache frame / embedding, encoder ảnh
  models.py         logprob -> phân phối; Qwen2.5-VL tại chỗ; Action Classifier của UniSoccer
  evidence.py       Evidence (phân phối trên các phương án) và các cách gộp
  baselines.py      B0-local (Qwen hỏi thẳng), B0-api (Gemini)
  b1.py             B1: dựng lại SoccerAgent (bộ công cụ và vòng thực thi của code gốc)
  branches.py       nhánh A (q10, q11), B (q8, q13), C (q9 so khớp replay, q14 nhiều góc quay)
  reasoner.py       Reasoner: gộp bằng chứng, chưa đủ tin thì tự hỏi thêm (theo luật hoặc do LLM chọn)
  pipeline.py       Router và SVA
  eval.py           metric, McNemar, chạy hệ thống, đọc / ghi kết quả
  agent.py          SVA-Agent: Question / Video / Grounding / Answer agent cho video và câu hỏi thật
scripts/            01 → 07 theo thứ tự chạy; hpc/ cho server Slurm
notebooks/          01 → 07 trình bày kết quả (02 và 07 chạy model thật, cần GPU)
tests/              pytest
results/            predictions/ (mọi dự đoán), tables/, dev/, ablation/, agent/, RUNS.md (sổ tay chạy)
```

## Cài đặt

```bash
pip install -e ".[dev]"            # them ".[api]" neu chay B0-api (Gemini)
```

- Dataset: `data/raw/datasets/SoccerBench/` (gồm `qa/` và `materials/`), không nằm trong repo.
- B1 và nhánh A cần code gốc SoccerAgent và checkpoint Action Classifier của UniSoccer trong `external/`:

```bash
git clone https://github.com/jyrao/SoccerAgent external/SoccerAgent
git -C external/SoccerAgent checkout 763c254     # ban da dung (31/10/2025)
huggingface-cli download Homie0609/UniSoccer pretrained_classification.pth --local-dir external/unisoccer
```

- B0-api cần file `.env` ở thư mục gốc: `GOOGLE_API_KEY=...`
- GPU 4 GB dùng cấu hình `qwen_local_3b` (Qwen2.5-VL-3B 4-bit, mặc định của các script). Nếu tải model bị treo: đặt `HF_HUB_DISABLE_XET=1`.

## Chạy lại toàn bộ

```bash
# 1. du lieu: kiem tra, kiem toan nhan, lam sach, chia dev / test
python scripts/01_data.py all

# 2. tinh truoc frame, embedding q9, phan phoi Action Classifier
python scripts/02_warm_cache.py --task all
python scripts/02_warm_cache.py --task q9 --encoder clip_vit_l --n-frames 24
python scripts/02_warm_cache.py --task q10 q11 --unisoccer

# 3. baseline
python scripts/03_run.py --system b0_local --task all
python scripts/03_run.py --system b0_api --task all --workers 4
python scripts/03_run.py --system b1_vanilla --task all
python scripts/03_run.py --system b0_loop_trace --theta 1.0 --split dev --task all
python scripts/03_run.py --system b0_loop --theta auto --task all

# 4. chon cau hinh cac nhanh tren dev
python scripts/04_dev_tuning.py q9
python scripts/04_dev_tuning.py q14
python scripts/04_dev_tuning.py branch_a

# 5. SVA: chon theta tren dev roi chay test
python scripts/03_run.py --system sva_trace --theta 1.0 --split dev --task all
python scripts/03_run.py --system sva --theta auto --task all
python scripts/03_run.py --system sva_noloop --theta 0 --task all

# 6. bang ket qua va ablation
python scripts/05_report.py main
python scripts/06_ablation.py q9
python scripts/06_ablation.py a5
python scripts/06_ablation.py table

# 7. SVA-Agent
python scripts/03_run.py --system sva_agent --theta 0.2 --task all
python scripts/03_run.py --system sva_agent_open --theta 0.2 --task all
python scripts/07_agent.py router        # con co: paraphrase, open, long, format, abstain, policy, table

pytest tests
```

Bản 7B: thêm `--backend qwen_local` vào các lệnh trên (kết quả có hậu tố `_7b`); trên server Slurm dùng
`sbatch scripts/hpc/job.sh scripts/03_run.py ...`. Mỗi lần chạy ghi log vào `logs/` và thêm một dòng vào
`results/RUNS.md` (lệnh, commit, kết quả, thời gian).

## Hỏi về video thật

Câu hỏi tiếng Anh, có 2–4 phương án hoặc không có phương án (agent tự tạo ứng viên); video có thể dài, fps / kích thước bất kỳ:

```bash
python scripts/07_agent.py ask --video highlight.mp4 --question "What happens at 1:10?"
python scripts/07_agent.py ask --video highlight.mp4 --question "Which earlier play is shown again in the replay at 2:40?"
python scripts/07_agent.py ask --video main.mp4 closeup.mp4 --question "Did the foul involve a handball?" --options No Yes
```

Giới hạn: q9 trên video dài cần mốc thời gian của replay; video dọc 9:16 làm q10 giảm mạnh.

## Notebook

| notebook | nội dung |
|---|---|
| `01_data` | dữ liệu thô, các lỗi nhãn, làm sạch, chia dev / test, lấy frame, cache |
| `02_evidence_scoring` | chấm phân phối bằng Qwen2.5-VL, nhiệt độ, các loại Evidence (cần GPU) |
| `03_baselines` | B0 trên test, độ tin cậy, thiên vị vị trí, vòng lặp agent |
| `04_branches` | nhánh C (q9 so khớp replay, q14 gộp góc quay), nhánh A (q10, q11), SVA trên 6 task |
| `05_evaluation` | kết quả chính, ablation, độ tin cậy, chi phí, Gemini + SVA, phân tích ca, 3B so với 7B |
| `06_agent` | đánh giá SVA-Agent (E0–E6, U2) |
| `07_demo` | chạy thử một câu bất kỳ, chế độ Open, hỏi về video dài (cần GPU) |
