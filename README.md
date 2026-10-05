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
pytest tests
```

Notebook:

- `notebooks/data_overview.ipynb`: dữ liệu, làm sạch, lấy frame
- `notebooks/evidence_scoring.ipynb`: chấm phân phối bằng Qwen2.5-VL (cần GPU)

Trên GPU 4GB dùng cấu hình `qwen_local_3b` (Qwen2.5-VL-3B, 4-bit). Nếu tải model bị treo: đặt `HF_HUB_DISABLE_XET=1`.

## Cấu trúc

```
configs/            siêu tham số, cấu hình model
src/sva/
  constants.py      đường dẫn, bảng task, danh sách lỗi annotation
  schema.py         Sample, Evidence
  data/             loader, audit, clean, split
  perception/       video (lấy frame), cache, encoder
  models/           giao diện VLM, Qwen2.5-VL, chấm phân phối
  evidence/         dựng Evidence
scripts/            chạy theo thứ tự số
notebooks/
tests/
```

## Tiến độ

- [x] Kiểm tra và làm sạch dữ liệu, chia dev/test
- [x] Lấy frame, cache, encoder (CLIP / SigLIP / DINOv2)
- [x] Evidence schema, chấm phân phối bằng VLM
- [ ] Baseline
- [ ] Router và các nhánh A / B / C
- [ ] Đánh giá, ablation
