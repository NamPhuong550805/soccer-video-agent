# SVA-Agent: kết quả đánh giá (bản 3B, tập test)

## E0 · Chế độ benchmark, giấu tên task

| Task | n | task đoán đúng | SVA (biết task) | SVA-Agent (tự đoán) | dự đoán trùng | b / c |
|---|---|---|---|---|---|---|
| q8 | 280 | 100.0% | 58,9 | 58,9 | 100.0% | 0 / 0 |
| q9 | 280 | 100.0% | 78,9 | 78,9 | 100.0% | 0 / 0 |
| q10 | 700 | 100.0% | 87,1 | 87,1 | 100.0% | 0 / 0 |
| q11 | 700 | 100.0% | 74,3 | 74,3 | 100.0% | 0 / 0 |
| q13 | 490 | 100.0% | 46,5 | 46,5 | 100.0% | 0 / 0 |
| q14 | 192 | 100.0% | 60,4 | 60,4 | 100.0% | 0 / 0 |

## E0 / E1 · Định tuyến

| cách | dev | test |
|---|---|---|
| luat (so clip that) | 100.00% (1134/1134) | 100.00% (2642/2642) |
| luat (1 video) | 100.00% (1134/1134) | 100.00% (2642/2642) |
| chi LLM | 98.32% (1115/1134) | 98.11% (2592/2642) |
| luat + LLM | 100.00% (1134/1134) | 100.00% (2642/2642) |

Câu ngoài phạm vi (30 câu tự viết): Question Agent từ chối đúng **22/30**; chỉ luật thì 28/30 câu không khớp luật nào.

## E1 · Câu hỏi viết lại (LLM viết lại, 40 câu test mỗi task)

| Task | định tuyến đúng | đúng (câu gốc) | đúng (câu viết lại) |
|---|---|---|---|
| q8 | 100.0% | 57,5 | 57,5 |
| q9 | 100.0% | 80,0 | 80,0 |
| q10 | 100.0% | 90,0 | 90,0 |
| q11 | 100.0% | 75,0 | 75,0 |
| q13 | 97.5% | 47,5 | 42,5 |
| q14 | 100.0% | 65,0 | 67,5 |

## E2 · Chế độ Open (không cho thấy phương án)

| Task | n | MCQ (cùng câu) | Open kiểu paper (ánh xạ vào phương án) | Open kiểu chặt (khớp đáp án mở) | b / c (MCQ vs Open) |
|---|---|---|---|---|---|
| q8 | 280 | 58,9 | 51,1 | 9,3 | 32 / 54 |
| q9 | 280 | 78,9 | 78,9 | 78,9 | 0 / 0 |
| q10 | 700 | 87,1 | 74,7 | 64,9 | 18 / 105 |
| q11 | 200 | 77,0 | 57,0 | sự kiện khớp 32,0, ROUGE-L 0.16 | 17 / 57 |
| q13 | 490 | 46,5 | 45,1 | 32,0 | 47 / 54 |
| q14 | 192 | 60,4 | 67,2 | 66,1 | 20 / 7 |

## E3 · Video dài (ghép từ clip test, câu hỏi kèm mốc thời gian)

| Task | n | đúng (clip gốc, cắt sẵn) | đúng (video dài, agent tự tìm đoạn) | IoU đoạn | trạng thái |
|---|---|---|---|---|---|
| q8 | 100 | 62,0 | 56,0 | 0.87 | answered 100 |
| q9 | 25 | 76,0 | 60,0 | — | answered 25 |
| q10 | 100 | 77,0 | 78,0 | 0.99 | answered 100 |
| q11 | 60 | 76,7 | 76,7 | 1.00 | answered 60 |
| q13 | 100 | 39,0 | 43,0 | 0.99 | answered 100 |
| q14 | 30 | 63,3 | 70,0 | — | answered 30 |

## E4 · Định dạng video khác (40 clip test mỗi task)

| Task | gốc | fps ×2 | 1280×720 | thêm bảng tỉ số / logo | cắt dọc 9:16 |
|---|---|---|---|---|---|
| q8 | 57,5 | 60,0 | 60,0 | 57,5 | 60,0 |
| q10 | 92,5 | 90,0 | 90,0 | 90,0 | 35,0 |

## E6 · Biết từ chối

Ngưỡng xác suất **0.527** chọn trên dev (mục tiêu đúng ≥ 85% trên câu trả lời).

| | trả lời (% câu) | đúng trên câu trả lời | đúng nếu trả lời hết |
|---|---|---|---|
| dev | 64,8 | 85,2 | 72,1 |
| test | 65,5 | 82,5 | 70,4 |

## U2 · Reasoner: luật hay LLM ra quyết định

| Task | dev luật | dev LLM | test luật | test LLM | b / c (test) | số bước TB (LLM, test) |
|---|---|---|---|---|---|---|
| q8 | 67,5 | 62,5 | 58,9 | 61,1 | 21 / 15 | 2,3 |
| q13 | 45,7 | 46,2 | 46,5 | 47,6 | 22 / 17 | 2,1 |

