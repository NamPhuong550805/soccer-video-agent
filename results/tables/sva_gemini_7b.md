# Gemini hỏi thẳng vs Gemini + các nhánh của SVA (bản lai, không gọi API thêm)

| Task | Gemini hỏi thẳng | Gemini + SVA | thay đổi | b / c | p | phần SVA thêm vào |
|---|---|---|---|---|---|---|
| q8 | 68,9 | 70,0 | 1,1 | 33 / 30 | 0,801 | kênh camera đầu/cuối (Qwen2.5-VL-7B-Instruct), gộp trung bình với Gemini |
| q9 | 32,5 | 78,9 | 46,4 | 148 / 18 | < 0,001 | so khớp CLIP (không cần VLM) |
| q10 | 64,4 | 87,1 | 22,7 | 201 / 42 | < 0,001 | Action Classifier (không cần VLM) |
| q11 | 70,1 | 72,0 | 1,9 | 116 / 103 | 0,417 | khớp loại sự kiện (UniSoccer + đọc chữ) |
| q13 | 48,8 | 48,8 | 0,0 | 0 / 0 | 1,000 | — (giữ Gemini) |
| q14 | 79,2 | 79,2 | 0,0 | 0 / 0 | 1,000 | — (giữ Gemini; bản đầy đủ: hỏi Gemini từng góc) |
| **trung bình** | 60,7 | 72,7 | 12,0 | | | |

Thiết kế gộp ở q8 lấy nguyên từ lựa chọn trên dev của SVA (Gemini không có số trên dev nên không chỉnh gì thêm).
