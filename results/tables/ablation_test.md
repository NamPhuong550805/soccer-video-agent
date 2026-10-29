# Ablation (tập test, backend qwen_local_3b)

Mọi cấu hình của SVA chọn trên dev; bảng này chỉ đo trên test. McNemar so với dòng **in đậm** của cùng nhóm (b = biến thể đúng & mốc sai, c = ngược lại). Hòa phiếu thì chọn ngẫu nhiên (seed 0) trong các phương án hòa.

## A1 · q8: phân phối vs nhãn cứng

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **phân phối (SVA không vòng lặp)** | 280 | 56,4 | — | — |
| nhãn cứng | 280 | 45,7 | 35 / 65 | 0,0037 |

## A1 · q9: phân phối vs nhãn cứng

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **điểm khớp mềm (SVA)** | 280 | 78,9 | — | — |
| mỗi frame bầu 1 phiếu | 280 | 73,9 | 5 / 19 | 0,0066 |

## A1 · q10: phân phối vs nhãn cứng

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **phân phối (SVA)** | 700 | 87,1 | — | — |
| chỉ nhãn top-1 của UniSoccer | 700 | 71,9 | 12 / 119 | 0,0000 |

## A1 · q11: phân phối vs nhãn cứng

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **phân phối (SVA)** | 700 | 74,3 | — | — |
| top-1 video == top-1 câu bình luận | 700 | 56,0 | 45 / 173 | 0,0000 |

## A1 · q14: phân phối vs nhãn cứng

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **log pool (SVA)** | 192 | 60,4 | — | — |
| bỏ phiếu nhãn cứng | 192 | 60,9 | 5 / 4 | 1,0000 |

## A2 · q8: chuỗi công cụ gốc (B1) vs SVA

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **SVA** | 280 | 58,9 | — | — |
| B1 (chuỗi công cụ) | 280 | 45,0 | 43 / 82 | 0,0007 |
| B0 (VLM hỏi thẳng) | 280 | 45,4 | 19 / 57 | 0,0000 |

## A2 · q9: chuỗi công cụ gốc (B1) vs SVA

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **SVA** | 280 | 78,9 | — | — |
| B1 (chuỗi công cụ) | 280 | 25,7 | 15 / 164 | 0,0000 |
| B0 (VLM hỏi thẳng) | 280 | 27,5 | 18 / 162 | 0,0000 |

## A2 · q10: chuỗi công cụ gốc (B1) vs SVA

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **SVA** | 700 | 87,1 | — | — |
| B1 (chuỗi công cụ) | 700 | 74,6 | 14 / 102 | 0,0000 |
| B0 (VLM hỏi thẳng) | 700 | 36,1 | 36 / 393 | 0,0000 |

## A2 · q11: chuỗi công cụ gốc (B1) vs SVA

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **SVA** | 700 | 74,3 | — | — |
| B1 (chuỗi công cụ) | 700 | 31,4 | 50 / 350 | 0,0000 |
| B0 (VLM hỏi thẳng) | 700 | 31,3 | 47 / 348 | 0,0000 |

## A2 · q13: chuỗi công cụ gốc (B1) vs SVA

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **SVA** | 490 | 46,5 | — | — |
| B1 (chuỗi công cụ) | 490 | 46,3 | 59 / 60 | 1,0000 |
| B0 (VLM hỏi thẳng) | 490 | 45,9 | 34 / 37 | 0,8124 |

## A2 · q14: chuỗi công cụ gốc (B1) vs SVA

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **SVA** | 192 | 60,4 | — | — |
| B1 (chuỗi công cụ) | 192 | 45,3 | 26 / 55 | 0,0019 |
| B0 (VLM hỏi thẳng) | 192 | 56,8 | 15 / 22 | 0,3239 |

## A3 · q9: các cách giải

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **CLIP khớp mềm, 24 frame (SVA)** | 280 | 78,9 | — | — |
| B1: bình luận ×5 rồi so chữ | 280 | 25,7 | 15 / 164 | 0,0000 |
| B0: VLM hỏi thẳng | 280 | 27,5 | 18 / 162 | 0,0000 |
| CLIP + phạt thứ tự λ = 0,1 | 280 | 76,8 | 13 / 19 | 0,3768 |
| CLIP + λ = 0,1 + UniSoccer (thiết kế ban đầu) | 280 | 80,7 | 20 / 15 | 0,4990 |
| CLIP, 4 frame/clip (bằng B0) | 280 | 60,0 | 10 / 63 | 0,0000 |
| chỉ UniSoccer (1 − JS) | 280 | 47,5 | 15 / 103 | 0,0000 |

## A9 · q9: encoder ảnh (24 frame/clip)

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **CLIP ViT-L** | 280 | 78,9 | — | — |
| SigLIP B/16 | 280 | 72,9 | 15 / 32 | 0,0196 |
| DINOv2 B/14 | 280 | 64,6 | 9 / 49 | 0,0000 |
| HSV histogram (không học) | 280 | 61,8 | 19 / 67 | 0,0000 |

## A4 · q14: cách gộp góc quay

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **log pool** | 192 | 60,4 | — | — |
| bỏ phiếu nhãn cứng | 192 | 60,9 | 5 / 4 | 1,0000 |
| linear pool | 192 | 60,4 | 0 / 0 | 1,0000 |
| log pool + độ rõ tự đánh giá | 192 | 60,4 | 0 / 0 | 1,0000 |
| B0: mọi góc trong 1 prompt | 192 | 56,8 | 15 / 22 | 0,3239 |

## A7 · q14 thẻ phạt: categorical vs thang thứ tự

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **severity: log pool** | 11 | 45,5 | — | — |
| severity: thang thứ tự | 11 | 63,6 | 2 / 0 | 0,5000 |

## A8 · q14 (test): trọng số theo loại camera / tốc độ (chữ trong câu hỏi)

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **log pool** | 192 | 60,4 | — | — |
| cận cảnh ×2 | 192 | 59,9 | 1 / 2 | 1,0000 |
| camera chính ×2 | 192 | 63,0 | 5 / 0 | 0,0625 |
| trọng số = hệ số tốc độ | 192 | 60,9 | 3 / 2 | 1,0000 |

## A8 · q14 (dev): trọng số theo loại camera / tốc độ (chữ trong câu hỏi)

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **log pool** | 84 | 63,1 | — | — |
| cận cảnh ×2 | 84 | 64,3 | 1 / 0 | 1,0000 |
| camera chính ×2 | 84 | 61,9 | 0 / 1 | 1,0000 |
| trọng số = hệ số tốc độ | 84 | 64,3 | 1 / 0 | 1,0000 |

## A7 · q14 thẻ phạt (dev): categorical vs thang thứ tự

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **severity: log pool** | 13 | 38,5 | — | — |
| severity: thang thứ tự | 13 | 46,2 | 1 / 0 | 1,0000 |

## A5 · q8: VLM thấy video + phương án vs quyết định từ chữ

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **VLM chấm thẳng phương án (B0)** | 280 | 45,4 | — | — |
| VLM tả bằng chữ → chọn từ chữ | 280 | 45,0 | 46 / 47 | 1,0000 |
| SVA | 280 | 58,9 | 57 / 19 | 0,0000 |

## A5 · q13: VLM thấy video + phương án vs quyết định từ chữ

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **VLM chấm thẳng phương án (B0)** | 490 | 45,9 | — | — |
| VLM tả bằng chữ → chọn từ chữ | 490 | 45,7 | 72 / 73 | 1,0000 |
| SVA | 490 | 46,5 | 37 / 34 | 0,8124 |

## Reasoner · q8: có / không vòng lặp

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **SVA** | 280 | 58,9 | — | — |
| không vòng lặp | 280 | 56,4 | 18 / 25 | 0,3602 |

## Reasoner · q13: có / không vòng lặp

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **SVA** | 490 | 46,5 | — | — |
| không vòng lặp | 490 | 45,5 | 37 / 42 | 0,6527 |

## Thành phần · q8: kênh camera

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **SVA không vòng lặp** | 280 | 56,4 | — | — |
| chỉ tổng thể | 280 | 45,4 | 36 / 67 | 0,0031 |
| chỉ kênh camera đầu/cuối | 280 | 54,6 | 6 / 11 | 0,3323 |

## A6 · đổi backbone 3B → 7B (giữ cấu hình chọn trên dev của 3B, chỉ chọn lại θ)

| hệ | q8 | q9 | q10 | q11 | q13 | q14 | TB |
|---|---|---|---|---|---|---|---|
| B0 | 45,4 → 54,3* | 27,5 → 24,3 | 36,1 → 45,4* | 31,3 → 44,6* | 45,9 → 45,1 | 56,8 → 63,0 | 40,5 → 46,1 |
| B0 + vòng lặp | 52,5 → 60,4* | 25,0 → 24,3 | 37,4 → 48,4* | 37,9 → 47,9* | 46,1 → 48,2 | 57,3 → 65,6 | 42,7 → 49,1 |
| B1 | 45,0 → 52,5 | 25,7 → 23,6 | 74,6 → 79,7* | 31,4 → 37,0* | 46,3 → 43,3 | 45,3 → 52,6 | 44,7 → 48,1 |
| SVA | 58,9 → 69,3* | 78,9 → 78,9 | 87,1 → 87,1 | 74,3 → 72,0 | 46,5 → 48,4 | 60,4 → 64,1 | 67,7 → 70,0 |

`*`: McNemar p < 0,05 giữa bản 3B và 7B của cùng hệ.

## A10 · trước và sau làm sạch dữ liệu

q14 bỏ 24 câu không có đáp án hợp lệ (không chấm được). Trên các câu test được **sửa đáp án**:

| task | hệ | số câu đã sửa | đúng theo đáp án đã sửa | đúng theo đáp án gốc |
|---|---|---|---|---|
| q13 | b0_local | 1 | 0 | 0 |
| q13 | b1_vanilla | 1 | 1 | 0 |
| q13 | b0_api | 1 | 0 | 1 |
| q13 | sva | 1 | 0 | 0 |
| q14 | b0_local | 5 | 0 | 0 |
| q14 | b1_vanilla | 5 | 1 | 0 |
| q14 | b0_api | 5 | 4 | 0 |
| q14 | sva | 5 | 1 | 0 |
