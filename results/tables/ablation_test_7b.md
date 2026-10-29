# Ablation (tập test, backend qwen_local)

Mọi cấu hình của SVA chọn trên dev; bảng này chỉ đo trên test. McNemar so với dòng **in đậm** của cùng nhóm (b = biến thể đúng & mốc sai, c = ngược lại). Hòa phiếu thì chọn ngẫu nhiên (seed 0) trong các phương án hòa.

## A1 · q8: phân phối vs nhãn cứng

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **phân phối (SVA không vòng lặp)** | 280 | 66,1 | — | — |
| nhãn cứng | 280 | 55,0 | 29 / 60 | 0,0015 |

## A1 · q9: phân phối vs nhãn cứng

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **điểm khớp mềm (SVA)** | 280 | 78,9 | — | — |
| mỗi frame bầu 1 phiếu | 280 | 72,9 | 5 / 22 | 0,0021 |

## A1 · q10: phân phối vs nhãn cứng

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **phân phối (SVA)** | 700 | 87,1 | — | — |
| chỉ nhãn top-1 của UniSoccer | 700 | 71,3 | 11 / 122 | 0,0000 |

## A1 · q11: phân phối vs nhãn cứng

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **phân phối (SVA)** | 700 | 72,0 | — | — |
| top-1 video == top-1 câu bình luận | 700 | 52,7 | 30 / 165 | 0,0000 |

## A1 · q14: phân phối vs nhãn cứng

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **log pool (SVA)** | 192 | 64,1 | — | — |
| bỏ phiếu nhãn cứng | 192 | 65,6 | 10 / 7 | 0,6291 |

## A2 · q8: chuỗi công cụ gốc (B1) vs SVA

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **SVA** | 280 | 69,3 | — | — |
| B1 (chuỗi công cụ) | 280 | 52,5 | 24 / 71 | 0,0000 |
| B0 (VLM hỏi thẳng) | 280 | 54,3 | 17 / 59 | 0,0000 |

## A2 · q9: chuỗi công cụ gốc (B1) vs SVA

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **SVA** | 280 | 78,9 | — | — |
| B1 (chuỗi công cụ) | 280 | 23,6 | 15 / 170 | 0,0000 |
| B0 (VLM hỏi thẳng) | 280 | 24,3 | 16 / 169 | 0,0000 |

## A2 · q10: chuỗi công cụ gốc (B1) vs SVA

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **SVA** | 700 | 87,1 | — | — |
| B1 (chuỗi công cụ) | 700 | 79,7 | 13 / 65 | 0,0000 |
| B0 (VLM hỏi thẳng) | 700 | 45,4 | 29 / 321 | 0,0000 |

## A2 · q11: chuỗi công cụ gốc (B1) vs SVA

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **SVA** | 700 | 72,0 | — | — |
| B1 (chuỗi công cụ) | 700 | 37,0 | 56 / 301 | 0,0000 |
| B0 (VLM hỏi thẳng) | 700 | 44,6 | 66 / 258 | 0,0000 |

## A2 · q13: chuỗi công cụ gốc (B1) vs SVA

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **SVA** | 490 | 48,4 | — | — |
| B1 (chuỗi công cụ) | 490 | 43,3 | 44 / 69 | 0,0240 |
| B0 (VLM hỏi thẳng) | 490 | 45,1 | 20 / 36 | 0,0450 |

## A2 · q14: chuỗi công cụ gốc (B1) vs SVA

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **SVA** | 192 | 64,1 | — | — |
| B1 (chuỗi công cụ) | 192 | 52,6 | 22 / 44 | 0,0097 |
| B0 (VLM hỏi thẳng) | 192 | 63,0 | 24 / 26 | 0,8875 |

## A3 · q9: các cách giải

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **CLIP khớp mềm, 24 frame (SVA)** | 280 | 78,9 | — | — |
| B1: bình luận ×5 rồi so chữ | 280 | 23,6 | 15 / 170 | 0,0000 |
| B0: VLM hỏi thẳng | 280 | 24,3 | 16 / 169 | 0,0000 |
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
| **log pool** | 192 | 64,1 | — | — |
| bỏ phiếu nhãn cứng | 192 | 65,6 | 10 / 7 | 0,6291 |
| linear pool | 192 | 64,1 | 0 / 0 | 1,0000 |
| log pool + độ rõ tự đánh giá | 192 | 65,6 | 4 / 1 | 0,3750 |
| B0: mọi góc trong 1 prompt | 192 | 63,0 | 24 / 26 | 0,8875 |

## A7 · q14 thẻ phạt: categorical vs thang thứ tự

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **severity: log pool** | 11 | 54,5 | — | — |
| severity: thang thứ tự | 11 | 54,5 | 1 / 1 | 1,0000 |

## A8 · q14 (test): trọng số theo loại camera / tốc độ (chữ trong câu hỏi)

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **log pool** | 192 | 64,1 | — | — |
| cận cảnh ×2 | 192 | 66,7 | 6 / 1 | 0,1250 |
| camera chính ×2 | 192 | 65,1 | 7 / 5 | 0,7744 |
| trọng số = hệ số tốc độ | 192 | 67,7 | 7 / 0 | 0,0156 |

## A5 · q8: VLM thấy video + phương án vs quyết định từ chữ

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **VLM chấm thẳng phương án (B0)** | 280 | 54,3 | — | — |
| VLM tả bằng chữ → chọn từ chữ | 280 | 46,8 | 39 / 60 | 0,0444 |
| SVA | 280 | 69,3 | 59 / 17 | 0,0000 |

## A5 · q13: VLM thấy video + phương án vs quyết định từ chữ

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **VLM chấm thẳng phương án (B0)** | 490 | 45,1 | — | — |
| VLM tả bằng chữ → chọn từ chữ | 490 | 45,9 | 64 / 60 | 0,7876 |
| SVA | 490 | 48,4 | 36 / 20 | 0,0450 |

## Reasoner · q8: có / không vòng lặp

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **SVA** | 280 | 69,3 | — | — |
| không vòng lặp | 280 | 66,1 | 12 / 21 | 0,1637 |

## Reasoner · q13: có / không vòng lặp

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **SVA** | 490 | 48,4 | — | — |
| không vòng lặp | 490 | 44,7 | 30 / 48 | 0,0542 |

## Thành phần · q8: kênh camera

| biến thể | n | accuracy | b / c | p |
|---|---|---|---|---|
| **SVA không vòng lặp** | 280 | 66,1 | — | — |
| chỉ tổng thể | 280 | 54,3 | 30 / 63 | 0,0009 |
| chỉ kênh camera đầu/cuối | 280 | 65,0 | 2 / 5 | 0,4531 |

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
| q13 | b0_local | 1 | 1 | 0 |
| q13 | b1_vanilla | 1 | 1 | 0 |
| q13 | b0_api | 1 | 0 | 1 |
| q13 | sva | 1 | 1 | 0 |
| q14 | b0_local | 5 | 1 | 0 |
| q14 | b1_vanilla | 5 | 0 | 0 |
| q14 | b0_api | 5 | 4 | 0 |
| q14 | sva | 5 | 1 | 0 |
