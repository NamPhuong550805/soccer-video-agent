# Chi phí và thời gian mỗi câu (tập test, Qwen2.5-VL-3B-Instruct 4-bit)

Thời gian = độ trễ đo khi chạy (giây/câu), **chưa gồm** giải mã frame (cache dùng chung cho mọi hệ); SVA lấy lần chạy đầu (chưa có cache phân phối sự kiện của câu bình luận). Lượt gọi model là ước lượng theo cấu trúc của từng hệ.

| Task | B0-local: giây / lượt gọi | B0 + vòng lặp: giây / lượt gọi | B0-api (Gemini): giây / lượt gọi | B1-vanilla: giây / lượt gọi | SVA: giây / lượt gọi | SVA: tính trước |
|---|---|---|---|---|---|---|
| q8 | 1,3 / 1,0 | 3,6 / 2,8 | 22,1 / 1,0 | 58,9 / 10,0 | 3,6 / 4,0 | — |
| q9 | 3,9 / 1,0 | 14,1 / 2,9 | 34,4 / 1,0 | 116,9 / 14,0 | 0,6 / 0,0 | embedding CLIP 5 clip × 24 frame ≈ 7 s |
| q10 | 1,1 / 1,0 | 3,7 / 2,7 | 32,3 / 1,0 | 24,6 / 6,0 | 0,0 / 1,0 | UniSoccer 1 clip ≈ 0,2 s |
| q11 | 1,1 / 1,0 | 4,4 / 3,1 | 49,0 / 1,0 | 23,6 / 6,0 | 0,3 / 5,0 | UniSoccer 1 clip ≈ 0,2 s |
| q13 | 1,1 / 1,0 | 4,0 / 3,1 | 65,6 / 1,0 | 33,1 / 6,0 | 3,7 / 3,1 | — |
| q14 | 11,6 / 1,0 | 28,0 / 2,4 | 46,8 / 1,0 | 30,1 / 6,0 | 5,5 / 4,6 | — |

- B0-api: gọi Gemini 3.5 Flash-Lite qua gói miễn phí (độ trễ gồm cả mạng và giới hạn tốc độ); chi phí tiền thực tế 0.
- B0-local, B1, SVA chạy tại chỗ, không tốn tiền API; dữ liệu không rời máy.
- SVA ở q9/q10/q11 gần như không gọi VLM: phần nặng là tính trước embedding / Action Classifier (cột cuối), làm một lần cho mỗi clip.
