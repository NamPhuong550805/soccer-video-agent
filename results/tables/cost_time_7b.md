# Chi phí và thời gian mỗi câu (tập test, Qwen2.5-VL-7B-Instruct)

Thời gian = độ trễ đo khi chạy (giây/câu), **chưa gồm** giải mã frame (cache dùng chung cho mọi hệ); SVA lấy lần chạy đầu (chưa có cache phân phối sự kiện của câu bình luận). Lượt gọi model là ước lượng theo cấu trúc của từng hệ.

| Task | B0-local: giây / lượt gọi | B0 + vòng lặp: giây / lượt gọi | B0-api (Gemini): giây / lượt gọi | B1-vanilla: giây / lượt gọi | SVA: giây / lượt gọi | SVA: tính trước |
|---|---|---|---|---|---|---|
| q8 | 0,3 / 1,0 | 0,6 / 1,9 | 22,1 / 1,0 | 16,6 / 10,0 | 0,7 / 3,7 | — |
| q9 | 0,8 / 1,0 | 2,0 / 2,9 | 34,4 / 1,0 | 27,5 / 14,0 | 0,0 / 0,0 | embedding CLIP 5 clip × 24 frame ≈ 7 s |
| q10 | 0,4 / 1,0 | 1,0 / 2,2 | 32,3 / 1,0 | 7,3 / 6,0 | 0,0 / 1,0 | UniSoccer 1 clip ≈ 0,2 s |
| q11 | 0,5 / 1,0 | 1,0 / 1,9 | 49,0 / 1,0 | 9,3 / 6,0 | 0,1 / 5,0 | UniSoccer 1 clip ≈ 0,2 s |
| q13 | 0,3 / 1,0 | 0,6 / 2,1 | 65,6 / 1,0 | 7,4 / 6,0 | 1,0 / 3,3 | — |
| q14 | 0,6 / 1,0 | 1,3 / 2,0 | 46,8 / 1,0 | 8,5 / 6,0 | 1,2 / 4,6 | — |

- B0-api: gọi Gemini 3.5 Flash-Lite qua gói miễn phí (độ trễ gồm cả mạng và giới hạn tốc độ); chi phí tiền thực tế 0.
- B0-local, B1, SVA chạy tại chỗ, không tốn tiền API; dữ liệu không rời máy.
- SVA ở q9/q10/q11 gần như không gọi VLM: phần nặng là tính trước embedding / Action Classifier (cột cuối), làm một lần cho mỗi clip.
