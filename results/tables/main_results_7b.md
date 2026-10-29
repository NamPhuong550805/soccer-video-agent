# Kết quả chính (tập test, Qwen2.5-VL-7B-Instruct)

| Task | n | B0-local | B0 + vòng lặp | B0-api (Gemini) | B1-vanilla | SVA | Δ SVA − B1 | SVA đúng & B1 sai / ngược lại | p (McNemar) | paper: Qwen2.5-VL-7B | paper: SoccerAgent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q8 (Q7) | 280 | 54,3 | 60,4 | 68,9 | 52,5 | **69,3** | 16,8 | 71 / 24 | < 0,001 | 56,8 | 51,1 |
| q9 (Q8) | 280 | 24,3 | 24,3 | 32,5 | 23,6 | **78,9** | 55,4 | 170 / 15 | < 0,001 | 31,6 | 35,7 |
| q10 (Q9) | 700 | 45,4 | 48,4 | 64,4 | 79,7 | **87,1** | 7,4 | 65 / 13 | < 0,001 | 52,2 | 85,0 |
| q11 (Q10) | 700 | 44,6 | 47,9 | 70,1 | 37,0 | **72,0** | 35,0 | 301 / 56 | < 0,001 | 51,6 | 72,9 |
| q13 (Q12) | 490 | 45,1 | 48,2 | **48,8** | 43,3 | 48,4 | 5,1 | 69 / 44 | 0,024 | 46,9 | 46,0 |
| q14 (Q13) | 192 | 63,0 | 65,6 | **79,2** | 52,6 | 64,1 | 11,5 | 44 / 22 | 0,010 | 50,7 | 55,5 |
| **Trung bình 6 task** |  | 46,1 | 49,1 | 60,7 | 48,1 | 70,0 | 21,9 |  |  | 48,3 | 57,7 |
| **Trung bình bỏ q10, q11** |  | 46,7 | 49,6 | 57,3 | 43,0 | 65,2 | 22,2 |  |  | 46,5 | 47,1 |

- n: số câu test (chia dev/test theo nhóm nội dung; q14 còn 276 câu sau khi bỏ 24 câu lỗi, test 192).
- B0-local, B1, SVA dùng cùng Qwen2.5-VL-7B-Instruct và cùng frame (8 frame một clip, tối đa 24 frame một câu).
- Cột paper: số công bố của SoccerAgent (Qwen2.5-VL-7B, cả video, toàn bộ 10 nghìn câu): **khác giao thức, chỉ để tham khảo, không so trực tiếp**.
- q10, q11 dùng Action Classifier của UniSoccer, được huấn luyện trên SoccerReplay-1988 — nguồn của clip q10: có thể trùng dữ liệu huấn luyện công cụ; so với B1 vẫn công bằng (cùng công cụ).
