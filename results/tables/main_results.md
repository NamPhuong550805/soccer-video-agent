# Kết quả chính (tập test, Qwen2.5-VL-3B-Instruct 4-bit)

| Task | n | B0-local | B0 + vòng lặp | B0-api (Gemini) | B1-vanilla | SVA | Δ SVA − B1 | SVA đúng & B1 sai / ngược lại | p (McNemar) | paper: Qwen2.5-VL-7B | paper: SoccerAgent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q8 (Q7) | 280 | 45,4 | 52,5 | **68,9** | 45,0 | 58,9 | 13,9 | 82 / 43 | < 0,001 | 56,8 | 51,1 |
| q9 (Q8) | 280 | 27,5 | 25,0 | 32,5 | 25,7 | **78,9** | 53,2 | 164 / 15 | < 0,001 | 31,6 | 35,7 |
| q10 (Q9) | 700 | 36,1 | 37,4 | 64,4 | 74,6 | **87,1** | 12,6 | 102 / 14 | < 0,001 | 52,2 | 85,0 |
| q11 (Q10) | 700 | 31,3 | 37,9 | 70,1 | 31,4 | **74,3** | 42,9 | 350 / 50 | < 0,001 | 51,6 | 72,9 |
| q13 (Q12) | 490 | 45,9 | 46,1 | **48,8** | 46,3 | 46,5 | 0,2 | 60 / 59 | 1,000 | 46,9 | 46,0 |
| q14 (Q13) | 192 | 56,8 | 57,3 | **79,2** | 45,3 | 60,4 | 15,1 | 55 / 26 | 0,002 | 50,7 | 55,5 |
| **Trung bình 6 task** |  | 40,5 | 42,7 | 60,7 | 44,7 | 67,7 | 23,0 |  |  | 48,3 | 57,7 |
| **Trung bình bỏ q10, q11** |  | 43,9 | 45,2 | 57,3 | 40,6 | 61,2 | 20,6 |  |  | 46,5 | 47,1 |

- n: số câu test (chia dev/test theo nhóm nội dung; q14 còn 276 câu sau khi bỏ 24 câu lỗi, test 192).
- B0-local, B1, SVA dùng cùng Qwen2.5-VL-3B-Instruct 4-bit và cùng frame (8 frame một clip, tối đa 24 frame một câu).
- Cột paper: số công bố của SoccerAgent (Qwen2.5-VL-7B, cả video, toàn bộ 10 nghìn câu): **khác giao thức, chỉ để tham khảo, không so trực tiếp**.
- q10, q11 dùng Action Classifier của UniSoccer, được huấn luyện trên SoccerReplay-1988 — nguồn của clip q10: có thể trùng dữ liệu huấn luyện công cụ; so với B1 vẫn công bằng (cùng công cụ).
