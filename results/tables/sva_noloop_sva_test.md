# Ket qua (test)

| system | task | paper | n | accuracy | 95% CI | ngau nhien | ECE | chon O1 / dap an O1 | loi |
|---|---|---|---|---|---|---|---|---|---|
| sva_noloop | q8 | Q7 | 280 | 56.4 | 50.6-62.1 | 25.0 | 0.045 | 25% / 21% | 0 |
| sva_noloop | q13 | Q12 | 490 | 45.5 | 41.2-49.9 | 25.0 | 0.066 | 34% / 27% | 0 |
| sva_noloop | **trung binh** | | | **51.0** | | | | | |
| sva | q8 | Q7 | 280 | 58.9 | 53.1-64.5 | 25.0 | 0.063 | 24% / 21% | 0 |
| sva | q13 | Q12 | 490 | 46.5 | 42.2-51.0 | 25.0 | 0.059 | 24% / 27% | 0 |
| sva | **trung binh** | | | **52.7** | | | | | |

## McNemar: b1_vanilla vs sva

| task | n | b1_vanilla | sva | b (b1_vanilla sai, sva dung) | c (b1_vanilla dung, sva sai) | chi2 | p |
|---|---|---|---|---|---|---|---|
| q8 | 280 | 45.0 | 58.9 | 82 | 43 | 11.55 | 0.0007 |
| q13 | 490 | 46.3 | 46.5 | 60 | 59 | 0.00 | 1.0000 |
