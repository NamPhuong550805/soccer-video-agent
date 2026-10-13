#!/bin/bash
# Xem tien do cac job nam_ cua du an (chi doc, khong tinh toan):
#   bash scripts/hpc/status.sh                 # xem 1 lan
#   watch -n 60 bash scripts/hpc/status.sh     # tu cap nhat moi 60 giay (Ctrl+C de thoat)
cd "$(dirname "$0")/../.."
echo "== job cua du an (R = dang chay, PD = dang cho; Dependency = cho job truoc, Priority = cho GPU)"
squeue -u "$USER" -o "%.8i %.14j %.2t %.10M %.22R" | awk 'NR == 1 || $2 ~ /^nam_/'
echo "== job dang chay cua ca tai khoan (moi nhom): $(squeue -u "$USER" -h -t R | wc -l) / 3"
for id in $(squeue -u "$USER" -h -t R -o "%i %j" | awk '$2 ~ /^nam_/ {print $1}'); do
  name=$(squeue -j "$id" -h -o %j)
  echo "== $name ($id) dang o:"
  tail -c 3000 "logs/${name}_${id}.err" 2>/dev/null | tr '\r' '\n' | grep -E "it/s|s/it|it\]" | tail -1
done
echo "== so tay chay: 5 dong cuoi (moi job xong ghi 1 dong)"
tail -5 results/RUNS.md 2>/dev/null | cut -c1-230
echo "== job nam_ da ket thuc hom nay"
sacct -u "$USER" -S today -n -X -o JobID,JobName%14,State,Elapsed 2>/dev/null | awk '$2 ~ /^nam_/'
echo "== dung luong du an: $(du -sh "/datastore/$USER/nam_sva" 2>/dev/null | cut -f1)"
