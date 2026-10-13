#!/bin/bash
# Cai moi truong tren may dang nhap HPC (chi tai va cai, khong chay tinh toan).
# Chay trong tmux (server dung tien trinh nen khi thoat SSH); chay lai thi tai tiep phan con thieu:
#   tmux new -s nam2
#   cd /datastore/$USER/nam_sva/soccer-video-agent && bash scripts/hpc/setup.sh 2>&1 | tee logs/setup.log
# Dang nhap Hugging Face truoc (hf auth login) de khong bi gioi han luot tai.
#
# Bo cuc theo yeu cau cua co (cac nhom dung chung, khong cai trung):
#   /datastore/$USER/envs/sva-py312-torch2.12-cu126   moi truong Python (~6-7 GB), nhom khac cung phien ban dung lai duoc
#   /datastore/$USER/models/hf/hub                     model Hugging Face dung chung (Qwen2.5-VL-7B 16.6 GB)
#   /datastore/$USER/nam_sva                           rieng du an: code, video 6 task 12.4 GB, ket qua, token HF
set -euo pipefail
ROOT=/datastore/$USER/nam_sva
ENV=/datastore/$USER/envs/sva-py312-torch2.12-cu126
MODELS=/datastore/$USER/models/hf/hub
cd "$(dirname "$0")/../.."
say() { echo; echo "===== $(date +%H:%M) $*"; }

# khong giu bo nho dem cua pip; file tam de trong datastore (khong dung /tmp, /home: gioi han 10 GB)
export PIP_NO_CACHE_DIR=1 HF_HOME=$ROOT/hf_cache HF_HUB_CACHE=$MODELS HF_HUB_DISABLE_XET=1 TMPDIR=$ROOT/tmp
mkdir -p "$TMPDIR" "$MODELS" "$(dirname "$ENV")"

say "1/5 moi truong Python: $ENV"
[ -d "$ENV" ] || python3 -m venv "$ENV"
source "$ENV/bin/activate"
pip install -q -U pip

# cung phien ban voi may local (opencv cung ban de frame giong het); khong cai goi sva vao env chung (dung PYTHONPATH=src)
say "2/5 cai PyTorch 2.12 (tai ~2.5 GB, 5-15 phut)"
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu126
say "3/5 cai cac thu vien con lai (1-3 phut)"
pip install transformers==5.18.0 accelerate==1.13.0 bitsandbytes==0.50.2 \
    opencv-contrib-python-headless==4.11.0.86 numpy==2.4.4 scipy==1.16.3 open_clip_torch==3.3.0 \
    scenedetect==0.7.1 einops==0.8.2 timm==1.0.22 pyyaml tqdm python-dotenv pandas statsmodels pytest

cat > "$ENV/README_nam_sva.txt" <<EOF
sva-py312-torch2.12-cu126: Python 3.12, torch 2.12.0 + torchvision 0.27.0 (CUDA 12.6), transformers 5.18.0,
accelerate 1.13.0, bitsandbytes 0.50.2, opencv-contrib-python-headless 4.11.0.86, numpy 2.4.4, open_clip_torch 3.3.0.
Du an: nam_sva (Nguyen Phuong Nam). Nhom khac dung duoc neu hop phien ban: source $ENV/bin/activate
Vui long KHONG pip install / nang cap goi trong env nay (se lam doi ket qua cua du an); can goi khac thi tao env rieng.
EOF

# code goc SoccerAgent (prompt, anh mau cua Camera Detection) - nho
[ -d external/SoccerAgent ] || git clone -q https://github.com/jyrao/SoccerAgent external/SoccerAgent
git -C external/SoccerAgent checkout -q 763c254
# Action Classifier (UniSoccer) va embedding CLIP khong phu thuoc VLM: ket qua da tinh o may local,
# nam trong data/cache/ cua goi code -> khong tai checkpoint. Neu log bao thieu cache moi tai:
#   hf download Homie0609/UniSoccer pretrained_classification.pth --local-dir external/unisoccer
#   hf download google/siglip-base-patch16-224

# du lieu 6 task video (~12.4 GB, rieng du an) va model 7B (thu muc model chung)
say "4/5 tai video 6 task (12.4 GB, 5640 file, 20-40 phut)"
hf download Homie0609/SoccerBench --repo-type dataset --local-dir data/raw/datasets/SoccerBench --max-workers 4 \
    --include "qa/*" --include "materials/q8/*" --include "materials/q9/*" --include "materials/q10/*" \
    --include "materials/q11/*" --include "materials/q13/*" --include "materials/q14/*"
say "5/5 tai Qwen2.5-VL-7B vao thu muc model chung (16.6 GB, 10-30 phut)"
hf download Qwen/Qwen2.5-VL-7B-Instruct

[ -f /datastore/$USER/models/README.txt ] || cat > /datastore/$USER/models/README.txt <<EOF
Model Hugging Face dung chung. Dung: export HF_HUB_CACHE=$MODELS (giu HF_HOME rieng cua minh de khong dung chung token).
Hien co: Qwen/Qwen2.5-VL-7B-Instruct (tai boi nam_sva). Tai them model: HF_HUB_CACHE=$MODELS hf download <ten model>
Truoc khi xoa model nao, hoi cac nhom dang dung.
EOF

rm -rf "$TMPDIR"
say "XONG"
echo "xong. du lieu: $(du -sh data/raw/datasets/SoccerBench | cut -f1), model chung: $(du -sh $MODELS | cut -f1), env: $(du -sh $ENV | cut -f1)"
echo "tong /datastore/$USER: $(du -sh /datastore/$USER | cut -f1) (rieng nam_sva: $(du -sh $ROOT | cut -f1))"
