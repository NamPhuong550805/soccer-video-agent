#!/bin/bash
# Cai moi truong tren may dang nhap HPC (chi tai va cai, khong chay tinh toan).
# Chay trong tmux (server dung tien trinh nen khi thoat SSH); chay lai thi tai tiep phan con thieu:
#   tmux new -s nam
#   cd /datastore/$USER/nam_sva/soccer-video-agent && bash scripts/hpc/setup.sh 2>&1 | tee logs/setup.log
# Dang nhap Hugging Face truoc (hf auth login) de khong bi gioi han luot tai.
# Dung luong du kien ~36 GB: video 6 task 12.4 GB, Qwen2.5-VL-7B 16.6 GB, venv ~6-7 GB.
set -euo pipefail
ROOT=/datastore/$USER/nam_sva
cd "$(dirname "$0")/../.."

# khong giu bo nho dem cua pip; model de trong /datastore, khong de vao /home (gioi han 10GB)
export PIP_NO_CACHE_DIR=1 HF_HOME=$ROOT/hf_cache HF_HUB_DISABLE_XET=1

[ -d "$ROOT/venv" ] || python3 -m venv "$ROOT/venv"
source "$ROOT/venv/bin/activate"
pip install -q -U pip

# cung phien ban voi may local (opencv cung ban de frame giong het)
pip install -q torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu126
pip install -q transformers==5.18.0 accelerate==1.13.0 bitsandbytes==0.50.2 \
    opencv-contrib-python-headless==4.11.0.86 numpy==2.4.4 scipy==1.16.3 open_clip_torch==3.3.0 \
    scenedetect==0.7.1 einops==0.8.2 timm==1.0.22 pyyaml tqdm python-dotenv pandas statsmodels pytest
pip install -q -e . --no-deps

# code goc SoccerAgent (prompt, anh mau cua Camera Detection) - nho
[ -d external/SoccerAgent ] || git clone -q https://github.com/jyrao/SoccerAgent external/SoccerAgent
git -C external/SoccerAgent checkout -q 763c254
# Action Classifier (UniSoccer) va embedding CLIP khong phu thuoc VLM: ket qua da tinh o may local,
# nam trong data/cache/ cua goi code -> khong tai checkpoint. Neu log bao thieu cache moi tai:
#   hf download Homie0609/UniSoccer pretrained_classification.pth --local-dir external/unisoccer
#   hf download google/siglip-base-patch16-224

# du lieu 6 task video (~12.4 GB) va model 7B
hf download Homie0609/SoccerBench --repo-type dataset --local-dir data/raw/datasets/SoccerBench --max-workers 4 \
    --include "qa/*" --include "materials/q8/*" --include "materials/q9/*" --include "materials/q10/*" \
    --include "materials/q11/*" --include "materials/q13/*" --include "materials/q14/*"
hf download Qwen/Qwen2.5-VL-7B-Instruct

echo "xong. du lieu: $(du -sh data/raw/datasets/SoccerBench | cut -f1), model: $(du -sh $HF_HOME | cut -f1), venv: $(du -sh $ROOT/venv | cut -f1)"
echo "tong /datastore/$USER/nam_sva: $(du -sh $ROOT | cut -f1)"
