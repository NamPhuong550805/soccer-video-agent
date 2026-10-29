#!/bin/bash
#SBATCH --job-name=nam_sva
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=uit
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2g.35gb:1
# Chay 1 script python cua du an tren GPU. Goi tu thu muc du an, vd:
#   sbatch --job-name=nam_b0 scripts/hpc/job.sh scripts/03_run.py --system b0_local --task all --backend qwen_local
ROOT=/datastore/$USER/nam_sva
source /datastore/$USER/envs/sva-py312-torch2.12-cu126/bin/activate
# model o thu muc chung; SVA_FRAME_CACHE=0: frame chi giu trong RAM, khong ghi cache ra dia (cache frame local ~22 GB)
export HF_HOME=$ROOT/hf_cache HF_HUB_CACHE=/datastore/$USER/models/hf/hub HF_HUB_OFFLINE=1 \
       PYTHONPATH=src PYTHONUNBUFFERED=1 SVA_FRAME_CACHE=0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "$@"
