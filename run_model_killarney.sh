#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=l40s:1
#SBATCH --ntasks=4
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --account=aip-vshwartz
#SBATCH --job-name=run_model

source /home/shenranw/scratch/envs/llm/bin/activate

export HF_HOME=/home/shenranw/scratch/tmp/transformers_cache
export TRITON_CACHE_DIR=/home/shenranw/scratch/tmp/triton_cache

cd /home/shenranw/projects/aip-vshwartz/shenranw/FP_Hallucination
python run_model.py \
    --dataset CREPE \
    --model_name_or_path /home/shenranw/scratch/models/Qwen2.5-1.5B-Instruct