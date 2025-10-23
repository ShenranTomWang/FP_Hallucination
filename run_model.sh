#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --ntasks=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --account=st-hgonen-1-gpu
#SBATCH --constraint=gpu_mem_32
#SBATCH --job-name=run_model

source /home/shenranw/.bashrc
source /home/shenranw/scratch/envs/llm/bin/activate

export HF_HOME=/home/shenranw/scratch/tmp/transformers_cache
export TRITON_CACHE_DIR=/home/shenranw/scratch/tmp/triton_cache

cd /home/shenranw/projects/def-vshwartz/shenranw/FP_Hallucination/
python -m data_gen.run_model \
    --dataset CREPE \
    --model_name_or_path /home/shenranw/scratch/models/Qwen2.5-1.5B-Instruct