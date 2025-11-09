#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --account=st-hgonen-1-gpu
#SBATCH --constraint=gpu_mem_32
#SBATCH --output=run_model.log
#SBATCH --error=run_model.log
#SBATCH --mail-user=shenranw@student.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=run_model

source /home/shenranw/.bashrc
conda activate FP
module load http_proxy

export HF_HOME=/home/shenranw/scratch-hgonen/tmp/transformers_cache
export TRITON_CACHE_DIR=/home/shenranw/scratch-hgonen/tmp/triton_cache

cd /home/shenranw/FP_Hallucination
python run_model.py \
    transformers \
        --dataset_dir /home/shenranw/scratch-hgonen/datasets \
        --model /scratch/st-hgonen-1/shenranw/models/Qwen2.5-1.5B-Instruct \
        --operator CREPEPresuppositionExtractionOperator \
        --out_dir /home/shenranw/scratch-hgonen/FP_Hallucination/out

python run_model.py \
    align_responses \
    --file /home/shenranw/scratch-hgonen/FP_Hallucination/out/curated_dataset_Qwen2.5-1.5B-Instruct.jsonl \
    --operator CREPEPresuppositionExtractionOperator \
    --model_type /home/shenranw/scratch-hgonen/models/roberta-large