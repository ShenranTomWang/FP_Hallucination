#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:0:0
#SBATCH --partition=nlpgpo

source /ubc/cs/home/s/shenranw/.bashrc
source .venv/bin/activate

cd /ubc/cs/home/s/shenranw/FP_Hallucination
python run_model.py \
    transformers \
        --dataset_dir /ubc/cs/home/s/shenranw/scratch/datasets \
        --model ${HF_HOME}/Qwen2.5-7B-Instruct \
        --operator CREPEPresuppositionExtractionOperator \
        --out_dir /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out

python run_model.py \
    align_responses \
    --file /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out/curated_dataset_Qwen2.5-7B-Instruct.jsonl \
    --operator CREPEPresuppositionExtractionOperator \
    --model_type ${HF_HOME}/roberta-large