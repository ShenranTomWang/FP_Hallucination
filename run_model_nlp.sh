#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:0:0
#SBATCH --partition=nlpgpo

source /ubc/cs/home/s/shenranw/.bashrc

cd /ubc/cs/home/s/shenranw/FP_Hallucination
source ../scratch/envs/FP_Hallucination/.venv/bin/activate

MODEL="Qwen2.5-1.5B-Instruct"

python run_model.py \
    transformers \
        --dataset_dir /ubc/cs/home/s/shenranw/scratch/datasets \
        --model ${HF_HOME}/${MODEL} \
        --operator CREPEPresuppositionExtractionOperator \
        --out_dir /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out

python run_model.py \
    align_responses \
    --file /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out/curated_dataset_${MODEL}.jsonl \
    --operator CREPEPresuppositionExtractionOperator \
    --model_type ${HF_HOME}/roberta-large

python run_model.py \
    evaluate \
    --file /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out/curated_dataset_${MODEL}.jsonl \
    --operator CREPEPresuppositionExtractionOperator \
    --show_top_bottom_k 20 \
    --use_aligned \
    --run_bert_score