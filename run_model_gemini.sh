#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --account=aip-vshwartz
#SBATCH --job-name=run_model

source /ubc/cs/home/s/shenranw/scratch/envs/FP_Hallucination/.venv/bin/activate

export HF_HOME=/ubc/cs/home/s/shenranw/scratch/tmp/transformers_cache
export TRITON_CACHE_DIR=/ubc/cs/home/s/shenranw/scratch/tmp/triton_cache

cd /ubc/cs/home/s/shenranw/FP_Hallucination
python run_model.py \
    gemini \
        --model gemini-2.5-flash \
        --operator CREPEPresuppositionExtractionOperator \
        --out_dir /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out

python run_model.py \
    gemini \
        --model gemini-2.5-flash \
        --operator CREPEFeedbackActionOperator \
        --out_dir /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out

python run_model.py \
    evaluate \
        --file out/curated_dataset_gemini-2.5-flash_CREPE_Presupposition_Extraction.jsonl \
        --operator CREPEPresuppositionExtractionOperator \
        --show_top_bottom_k 20

python run_model.py \
    evaluate \
        --file out/curated_dataset_gemini-2.5-flash_CREPE_Feedback_Action.jsonl \
        --operator CREPEFeedbackActionOperator \
        --show_top_bottom_k 20