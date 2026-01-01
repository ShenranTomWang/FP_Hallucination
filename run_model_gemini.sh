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
    gemini \
        --model gemini-2.5-flash \
        --batched_job \
        --operator CREPEPresuppositionExtractionOperator

python run_model.py \
    gemini_check \
        --model gemini-2.5-flash \
        --operator CREPEPresuppositionExtractionOperator

python run_model.py \
    gemini \
        --model gemini-2.5-flash \
        --batched_job \
        --operator CREPEFeedbackActionOperator

python run_model.py \
    gemini_check \
        --model gemini-2.5-flash \
        --operator CREPEFeedbackActionOperator

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