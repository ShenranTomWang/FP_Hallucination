#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:0:0
#SBATCH --partition=nlpgpo

source /ubc/cs/home/s/shenranw/scratch/envs/FP_Hallucination/.venv/bin/activate

export HF_HOME=/ubc/cs/home/s/shenranw/scratch/tmp/transformers_cache
export TRITON_CACHE_DIR=/ubc/cs/home/s/shenranw/scratch/tmp/triton_cache
export GOOGLE_API_KEY=AQ.Ab8RN6LowP5wKbTJrGhnpW82W-5IpG3fLEosG94PPvvMEaD0_w

cd /ubc/cs/home/s/shenranw/FP_Hallucination
python run_model.py \
    gemini \
        --model gemini-2.5-flash \
        --operator CREPEPresuppositionExtractionOperator \
        --out_dir /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out \
        --dataset_path /ubc/cs/home/s/shenranw/scratch/datasets/CREPE/dev.jsonl

python run_model.py \
    gemini \
        --model gemini-2.5-flash \
        --operator CREPEDirectQAOperator \
        --out_dir /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out \
        --dataset_path /ubc/cs/home/s/shenranw/scratch/datasets/CREPE/dev.jsonl

python run_model.py \
    align_responses \
        --file /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out/curated_dataset_gemini-2.5-flash_CREPE_Presupposition_Extraction.jsonl \
        --operator CREPEPresuppositionExtractionOperator \
        --model_type ${HF_HOME}/roberta-large

python run_model.py \
    gemini \
        --model gemini-2.5-flash \
        --operator CREPEFeedbackActionOperator \
        --out_dir /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out \
        --dataset_path /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out/curated_dataset_gemini-2.5-flash_CREPE_Presupposition_Extraction.jsonl

python run_model.py \
    gemini \
        --model gemini-2.5-flash \
        --operator CREPEFinalAnswerOperator \
        --out_dir /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out \
        --dataset_path /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out/curated_dataset_gemini-2.5-flash_CREPE_Feedback_Action.jsonl

python run_model.py \
    evaluate \
        --file /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out/curated_dataset_gemini-2.5-flash_CREPE_Presupposition_Extraction.jsonl \
        --operator CREPEPresuppositionExtractionOperator \
        --show_top_bottom_k 20 \
        --run_bert_score

python run_model.py \
    evaluate \
        --file /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out/curated_dataset_gemini-2.5-flash_CREPE_Final_Answer.jsonl \
        --operator CREPEFinalAnswerOperator \
        --show_top_bottom_k 20 \
        --run_bert_score \
        --run_fp_score

python run_model.py \
    evaluate \
        --file /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out/curated_dataset_gemini-2.5-flash_CREPE_Direct_QA.jsonl \
        --operator CREPEDirectQAOperator \
        --show_top_bottom_k 20 \
        --run_bert_score \
        --run_fp_score