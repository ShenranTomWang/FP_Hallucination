#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:0:0
#SBATCH --partition=nlpgpo

source /ubc/cs/home/s/shenranw/.bashrc

cd /ubc/cs/home/s/shenranw/FP_Hallucination
source ../scratch/envs/FP_Hallucination/.venv/bin/activate

MODEL="Llama-3.2-3B-Instruct"

python run_model.py \
    transformers \
        --dataset_path /ubc/cs/home/s/shenranw/scratch/datasets/CREPE/dev.jsonl \
        --model ${HF_HOME}/${MODEL} \
        --operator CREPEDirectQAOperator \
        --out_dir /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out

python run_model.py \
    transformers \
        --dataset_path /ubc/cs/home/s/shenranw/scratch/datasets/CREPE/dev.jsonl \
        --model ${HF_HOME}/${MODEL} \
        --operator CREPEPresuppositionExtractionOperator \
        --out_dir /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out

python run_model.py \
    transformers \
        --dataset_path /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out/curated_dataset_${MODEL}_CREPE_Presupposition_Extraction.jsonl \
        --model ${HF_HOME}/${MODEL} \
        --operator CREPEFeedbackActionOperator \
        --out_dir /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out

python run_model.py \
    transformers \
        --dataset_path /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out/curated_dataset_${MODEL}_CREPE_Feedback_Action.jsonl \
        --model ${HF_HOME}/${MODEL} \
        --operator CREPEFinalAnswerOperator \
        --out_dir /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out

python run_model.py \
    align_responses \
    --file /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out/curated_dataset_${MODEL}_CREPE_Presupposition_Extraction.jsonl \
    --operator CREPEPresuppositionExtractionOperator \
    --model_type ${HF_HOME}/roberta-large

python run_model.py \
    evaluate \
    --file /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out/curated_dataset_${MODEL}_CREPE_Presupposition_Extraction.jsonl \
    --operator CREPEPresuppositionExtractionOperator \
    --show_top_bottom_k 20 \
    --run_bert_score

python run_model.py \
    evaluate \
    --file /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out/curated_dataset_${MODEL}_CREPE_Final_Answer.jsonl \
    --operator CREPEFinalAnswerOperator \
    --show_top_bottom_k 20 \
    --run_bert_score

python run_model.py \
    evaluate \
    --file /ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out/curated_dataset_${MODEL}_CREPE_Direct_QA.jsonl \
    --operator CREPEDirectQAOperator \
    --show_top_bottom_k 20 \
    --run_bert_score