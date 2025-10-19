#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --account=def-vshwartz
#SBATCH --job-name=curate_model_dataset

source /home/shenranw/.bashrc
source /home/shenranw/scratch/envs/llm/bin/activate

cd /home/shenranw/projects/def-vshwartz/shenranw/FP_Hallucination/
python -m data_gen.curate_model_dataset \
    --dataset_path /home/shenranw/projects/def-vshwartz/shenranw/FP_Hallucination/data/toy_datasets/movies/wikidata_movies.jsonl \
    --model_name_or_path /home/shenranw/scratch/tmp/transformers_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct \
    --operator Qwen2Operator