import json

with open('/ubc/cs/home/s/shenranw/scratch/FP_Hallucination/out/curated_dataset_Qwen2.5-1.5B-Instruct.jsonl', 'r') as f:
    data = [json.loads(line.strip()) for line in f]

print(all(len(dp['model_detected_presuppositions']['presuppositions']) > 0 for dp in data))
print(all(len(dp['model_detected_presuppositions_aligned_recall']) == 0 for dp in data if len(dp['presuppositions']) == 0))