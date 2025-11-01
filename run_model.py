from transformers import AutoTokenizer, AutoModelForCausalLM
import json, argparse, os, random, time
import torch
import openai
from data_gen.data_loader import instantiate_dataloader, DataLoader
from tqdm import tqdm
import data_gen.template as template
import evaluator
import numpy as np

def main(args):
    if args.command == 'transformers':
        args.dtype = getattr(torch, args.dtype)
        args.device = torch.device(args.device)
    if args.command in ['transformers', 'openai', 'openai_check']:
        args.out_file = args.out_file.format(args.model.split('/')[-1])

    if args.command == 'evaluate':
        run_evaluate(args)
        return

    data_loader = instantiate_dataloader(dataset_name=args.dataset, file_dir=args.dataset_dir)
    dataset_full = data_loader.load_data(split=args.split)
    dataset = dataset_full[args.start_idx:]
    if dataset[0].get('few_shot_data') is None:
        few_shot_data = data_loader.load_data(split='train')
        few_shot_data = [data for data in few_shot_data if len(data['presuppositions']) != 0]
        few_shot_data = random.sample(few_shot_data, args.k)
        for data in dataset_full:
            data['few_shot_data'] = few_shot_data
        data_loader.save_data(dataset_full, split=args.split)
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    
    if args.command == 'transformers':
        run_transformers_model(args, dataset, data_loader)
    elif args.command == 'openai':
        run_openai_model(args, dataset, data_loader)
    elif args.command == 'openai_check':
        run_openai_model_check(args, dataset)
        
def run_evaluate(args):
    with open(args.file, 'r') as f:
        data = [json.loads(line.strip()) for line in f]
    evaluator_class: evaluator.Evaluator = getattr(evaluator, args.evaluator)
    evaluators = [evaluator_class(**dp) for dp in data if dp.get('model_answer') is not None]
    for i, ev in tqdm(enumerate(evaluators), desc='Evaluating'):
        rouge1_f1 = ev.evaluate_rouge1_f1()
        rougeL_f1 = ev.evaluate_rougeL_f1()
        data[i]['rouge1_f1'] = rouge1_f1
        data[i]['rougeL_f1'] = rougeL_f1
        if args.run_bleurt:
            bleurt_f1 = ev.evaluate_bleurt_f1()
            data[i]['bleurt_f1'] = bleurt_f1
    with open(args.file, 'w') as f:
        for dp in data:
            f.write(json.dumps(dp) + '\n')
    avg_rouge1 = np.mean([dp['rouge1_f1'] for dp in data if dp.get('rouge1_f1') is not None])
    avg_rougeL = np.mean([dp['rougeL_f1'] for dp in data if dp.get('rougeL_f1') is not None])
    print(f'Average ROUGE-1 F1: {avg_rouge1:.4f}')
    print(f'Average ROUGE-L F1: {avg_rougeL:.4f}')
    if args.run_bleurt:
        avg_bleurt = np.mean([dp['bleurt_f1'] for dp in data if dp.get('bleurt_f1') is not None])
        print(f'Average BLEURT F1: {avg_bleurt:.4f}')
    
    if args.show_top_bottom_k > 0:
        k = args.show_top_bottom_k
        for score_key in ['rouge1_f1', 'rougeL_f1'] + (['bleurt_f1'] if args.run_bleurt else []):
            _save_top_bottom_k(data, score_key, k, os.path.dirname(args.file))

def _save_top_bottom_k(data: list, score_key: str, k: int, out_dir: str):
    sorted_data = sorted(
        [dp for dp in data if dp.get("model_answer") is not None and dp.get(score_key) is not None],
        key=lambda x: x[score_key]
    )
    with open(os.path.join(out_dir, f'top_{k}_{score_key}.txt'), 'w') as f:
        for dp in sorted_data[-k:]:
            f.write(f'{score_key}: {dp[score_key]:.4f}\n')
            f.write(f'Question: {dp["question"]}\n')
            f.write(f'GT Presuppositions: {"; ".join(dp["presuppositions"] + dp["raw_presuppositions"])}\n')
            f.write(f'Model Answer: {dp["model_answer"]}\n')
            f.write('-' * 20 + '\n\n')
    with open(os.path.join(out_dir, f'bottom_{k}_{score_key}.txt'), 'w') as f:
        for dp in sorted_data[:k]:
            f.write(f'{score_key}: {dp[score_key]:.4f}\n')
            f.write(f'Question: {dp["question"]}\n')
            f.write(f'GT Presuppositions: {"; ".join(dp["presuppositions"] + dp["raw_presuppositions"])}\n')
            f.write(f'Model Answer: {dp["model_answer"]}\n')
            f.write('-' * 20 + '\n\n')

def run_openai_model_check(args, dataset: list):
    result_file_name = f"tmp/openai_results_{args.dataset}.jsonl"
    if not os.path.exists(result_file_name):
        with open(args.batch_job_info_file, 'r') as f:
            id = json.load(f)['id']
        client = openai.Client()
        completed = False
        while not completed:
            time.sleep(10)
            batch_job = client.batches.retrieve(id)
            print(f'Batch job status: {batch_job.status}')
            if batch_job.status in ['failed', 'canceled', 'expired']:
                raise RuntimeError(f'Batch job failed with status: {batch_job.status}')
            completed = batch_job.status == 'completed'
        time.sleep(5)
        result = client.files.content(batch_job.output_file_id).content
        with open(result_file_name, 'wb') as f:
            f.write(result)
    with open(result_file_name, 'r') as f:
        results = [json.loads(line.strip()) for line in f]
    for res in tqdm(results, desc='Organizing responses'):
        i = int(res['custom_id'])
        dataset[i]['model_answer'] = res['response']['body']['choices'][0]['message']['content']
    with open(args.out_file, 'w') as f:
        for data in dataset:
            f.write(json.dumps(data) + '\n')
    os.removedirs('tmp')

def run_transformers_model(args, dataset: list, data_loader: DataLoader):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=args.dtype,
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    count = 0
    for data in tqdm(dataset, desc='Processing dataset'):
        messages = data_loader.get_question(data, template=getattr(template, args.template), system_role=args.system_role)
        inputs = tokenizer.apply_chat_template(messages, return_tensors='pt').to(args.device)
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=512)
        outputs = outputs.cpu()[:, inputs.shape[1]:]
        generation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        data['model_answer'] = generation
        with open(args.out_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
        count += 1
        print(f'Progress: {count}/{len(dataset) + args.start_idx}')

def _run_openai_model_batched(args, client: openai.Client, dataset: list, data_loader: DataLoader):
    all_messages = []
    for data in tqdm(dataset, desc='Processing dataset'):
        os.makedirs('tmp', exist_ok=True)
        messages = data_loader.get_question(data, template=getattr(template, args.template), system_role=args.system_role)
        all_messages.append(messages)
    with open('tmp/temp_messages.jsonl', 'w') as f:
        for i, messages in enumerate(all_messages):
            task = {
                "custom_id": f"{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": messages
                }
            }
            f.write(json.dumps(task) + '\n')
    batch_file = client.files.create(
        file=open('tmp/temp_messages.jsonl', "rb"),
        purpose="batch"
    )
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    with open('tmp/batch_job_info.json', 'w') as f:
        json.dump({"id": batch_job.id}, f, indent=4)

def _run_openai_model_one_by_one(args, client: openai.Client, dataset: list, data_loader: DataLoader):
    count = 0
    for data in tqdm(dataset, desc='Processing dataset'):
        messages = data_loader.get_question(data, template=getattr(template, args.template), system_role=args.system_role)
        response = client.chat.completions.create(
            model=args.model,
            messages=messages
        )
        generation = response.choices[0].message.content
        data['model_answer'] = generation
        with open(args.out_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
        count += 1
        print(f'Progress: {count}/{len(dataset) + args.start_idx}')

def run_openai_model(args, dataset: list, data_loader: DataLoader):
    client = openai.Client()
    if args.batched_job:
        _run_openai_model_batched(args, client, dataset, data_loader)
    else:
        _run_openai_model_one_by_one(args, client, dataset, data_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='For the selected model, generate responses using FP extraction prompt templates, store to output file.')
    model_subparsers = parser.add_subparsers(title='commands', dest='command')
    
    transformers_parser = model_subparsers.add_parser('transformers', help='Arguments for transformers models')
    transformers_parser.add_argument('--model', type=str, required=True, help='Model name or path for loading from transformers')
    transformers_parser.add_argument('--system_role', type=str, default='system', help='Name of the instruction-giving role')
    transformers_parser.add_argument('--dataset_dir', type=str, default='dataset', help='Path to the dataset directory (JSONL format)')
    transformers_parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use (e.g., movies, CREPE)')
    transformers_parser.add_argument('--start_idx', type=int, default=0, help='Starting index for cached runs')
    transformers_parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, dev, test)')
    transformers_parser.add_argument('--template', type=str, required=True, help='Template class to use for generating prompts')
    transformers_parser.add_argument('--k', type=int, default=4, help='Number of few-shot examples to use for presupposition extraction')
    transformers_parser.add_argument('--device', type=str, default='cpu' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')
    transformers_parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type for model parameters')
    transformers_parser.add_argument('--out_file', type=str, default='out/curated_dataset_{}.jsonl', help='Output file to save the curated dataset')

    openai_parser = model_subparsers.add_parser('openai', help='Arguments for OpenAI models')
    openai_parser.add_argument('--model', type=str, required=True, help='OpenAI model name (e.g., gpt-5)')
    openai_parser.add_argument('--system_role', type=str, default='developer', help='Name of the instruction-giving role')
    openai_parser.add_argument('--batched_job', action='store_true', help='Whether to use batched job submission')
    openai_parser.add_argument('--dataset_dir', type=str, default='dataset', help='Path to the dataset directory (JSONL format)')
    openai_parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use (e.g., movies, CREPE)')
    openai_parser.add_argument('--start_idx', type=int, default=0, help='Starting index for cached runs')
    openai_parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, dev, test)')
    openai_parser.add_argument('--template', type=str, required=True, help='Template class to use for generating prompts')
    openai_parser.add_argument('--k', type=int, default=4, help='Number of few-shot examples to use for presupposition extraction')
    openai_parser.add_argument('--out_file', type=str, default='out/curated_dataset_{}.jsonl', help='Output file to save the curated dataset')
    
    openai_check_parser = model_subparsers.add_parser('openai_check', help='Check status of OpenAI batched job')
    openai_check_parser.add_argument('--model', type=str, required=True, help='OpenAI model name (e.g., gpt-5)')
    openai_check_parser.add_argument('--batch_job_info_file', type=str, default='tmp/batch_job_info.json', help='File containing batch job info')
    openai_check_parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use (e.g., movies, CREPE)')
    openai_check_parser.add_argument('--out_file', type=str, default='out/curated_dataset_{}.jsonl', help='Output file to save the curated dataset')
    
    evaluate_parser = model_subparsers.add_parser('evaluate', help='Evaluate model outputs using specified evaluator')
    evaluate_parser.add_argument('--file', type=str, required=True, help='File containing model outputs to evaluate')
    evaluate_parser.add_argument('--evaluator', type=str, required=True, help='Evaluator class to use for evaluation (e.g., CREPEEvaluator)')
    evaluate_parser.add_argument('--run_bleurt', action='store_true', help='Whether to run BLEURT evaluation (may be slow)')
    evaluate_parser.add_argument('--show_top_bottom_k', type=int, default=0, help='Show top and bottom k examples based on evaluated scores')
    
    args = parser.parse_args()

    main(args)