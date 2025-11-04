from transformers import AutoTokenizer, AutoModelForCausalLM
import json, argparse, os, random, time
import torch
import openai
from data_gen.data_loader import instantiate_dataloader, DataLoader
from tqdm import tqdm
import data_operator
import evaluator
import numpy as np

def main(args):
    if args.command == 'transformers':
        args.dtype = getattr(torch, args.dtype)
        args.device = torch.device(args.device)
    if args.command in ['transformers', 'openai']:
        args.out_file = args.out_file.format(args.model.split('/')[-1])
    if hasattr(args, 'operator'):
        operator_class: data_operator.DataOperator = getattr(data_operator, args.operator)
        operator = operator_class()

    if args.command == 'evaluate':
        run_evaluate(args, operator)
        return

    if args.dataset_dir is None:
        operator.add_data_module(model_name=args.model)
    else:
        operator.add_data_module(file_dir=args.dataset_dir, model_name=args.model)
    dataset_full = operator.load_data(split=args.split, k = args.k if hasattr(args, 'k') else 0)
    dataset = dataset_full[args.start_idx:]
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    
    if args.command == 'transformers':
        run_transformers_model(args, dataset, operator)
    elif args.command == 'openai':
        run_openai_model(args, dataset, operator)
    elif args.command == 'openai_check':
        run_openai_model_check(args, dataset, operator)

def run_evaluate(args, operator: data_operator.DataOperator):
    with open(args.file, 'r') as f:
        data = [json.loads(line.strip()) for line in f]
    evaluator_class: evaluator.Evaluator = getattr(evaluator, args.evaluator)
    evaluators = [evaluator_class(**dp) for dp in data if dp.get('model_detected_presuppositions') is not None]
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
            operator.save_top_bottom_k(data, score_key, k, os.path.dirname(args.file))

def run_openai_model_check(args, dataset: list, operator: data_operator.DataOperator):
    result_file_name = f"tmp/openai_results_{operator.dataloader.dataset_name}_{operator.action_name}.jsonl"
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
        dataset[i] = operator.parse_response_openai(res, dataset[i])
    args.out_file = args.out_file.format(f'{args.model.split('/')[-1]}_{operator.action_name}')
    with open(args.out_file, 'w') as f:
        for data in dataset:
            f.write(json.dumps(data) + '\n')

def run_transformers_model(args, dataset: list, operator: data_operator.DataOperator):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=args.dtype,
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    count = 0
    for data in tqdm(dataset, desc='Processing dataset'):
        messages = operator.prepare_message(data, system_role=args.system_role)
        inputs = tokenizer.apply_chat_template(messages, return_tensors='pt').to(args.device)
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=512)
        outputs = outputs.cpu()[:, inputs.shape[1]:]
        generation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        data['model_answer'] = generation       # TODO: this needs to be refactored
        with open(args.out_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
        count += 1
        print(f'Progress: {count}/{len(dataset) + args.start_idx}')

def _run_openai_model_batched(args, client: openai.Client, dataset: list, operator: data_operator.DataOperator):
    all_messages = []
    for data in tqdm(dataset, desc='Processing dataset'):
        os.makedirs('tmp', exist_ok=True)
        messages = operator.prepare_message(data, system_role=args.system_role)
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

def _run_openai_model_one_by_one(args, client: openai.Client, dataset: list, operator: data_operator.DataOperator):
    count = 0
    for data in tqdm(dataset, desc='Processing dataset'):
        messages = operator.prepare_message(data, system_role=args.system_role)
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

def run_openai_model(args, dataset: list, operator: data_operator.DataOperator):
    client = openai.Client()
    if args.batched_job:
        _run_openai_model_batched(args, client, dataset, operator)
    else:
        _run_openai_model_one_by_one(args, client, dataset, operator)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='For the selected model, generate responses using FP extraction prompt templates, store to output file.')
    model_subparsers = parser.add_subparsers(title='commands', dest='command')
    
    transformers_parser = model_subparsers.add_parser('transformers', help='Arguments for transformers models')
    transformers_parser.add_argument('--model', type=str, required=True, help='Model name or path for loading from transformers')
    transformers_parser.add_argument('--system_role', type=str, default='system', help='Name of the instruction-giving role')
    transformers_parser.add_argument('--dataset_dir', type=str, default=None, help='Path to the dataset directory (JSONL format)')
    transformers_parser.add_argument('--start_idx', type=int, default=0, help='Starting index for cached runs')
    transformers_parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, dev, test)')
    transformers_parser.add_argument('--operator', type=str, required=True, help='Operator class to use for generating prompts, extract responses and evaluate')
    transformers_parser.add_argument('--k', type=int, default=4, help='Number of few-shot examples to use for presupposition extraction')
    transformers_parser.add_argument('--device', type=str, default='cpu' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')
    transformers_parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type for model parameters')
    transformers_parser.add_argument('--out_file', type=str, default='out/curated_dataset_{}.jsonl', help='Output file to save the curated dataset')

    openai_parser = model_subparsers.add_parser('openai', help='Arguments for OpenAI models')
    openai_parser.add_argument('--model', type=str, required=True, help='OpenAI model name (e.g., gpt-5)')
    openai_parser.add_argument('--system_role', type=str, default='developer', help='Name of the instruction-giving role')
    openai_parser.add_argument('--batched_job', action='store_true', help='Whether to use batched job submission')
    openai_parser.add_argument('--dataset_dir', type=str, default=None, help='Path to the dataset directory (JSONL format)')
    openai_parser.add_argument('--start_idx', type=int, default=0, help='Starting index for cached runs')
    openai_parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, dev, test)')
    openai_parser.add_argument('--operator', type=str, required=True, help='Operator class to use for generating prompts, extract responses and evaluate')
    openai_parser.add_argument('--k', type=int, default=4, help='Number of few-shot examples to use for presupposition extraction')
    openai_parser.add_argument('--out_file', type=str, default='out/curated_dataset_{}.jsonl', help='Output file to save the curated dataset')
    
    openai_check_parser = model_subparsers.add_parser('openai_check', help='Check status of OpenAI batched job')
    openai_check_parser.add_argument('--model', type=str, required=True, help='OpenAI model name (e.g., gpt-5)')
    openai_check_parser.add_argument('--batch_job_info_file', type=str, default='tmp/batch_job_info.json', help='File containing batch job info')
    openai_check_parser.add_argument('--out_file', type=str, default='out/curated_dataset_{}.jsonl', help='Output file to save the curated dataset')
    openai_check_parser.add_argument('--operator', type=str, required=True, help='Operator class to use for generating prompts, extract responses and evaluate')
    openai_check_parser.add_argument('--dataset_dir', type=str, default=None, help='Path to the dataset directory (JSONL format)')
    openai_check_parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, dev, test)')
    openai_check_parser.add_argument('--start_idx', type=int, default=0, help='Starting index for cached runs')
    
    evaluate_parser = model_subparsers.add_parser('evaluate', help='Evaluate model outputs using specified evaluator')
    evaluate_parser.add_argument('--file', type=str, required=True, help='File containing model outputs to evaluate')
    evaluate_parser.add_argument('--evaluator', type=str, required=True, help='Evaluator class to use for evaluation (e.g., CREPEEvaluator)')
    evaluate_parser.add_argument('--run_bleurt', action='store_true', help='Whether to run BLEURT evaluation (may be slow)')
    evaluate_parser.add_argument('--show_top_bottom_k', type=int, default=0, help='Show top and bottom k examples based on evaluated scores')
    
    args = parser.parse_args()

    main(args)