from transformers import AutoTokenizer, AutoModelForCausalLM
import json, argparse, os, time
import torch
import openai
from tqdm import tqdm
import data_operator
import numpy as np
from google import genai
from google.genai import types

def main(args):
    if args.command == 'transformers':
        args.dtype = getattr(torch, args.dtype)
        args.device = torch.device(args.device)
    if hasattr(args, 'operator'):
        operator_class: data_operator.DataOperator = getattr(data_operator, args.operator)
        operator = operator_class()
    if args.command in ['transformers', 'openai', 'openai_check', 'gemini', 'gemini_check']:
        args.out_file = args.out_file.format(f'{args.model.split("/")[-1]}_{operator.action_name}')
        args.out_file = os.path.join(args.out_dir, args.out_file)

    if args.command == 'evaluate':
        run_evaluate(args, operator)
        return
    elif args.command == 'print_examples':
        run_print_examples(args)
        return
    elif args.command == 'align_responses':
        run_align_responses(args, operator)
        return

    dataset_full = operator.load_data(file_path=args.dataset_path, k=args.k if hasattr(args, 'k') else None)
    dataset = dataset_full[args.start_idx:]
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    
    if args.command == 'transformers':
        run_transformers_model(args, dataset, operator)
    elif args.command == 'openai':
        run_openai_model(args, dataset, operator)
    elif args.command == 'openai_check':
        run_openai_model_check(args, dataset, operator)
    elif args.command == 'gemini':
        run_gemini_model(args, dataset, operator)
    elif args.command == 'gemini_check':
        run_gemini_model_check(args, dataset, operator)
        
def run_print_examples(args):
    with open(args.dataset_path, 'r') as f:
        dataset = [json.loads(line.strip()) for line in f]
    with open(os.path.join(args.out_dir, 'printed_examples.txt'), 'w') as f:
        for dp in dataset[:args.k]:
            f.write(f'ID: {dp["id"]}\n')
            f.write(f'Question: {dp["question"]}\n')
            f.write(f'Comment: {dp["comment"]}\n')
            f.write(f'GT Presuppositions: {"; ".join(dp["presuppositions"])}\n')
            f.write('-' * 40 + '\n')
            
def _avg_report(args, data: list, measure: str):
    rouge1_f1_key = f'rouge1_f1_{measure}'
    rougeL_f1_key = f'rougeL_f1_{measure}'
    avg_rouge1 = np.mean([dp[rouge1_f1_key] for dp in data if dp.get(rouge1_f1_key) is not None])
    avg_rougeL = np.mean([dp[rougeL_f1_key] for dp in data if dp.get(rougeL_f1_key) is not None])
    print(f'Average ROUGE-1 F1 {measure.capitalize()}: {avg_rouge1:.4f}')
    print(f'Average ROUGE-L F1 {measure.capitalize()}: {avg_rougeL:.4f}')
    if args.run_bleurt:
        bleurt_key = f'bleurt_f1_{measure}'
        avg_bleurt = np.mean([dp[bleurt_key] for dp in data if dp.get(bleurt_key) is not None])
        print(f'Average BLEURT F1 {measure.capitalize()}: {avg_bleurt:.4f}')
    if args.run_bert_score:
        bert_score_key = f'bert_score_f1_{measure}'
        avg_bert_score = np.mean([dp[bert_score_key] for dp in data if dp.get(bert_score_key) is not None])
        print(f'Average BERTScore F1 {measure.capitalize()}: {avg_bert_score:.4f}')

def run_evaluate(args, operator: data_operator.DataOperator):
    with open(args.file, 'r') as f:
        data = [json.loads(line.strip()) for line in f]
    for dp in tqdm(data, desc='Evaluating'):
        if dp.get(operator.answer_key) is None:
            dp['rouge1_f1_precision'], dp['rouge1_f1_recall'] = 0, 0
            dp['rougeL_f1_precision'], dp['rougeL_f1_recall'] = 0, 0
            if args.run_bleurt:
                dp['bleurt_f1_precision'], dp['bleurt_f1_recall'] = 0, 0
            if args.run_bert_score:
                dp['bert_score_f1_precision'], dp['bert_score_f1_recall'] = 0, 0
            continue
        dp['rouge1_f1_precision'], dp['rougeL_f1_precision'], bleurt_f1_precision, bert_score_f1_precision = operator.evaluate(dp, run_bleurt=args.run_bleurt, run_bert_score=args.run_bert_score, use_aligned="precision")
        dp['rouge1_f1_recall'], dp['rougeL_f1_recall'], bleurt_f1_recall, bert_score_f1_recall = operator.evaluate(dp, run_bleurt=args.run_bleurt, run_bert_score=args.run_bert_score, use_aligned="recall")
        if args.run_bleurt:
            dp['bleurt_f1_precision'] = bleurt_f1_precision
            dp['bleurt_f1_recall'] = bleurt_f1_recall
        if args.run_bert_score:
            dp['bert_score_f1_precision'] = bert_score_f1_precision
            dp['bert_score_f1_recall'] = bert_score_f1_recall
    with open(args.file, 'w') as f:
        for dp in data:
            f.write(json.dumps(dp) + '\n')

    _avg_report(args, data, 'precision')
    _avg_report(args, data, 'recall')
    
    if args.show_top_bottom_k > 0:
        k = args.show_top_bottom_k
        for score_key in ['rouge1_f1_precision', 'rougeL_f1_precision'] + (['bleurt_f1_precision'] if args.run_bleurt else []) + (['bert_score_f1_precision'] if args.run_bert_score else []):
            operator.save_top_bottom_k(data, score_key, k, os.path.dirname(args.file), use_aligned='precision')
        for score_key in ['rouge1_f1_recall', 'rougeL_f1_recall'] + (['bleurt_f1_recall'] if args.run_bleurt else []) + (['bert_score_f1_recall'] if args.run_bert_score else []):
            operator.save_top_bottom_k(data, score_key, k, os.path.dirname(args.file), use_aligned='recall')

def run_align_responses(args, operator: data_operator.DataOperator):
    with open(args.file, 'r') as f:
        data = [json.loads(line.strip()) for line in f]
    for dp in tqdm(data, desc='Aligning responses'):
        dp = operator.align_response(dp, model_type=args.model_type)
    with open(args.file, 'w') as f:
        for dp in data:
            f.write(json.dumps(dp) + '\n')

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
    with open(args.out_file, 'w') as f:
        for data in dataset:
            f.write(json.dumps(data) + '\n')

def run_gemini_model_check(args, dataset: list, operator: data_operator.DataOperator):
    result_file_name = f"tmp/gemini_results_{operator.dataloader.dataset_name}_{operator.action_name}.jsonl"
    if not os.path.exists(result_file_name):
        with open(args.batch_job_info_file, 'r') as f:
            batch_job = json.load(f)
        genai_client = genai.Client(vertexai=True)
        completed = False
        while not completed:
            time.sleep(10)
            batch_job = genai_client.batches.get(name=batch_job['name'])
            print(f'Batch job state: {batch_job.state}')
            if batch_job.state in [types.JobState.JOB_STATE_FAILED, types.JobState.JOB_STATE_CANCELED, types.JobState.JOB_STATE_EXPIRED]:
                raise RuntimeError(f'Batch job failed with state: {batch_job.state}')
            completed = batch_job.state == types.JobState.JOB_STATE_SUCCEEDED
        time.sleep(5)
        if batch_job.dest and batch_job.dest.file_name:
            file_content = genai_client.files.download(batch_job.dest.file_name)
            with open(result_file_name, 'wb') as f:
                f.write(file_content)
    with open(result_file_name, 'r') as f:
        results = [json.loads(line.strip()) for line in f]
    for res in tqdm(results, desc='Organizing responses'):
        i = int(res['key'])
        dataset[i] = operator.parse_response_gemini(res, dataset[i])
    with open(args.out_file, 'w') as f:
        for data in dataset:
            f.write(json.dumps(data) + '\n')

def run_transformers_model(args, dataset: list, operator: data_operator.DataOperator):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=args.dtype,
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    count = 0
    for data in tqdm(dataset, desc='Processing dataset'):
        messages = operator.prepare_message(data, system_role=args.system_role)
        response = operator.run_transformer_model(model, tokenizer, messages, device=args.device)
        data = operator.parse_response_transformers(response, data)
        with open(args.out_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
        count += 1
        print(f'Progress: {count}/{len(dataset) + args.start_idx}')

def _run_openai_model_batched(args, client: openai.Client, dataset: list, operator: data_operator.DataOperator):
    all_messages = []
    for data in tqdm(dataset, desc='Processing dataset'):
        os.makedirs('tmp', exist_ok=True)
        messages = operator.prepare_message(data, system_role='developer')
        all_messages.append(messages)
    with open('tmp/temp_messages.jsonl', 'w') as f:
        for i, messages in enumerate(all_messages):
            task = operator.message2openai_request(f"{i}", args.model, messages, use_web_search=args.use_web_search)
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

def _run_gemini_model_batched(args, genai_client: genai.Client, dataset: list, operator: data_operator.DataOperator):
    all_messages = []
    for data in tqdm(dataset, desc='Processing dataset'):
        os.makedirs('tmp', exist_ok=True)
        messages = operator.prepare_message(data, system_role='system', model_role='model', user_role='user')
        all_messages.append(messages)
    with open('tmp/temp_messages.jsonl', 'w') as f:
        for i, messages in enumerate(all_messages):
            task = operator.message2gemini_request(f"{i}", messages, use_web_search=args.use_web_search)
            f.write(json.dumps(task) + '\n')
    uploaded_file = genai_client.files.upload(
        file='tmp/temp_messages.jsonl',
        config=types.UploadFileConfig(display_name='gemini_batch_input', mime_type='jsonl')
    )
    batch_job = genai_client.batches.create(
        model=args.model,
        src=uploaded_file.name,
        config={
            'display_name': f'gemini_batch_job_{operator.dataloader.dataset_name}_{operator.action_name}',
        }
    )
    with open('tmp/batch_job_info.json', 'w') as f:
        json.dump(batch_job, f, indent=4)

def _run_openai_model_one_by_one(args, client: openai.Client, dataset: list, operator: data_operator.DataOperator):
    count = 0
    for data in tqdm(dataset, desc='Processing dataset'):
        messages = operator.prepare_message(data, system_role='developer')
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            tools=[
                {
                    "type": "web_search",
                    "filters": {
                        "allowed_domains": ["wikipedia.org"]
                    }
                }
            ] if args.use_web_search else []
        )
        data = operator.parse_response_openai(response.text, data)
        with open(args.out_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
        count += 1
        print(f'Progress: {count}/{len(dataset) + args.start_idx}')

def _run_gemini_model_one_by_one(args, client: genai.Client, dataset: list, operator: data_operator.DataOperator):
    count = 0
    for data in tqdm(dataset, desc='Processing dataset'):
        messages = operator.prepare_message(data, system_role='system', model_role='model', user_role='user')
        response = client.models.generate_content(
            model=args.model,
            contents=[{"role": message["role"], "parts": [{"text": message["content"]}]} for message in messages[1:]],
            config=types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=messages[0]['content'],
                tools=[
                    types.Tool(
                        google_search=types.GoogleSearch(include_domains=operator.exclude_domains)
                    )
                ] if args.use_web_search else []
            )
        )
        data = operator.parse_response_gemini(response.text, data)
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
        
def run_gemini_model(args, dataset: list, operator: data_operator.DataOperator):
    genai_client = genai.Client(vertexai=True)
    if args.batched_job:
        _run_gemini_model_batched(args, genai_client, dataset, operator)
    else:
        _run_gemini_model_one_by_one(args, genai_client, dataset, operator)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='For the selected model, generate responses using FP extraction prompt templates, store to output file.')
    model_subparsers = parser.add_subparsers(title='commands', dest='command')
    
    transformers_parser = model_subparsers.add_parser('transformers', help='Arguments for transformers models')
    transformers_parser.add_argument('--model', type=str, required=True, help='Model name or path for loading from transformers')
    transformers_parser.add_argument('--system_role', type=str, default='system', help='Name of the instruction-giving role')
    transformers_parser.add_argument('--dataset_path', type=str, default=None, help='Path to the dataset file (JSONL format)')
    transformers_parser.add_argument('--start_idx', type=int, default=0, help='Starting index for cached runs')
    transformers_parser.add_argument('--operator', type=str, required=True, help='Operator class to use for generating prompts, extract responses and evaluate')
    transformers_parser.add_argument('--k', type=int, default=None, help='Number of few-shot examples to use for presupposition extraction')
    transformers_parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')
    transformers_parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type for model parameters')
    transformers_parser.add_argument('--out_dir', type=str, default='out', help='Output directory to save the curated dataset')
    transformers_parser.add_argument('--out_file', type=str, default='curated_dataset_{}.jsonl', help='Output file to save the curated dataset')

    openai_parser = model_subparsers.add_parser('openai', help='Arguments for OpenAI models')
    openai_parser.add_argument('--model', type=str, required=True, help='OpenAI model name (e.g., gpt-5)')
    openai_parser.add_argument('--system_role', type=str, default='developer', help='Name of the instruction-giving role')
    openai_parser.add_argument('--batched_job', action='store_true', help='Whether to use batched job submission')
    openai_parser.add_argument('--dataset_path', type=str, default=None, help='Path to the dataset file (JSONL format)')
    openai_parser.add_argument('--start_idx', type=int, default=0, help='Starting index for cached runs')
    openai_parser.add_argument('--operator', type=str, required=True, help='Operator class to use for generating prompts, extract responses and evaluate')
    openai_parser.add_argument('--k', type=int, default=None, help='Number of few-shot examples to use for presupposition extraction')
    openai_parser.add_argument('--out_dir', type=str, default='out', help='Output directory to save the curated dataset')
    openai_parser.add_argument('--out_file', type=str, default='curated_dataset_{}.jsonl', help='Output file to save the curated dataset')
    openai_parser.add_argument('--use_web_search', action='store_true', help='Whether to use web search tool in Gemini model calls')
    
    gemini_parser = model_subparsers.add_parser('gemini', help='Arguments for Gemini models')
    gemini_parser.add_argument('--model', type=str, required=True, help='Gemini model name (e.g., gemini-2.5-flash)')
    gemini_parser.add_argument('--system_role', type=str, default='system', help='Name of the instruction-giving role')
    gemini_parser.add_argument('--batched_job', action='store_true', help='Whether to use batched job submission')
    gemini_parser.add_argument('--dataset_path', type=str, default=None, help='Path to the dataset file (JSONL format)')
    gemini_parser.add_argument('--start_idx', type=int, default=0, help='Starting index for cached runs')
    gemini_parser.add_argument('--operator', type=str, required=True, help='Operator class to use for generating prompts, extract responses and evaluate')
    gemini_parser.add_argument('--k', type=int, default=None, help='Number of few-shot examples to use for presupposition extraction')
    gemini_parser.add_argument('--out_dir', type=str, default='out', help='Output directory to save the curated dataset')
    gemini_parser.add_argument('--out_file', type=str, default='curated_dataset_{}.jsonl', help='Output file to save the curated dataset')
    gemini_parser.add_argument('--use_web_search', action='store_true', help='Whether to use web search tool in Gemini model calls')
    
    openai_check_parser = model_subparsers.add_parser('openai_check', help='Check status of OpenAI batched job')
    openai_check_parser.add_argument('--model', type=str, required=True, help='OpenAI model name (e.g., gpt-5)')
    openai_check_parser.add_argument('--batch_job_info_file', type=str, default='tmp/batch_job_info.json', help='File containing batch job info')
    openai_check_parser.add_argument('--out_dir', type=str, default='out', help='Output directory to save the curated dataset')
    openai_check_parser.add_argument('--out_file', type=str, default='curated_dataset_{}.jsonl', help='Output file to save the curated dataset')
    openai_check_parser.add_argument('--operator', type=str, required=True, help='Operator class to use for generating prompts, extract responses and evaluate')
    openai_check_parser.add_argument('--dataset_path', type=str, default=None, help='Path to the dataset file (JSONL format)')
    openai_check_parser.add_argument('--start_idx', type=int, default=0, help='Starting index for cached runs')
    
    gemini_check_parser = model_subparsers.add_parser('gemini_check', help='Check status of Gemini batched job')
    gemini_check_parser.add_argument('--model', type=str, required=True, help='Gemini model name (e.g., gpt-5)')
    gemini_check_parser.add_argument('--batch_job_info_file', type=str, default='tmp/batch_job_info.json', help='File containing batch job info')
    gemini_check_parser.add_argument('--out_dir', type=str, default='out', help='Output directory to save the curated dataset')
    gemini_check_parser.add_argument('--out_file', type=str, default='curated_dataset_{}.jsonl', help='Output file to save the curated dataset')
    gemini_check_parser.add_argument('--operator', type=str, required=True, help='Operator class to use for generating prompts, extract responses and evaluate')
    gemini_check_parser.add_argument('--dataset_path', type=str, default=None, help='Path to the dataset file (JSONL format)')
    gemini_check_parser.add_argument('--start_idx', type=int, default=0, help='Starting index for cached runs')
    
    align_responses_parser = model_subparsers.add_parser('align_responses', help='Align model responses with original dataset GT answers')
    align_responses_parser.add_argument('--file', type=str, required=True, help='File containing model outputs to align')
    align_responses_parser.add_argument('--operator', type=str, required=True, help='Operator class to use for alignment')
    align_responses_parser.add_argument('--model_type', type=str, default=None, help='Model type for BERTScore computation during alignment')
    
    evaluate_parser = model_subparsers.add_parser('evaluate', help='Evaluate model outputs using specified operator')
    evaluate_parser.add_argument('--file', type=str, required=True, help='File containing model outputs to evaluate')
    evaluate_parser.add_argument('--operator', type=str, required=True, help='Operator class to use for evaluation')
    evaluate_parser.add_argument('--run_bleurt', action='store_true', help='Whether to run BLEURT evaluation (may be slow)')
    evaluate_parser.add_argument('--run_bert_score', action='store_true', help='Whether to run BERTScore evaluation (may be slow)')
    evaluate_parser.add_argument('--show_top_bottom_k', type=int, default=0, help='Show top and bottom k examples based on evaluated scores')
    
    print_parser = model_subparsers.add_parser('print_examples', help='Print examples from dataset')
    print_parser.add_argument('--dataset', type=str, required=True, help='Dataset name to use for printing examples')
    print_parser.add_argument('--dataset_path', type=str, default=None, help='Path to the dataset file (JSONL format)')
    print_parser.add_argument('--k', type=int, default=5, help='Number of examples to print')
    print_parser.add_argument('--out_dir', type=str, default='out', help='Output directory to save the printed examples')
    
    args = parser.parse_args()

    main(args)