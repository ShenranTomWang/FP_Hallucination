from transformers import AutoTokenizer, AutoModelForCausalLM
import json, argparse, os, random, time
import torch
import openai
from data_gen.data_loader import instantiate_dataloader, DataLoader
from tqdm import tqdm
from data_gen.template import PresuppositionExtractionTemplate

def main(args):
    data_loader = instantiate_dataloader(dataset_name=args.dataset, file_dir=args.dataset_dir)
    dataset = data_loader.load_data(split=args.split)
    dataset = dataset[args.start_idx:]
    if dataset[0].get('few_shot_data') is None:
        few_shot_data = data_loader.load_data(split='train')
        few_shot_data = [data for data in few_shot_data if len(data['presuppositions']) != 0]
        few_shot_data = random.sample(few_shot_data, args.k)
        for data in dataset:
            data['few_shot_data'] = few_shot_data
        data_loader.save_data(dataset, split=args.split)
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    
    if args.model_subcommand == 'transformers':
        run_transformers_model(args, dataset, data_loader)
    elif args.model_subcommand == 'openai':
        run_openai_model(args, dataset, data_loader)
    elif args.model_subcommand == 'openai_check':
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
        result_file_name = "tmp/openai_results_{}.jsonl".format(args.dataset)
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

def run_transformers_model(args, dataset: list, data_loader: DataLoader):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=args.dtype,
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    count = 0
    for data in tqdm(dataset, desc='Processing dataset'):
        messages = data_loader.get_question(data, template=PresuppositionExtractionTemplate, system_role=args.system_role)
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

def run_openai_model_batched(args, client: openai.Client, dataset: list, data_loader: DataLoader):
    all_messages = []
    for data in tqdm(dataset, desc='Processing dataset'):
        os.makedirs('tmp', exist_ok=True)
        messages = data_loader.get_question(data, template=PresuppositionExtractionTemplate, system_role=args.system_role)
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

def run_openai_model_one_by_one(args, client: openai.Client, dataset: list, data_loader: DataLoader):
    count = 0
    for data in tqdm(dataset, desc='Processing dataset'):
        messages = data_loader.get_question(data, template=PresuppositionExtractionTemplate, system_role=args.system_role)
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
        run_openai_model_batched(args, client, dataset, data_loader)
    else:
        run_openai_model_one_by_one(args, client, dataset, data_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='For the selected model, generate responses using FP extraction prompt templates, store to output file.')
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Path to the dataset directory (JSONL format)')
    parser.add_argument('--k', type=int, default=4, help='Number of few-shot examples to use for presupposition extraction')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting index for cached runs')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use (e.g., movies, CREPE)')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, dev, test)')
    parser.add_argument('--out_file', type=str, default='out/curated_dataset_{}.jsonl', help='Output file to save the curated dataset')
    parser.add_argument('--device', type=str, default='cpu' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type for model parameters')
    model_subparsers = parser.add_subparsers(title='model_subcommands', dest='model_subcommand')
    
    transformers_parser = model_subparsers.add_parser('transformers', help='Arguments for transformers models')
    transformers_parser.add_argument('--model', type=str, required=True, help='Model name or path for loading from transformers')
    transformers_parser.add_argument('--system_role', type=str, default='system', help='Name of the instruction-giving role')
    
    openai_parser = model_subparsers.add_parser('openai', help='Arguments for OpenAI models')
    openai_parser.add_argument('--model', type=str, required=True, help='OpenAI model name (e.g., gpt-5)')
    openai_parser.add_argument('--system_role', type=str, default='developer', help='Name of the instruction-giving role')
    openai_parser.add_argument('--batched_job', action='store_true', help='Whether to use batched job submission')
    
    openai_check_parser = model_subparsers.add_parser('openai_check', help='Check status of OpenAI batched job')
    openai_check_parser.add_argument('--batch_job_info_file', type=str, default='tmp/batch_job_info.json', help='File containing batch job info')
    
    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)
    args.device = torch.device(args.device)
    args.out_file = args.out_file.format(args.model.split('/')[-1])

    main(args)