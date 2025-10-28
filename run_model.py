from transformers import AutoTokenizer, AutoModelForCausalLM
import json, argparse, os
import torch
from data_gen.data_loader import instantiate_dataloader
from tqdm import tqdm
from data_gen.template import PresuppositionExtractionTemplate

def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=args.dtype,
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    data_loader = instantiate_dataloader(dataset_name=args.dataset, file_dir=args.dataset_dir)
    dataset = data_loader.load_data(split=args.split)
    dataset = dataset[args.start_idx:]
    count = 0
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    
    for data in tqdm(dataset, desc='Processing dataset'):
        template = PresuppositionExtractionTemplate(**data)
        messages = data_loader.get_question(data, template=template)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='For the selected model, generate responses using FP extraction prompt templates, store to output file.')
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Path to the dataset directory (JSONL format)')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting index for cached runs')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use (e.g., movies, CREPE)')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (e.g., train, dev, test)')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Model name or path for loading from transformers')
    parser.add_argument('--out_file', type=str, default='out/curated_dataset_{}.jsonl', help='Output file to save the curated dataset')
    parser.add_argument('--device', type=str, default='cpu' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type for model parameters')
    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)
    args.device = torch.device(args.device)
    args.out_file = args.out_file.format(args.model_name_or_path.split('/')[-1])

    main(args)