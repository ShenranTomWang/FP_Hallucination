from transformers import AutoTokenizer, AutoModelForCausalLM
import os, json, argparse
import torch
from data_gen.data_loader import DataLoader, instantiate_dataloader
from tqdm import tqdm

def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=args.dtype,
        device_map='auto' if args.device.type == 'cuda' else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    data_loader = instantiate_dataloader(dataset_name=args.dataset, file_dir=args.dataset_dir)
    dataset = data_loader.load_data()
    if hasattr(data_loader, 'get_templates'):
        template = data_loader.get_templates('KNOWLEDGE_TEST_TEMPLATES')[0]
    sys_message = {"role": "system", "content": "You are a knowledgeable assistant that answers questions based on your knowledge."}
    
    for data in tqdm(dataset, desc='Processing dataset'):
        question = data_loader.get_question(data, template=template)
        messages = [sys_message, {"role": "user", "content": question}]
        inputs = tokenizer.apply_chat_template(messages, return_tensors='pt')
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512)
        generation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        data['model_answer'] = generation
    
    with open(args.out_file, 'w') as f:
        for data in dataset:
            f.write(json.dumps(data) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='For the selected model, create a dataset with labels indicating whether the model knows each fact.')
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Path to the dataset directory (JSONL format)')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use (e.g., movies, CREPE)')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Model name or path for loading from transformers')
    parser.add_argument('--out_file', type=str, default='out/curated_dataset_{}.jsonl', help='Output file to save the curated dataset')
    parser.add_argument('--device', type=str, default='cpu' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type for model parameters')
    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)
    args.device = torch.device(args.device)
    args.out_file = args.out_file.format(args.model_name_or_path.split('/')[-1])

    main(args)