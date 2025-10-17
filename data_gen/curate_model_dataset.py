import os, json, argparse
import interpretability
import torch
from .data_loader import DataLoader
from tqdm import tqdm

def main(args):
    operator: interpretability.operators.Operator = args.operator(path=args.model_name_or_path, device=args.device, dtype=args.dtype)
    data_loader = DataLoader(dataset_name='movies', file_path=args.dataset_path)
    dataset = data_loader.load_data()
    dataset_name = os.path.dirname(args.dataset_path).split('/')[-1]
    template = data_loader.get_templates(dataset_name, 'KNOWLEDGE_TEST_TEMPLATES')[0]
    n_correct = 0
    for data in tqdm(dataset, desc='Processing dataset'):
        question = template(**data)
        generation = operator.generate(question)
        correct = data[data_loader.get_correct_key()]
        data['model_knows'] = correct in generation
        if data['model_knows']:
            n_correct += 1
    print(f"Model knows {n_correct} out of {len(dataset)} facts. Accuracy: {n_correct / len(dataset):.4f}")
    with open(args.dataset_path, 'w') as f:
        for data in dataset:
            f.write(json.dumps(data) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='For the selected model, create a dataset with labels indicating whether the model knows each fact.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file (JSONL format)')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Model name or path for loading from transformers')
    parser.add_argument('--operator', type=str, required=True, help='Operator class to run model')
    parser.add_argument('--device', type=str, default='cpu' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type for model parameters')
    args = parser.parse_args()
    args.operator = getattr(interpretability.operators, args.operator)
    args.dtype = getattr(torch, args.dtype)
    args.device = torch.device(args.device)

    main(args)