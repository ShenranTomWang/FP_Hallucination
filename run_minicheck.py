import argparse, json, os
import minicheck_operator
from tqdm import tqdm

def main(args):
    operator = getattr(minicheck_operator, args.operator)(model_name=args.model_name, cache_dir=args.cache_dir)
    if args.out_file is None:
        fnames = args.file.split('.')
        args.out_file = f"{'.'.join(fnames[:-1])}_minichecked.{fnames[-1]}"
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.file, 'r') as f:
        data = [json.loads(line) for line in f]
    for i, dp in enumerate(tqdm(data, desc="Running MiniCheck")):
        dp = operator.check(dp)
        data[i] = dp
    with open(args.out_file, 'w') as f:
        for dp in data:
            f.write(json.dumps(dp) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the file.jsonl to check.")
    parser.add_argument('--out_file', type=str, default=None, help='Output file to save the minichecked results, defaults to {--file}_minichecked.jsonl')
    parser.add_argument("--operator", type=str, required=True, help="The operator to use.")
    parser.add_argument("--model_name", type=str, default="flan-t5-large", help="The MiniCheck model name.")
    parser.add_argument("--cache_dir", type=str, default=os.getenv("HF_HOME"), help="The cache directory for the MiniCheck model.")
    args = parser.parse_args()
    main(args)