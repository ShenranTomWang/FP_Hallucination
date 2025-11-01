from data_gen.template import CREPEPresuppositionExtractionTemplate
from evaluator import CREPEEvaluator
from .data_operator import DataOperator
from data_gen.data_loader import instantiate_dataloader

class CREPEPresuppositionExtractionOperator(DataOperator):
    def __init__(self):
        self.action_name = "CREPE_Presupposition_Extraction"
        self.dataloader = None

    def prepare_message(self, raw_dp: dict, system_role: str, **kwargs) -> str:
        template = CREPEPresuppositionExtractionTemplate(**raw_dp, system_role=system_role)
        return template.generate()
    
    def parse_response_openai(self, response: dict, save_dp: dict, **kwargs) -> dict:
        save_dp['model_detected_presuppositions'] = response['response']['body']['choices'][0]['message']['content']
        return save_dp
    
    def evaluate(self, eval_dp: dict, run_bleurt: bool) -> tuple:
        evaluator = CREPEEvaluator()
        rouge1_f1 = evaluator.evaluate_rouge1_f1()
        rougeL_f1 = evaluator.evaluate_rougeL_f1()
        if run_bleurt:
            bleurt_f1 = evaluator.evaluate_bleurt_f1()
            return rouge1_f1, rougeL_f1, bleurt_f1
        return rouge1_f1, rougeL_f1, None

    def add_data_module(self, file_dir: str = 'dataset'):
        self.dataloader = instantiate_dataloader(dataset_name="CREPE", file_dir=file_dir)
    
    def save_top_bottom_k(self, data: list, score_key: str, k: int, out_dir: str):
        sorted_data = sorted(
            [dp for dp in data if dp.get("model_detected_presuppositions") is not None and dp.get(score_key) is not None],
            key=lambda x: x[score_key]
        )
        with open(os.path.join(out_dir, f'top_{k}_{score_key}_{self.action_name}.txt'), 'w') as f:
            for dp in sorted_data[-k:]:
                f.write(f'{score_key}: {dp[score_key]:.4f}\n')
                f.write(f'Question: {dp["question"]}\n')
                f.write(f'GT Presuppositions: {"; ".join(dp["presuppositions"] + dp["raw_presuppositions"])}\n')
                f.write(f'Model Answer: {dp["model_detected_presuppositions"]}\n')
                f.write('-' * 20 + '\n\n')
        with open(os.path.join(out_dir, f'bottom_{k}_{score_key}_{self.action_name}.txt'), 'w') as f:
            for dp in sorted_data[:k]:
                f.write(f'{score_key}: {dp[score_key]:.4f}\n')
                f.write(f'Question: {dp["question"]}\n')
                f.write(f'GT Presuppositions: {"; ".join(dp["presuppositions"] + dp["raw_presuppositions"])}\n')
                f.write(f'Model Answer: {dp["model_detected_presuppositions"]}\n')
                f.write('-' * 20 + '\n\n')