from data_gen.template import CREPEPresuppositionExtractionTemplate, CREPEFeedbackActionTemplate
from evaluator import CREPEEvaluator
from .data_operator import DataOperator
from data_gen.data_loader import instantiate_dataloader
import random, os
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from response import CREPEPresuppositionExtractionResponse, CREPEFeedbackActionResponse, Response
import torch
from typing import List
from pydantic import BaseModel

class CREPEOperator(DataOperator):
    transformer_model: outlines.models.Transformers = None

    def evaluate(self, eval_dp: dict, run_bleurt: bool) -> tuple:
        evaluator = CREPEEvaluator()
        rouge1_f1 = evaluator.evaluate_rouge1_f1()
        rougeL_f1 = evaluator.evaluate_rougeL_f1()
        if run_bleurt:
            bleurt_f1 = evaluator.evaluate_bleurt_f1()
            return rouge1_f1, rougeL_f1, bleurt_f1
        return rouge1_f1, rougeL_f1, None

    def message2openai_request(self, id: str, model: str, messages: List[str], **kwargs) -> dict:
        return {
            "custom_id": id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
                "response_format": self.response_cls.model_json_schema()
            }
        }

    def run_transformer_model(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, messages: List[str], device: torch.DeviceObjType, **kwargs) -> Response:
        if not self.transformer_model:
            self.transformer_model = outlines.from_transformers(
                model, tokenizer
            )
        prompt = tokenizer.apply_chat_template(messages)
        prompt = tokenizer.decode(prompt)
        response = self.transformer_model(prompt, self.response_cls, max_new_tokens=512)
        return self.response_cls.model_validate_json(response)

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

class CREPEPresuppositionExtractionOperator(CREPEOperator):
    def __init__(self):
        self.action_name = "CREPE_Presupposition_Extraction"
        self.dataloader = None
        self.response_cls = CREPEPresuppositionExtractionResponse

    def add_data_module(self, file_dir: str = 'dataset', **kwargs):
        self.dataloader = instantiate_dataloader(dataset_name="CREPE", file_dir=file_dir)

    def load_data(self, split: str, k: int, **kwargs):
        dataset = self.dataloader.load_data(split=split)
        if dataset[0].get('few_shot_data') is None:
            few_shot_data = self.dataloader.load_data(split='train')
            few_shot_data = [data for data in few_shot_data if len(data['presuppositions']) != 0]
            few_shot_data = random.sample(few_shot_data, k)
            for data in dataset:
                data['few_shot_data'] = few_shot_data
            self.dataloader.save_data(dataset, split=split)
        return self.dataloader.load_data(split)

    def prepare_message(self, raw_dp: dict, system_role: str, **kwargs) -> str:
        template = CREPEPresuppositionExtractionTemplate(**raw_dp, system_role=system_role)
        return template.generate()
    
    def parse_response_openai(self, response: dict, save_dp: dict, **kwargs) -> dict:
        save_dp['model_detected_presuppositions'] = response['response']['body']['choices'][0]['message']['content']
        return save_dp
    
    def parse_response_transformers(self, response: BaseModel, save_dp: dict, **kwargs) -> dict:
        save_dp['model_detected_presuppositions'] = response.model_dump()
        return save_dp

class CREPEFeedbackActionOperator(CREPEOperator):
    def __init__(self):
        self.action_name = "CREPE_Feedback_Action"
        self.dataloader = None
        self.response_cls = CREPEFeedbackActionResponse

    def add_data_module(self, model_name: str, file_dir: str = 'out', **kwargs):
        self.dataloader = instantiate_dataloader(dataset_name="CREPE", file_dir=file_dir, model_name=model_name)
        
    def load_data(self, **kwargs):
        return self.dataloader.load_data("CREPE_Presupposition_Extraction")

    def prepare_message(self, raw_dp: dict, system_role: str, **kwargs) -> str:
        template = CREPEFeedbackActionTemplate(**raw_dp, system_role=system_role)
        return template.generate()
    
    def parse_response_openai(self, response: dict, save_dp: dict, **kwargs) -> dict:
        save_dp['model_feedback_action'] = response['response']['body']['choices'][0]['message']['content']
        return save_dp
    
    def parse_response_transformers(self, response: BaseModel, save_dp: dict, **kwargs) -> dict:
        save_dp['model_feedback_action'] = response.model_dump()
        return save_dp