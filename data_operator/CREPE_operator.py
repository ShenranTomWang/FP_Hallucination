import json
from pathlib import Path
from data_gen.template import CREPEPresuppositionExtractionTemplate, CREPEFeedbackActionTemplate
from evaluator import CREPEPresuppositionExtractionEvaluator
from .data_operator import DataOperator
from data_gen.data_loader import instantiate_dataloader
import random, os
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from response import CREPEPresuppositionExtractionResponse, CREPEFeedbackActionResponse, Response
import torch
from typing import List
from pydantic import BaseModel
from evaluator.utils import bert_score_f1

class CREPEOperator(DataOperator):
    transformer_model: outlines.models.Transformers = None
    answer_key: str
    response_cls: Response
    action_name: str

    def evaluate(self, eval_dp: dict, run_bleurt: bool, run_bert_score: bool, use_aligned: bool) -> tuple:
        key = f'{self.answer_key}_aligned' if use_aligned else self.answer_key
        if eval_dp.get(key) is None:
            return 0.0, 0.0, 0.0, 0.0
        evaluator = CREPEPresuppositionExtractionEvaluator(**eval_dp, model_answer=eval_dp[key], use_aligned=use_aligned)
        rouge1_f1 = evaluator.evaluate_rouge1_f1()
        rougeL_f1 = evaluator.evaluate_rougeL_f1()
        if run_bleurt:
            bleurt_f1 = evaluator.evaluate_bleurt_f1()
        else:
            bleurt_f1 = None
        if run_bert_score:
            bert_score_f1 = evaluator.evaluate_bert_score_f1()
        else:
            bert_score_f1 = None
        return rouge1_f1, rougeL_f1, bleurt_f1, bert_score_f1

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
        
    def parse_response_openai(self, response: dict, save_dp: dict, **kwargs) -> dict:
        save_dp[self.answer_key] = response['response']['body']['choices'][0]['message']['content']
        return save_dp
    
    def parse_response_transformers(self, response: BaseModel, save_dp: dict, **kwargs) -> dict:
        save_dp[self.answer_key] = response.model_dump()
        return save_dp

    def run_transformer_model(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, messages: List[str], device: torch.DeviceObjType, json_format: bool = False, **kwargs) -> Response:
        if json_format:
            if not self.transformer_model:
                self.transformer_model = outlines.from_transformers(
                    model, tokenizer
                )
            prompt = tokenizer.apply_chat_template(messages)
            prompt = tokenizer.decode(prompt)
            response = self.transformer_model(prompt, self.response_cls, max_new_tokens=512)
            return self.response_cls.model_validate_json(response)
        else:
            model = model.to(device)
            prompt = tokenizer.apply_chat_template(messages)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            output_ids = model.generate(input_ids, max_new_tokens=512)
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            response_text = output_text[len(prompt):].strip()
            return self.response_cls.model_validate_plain_text(response_text)

    def save_top_bottom_k(self, data: list, score_key: str, k: int, out_dir: str, use_aligned: bool, fname: str = 'top_{}_{}_{}.txt'):
        key = f'{self.answer_key}_aligned' if use_aligned else self.answer_key
        sorted_data = sorted(
            [dp for dp in data if dp.get(key) is not None and dp.get(score_key) is not None],
            key=lambda x: x[score_key]
        )
        sorted_data = [dp for dp in sorted_data if len(dp['presuppositions']) > 0]
        with open(os.path.join(out_dir, fname.format(k, score_key, self.action_name)), 'w') as f:
            for dp in sorted_data[-k:][::-1]:
                f.write(f'{score_key}: {dp[score_key]:.4f}\n')
                f.write(f'id: {dp["id"]}\n')
                f.write(f'Question: {dp["question"]}\n')
                f.write(f'Comment: {dp.get("comment", "")}\n')
                f.write(f'GT Presuppositions: {"; ".join(dp["presuppositions"])}\n')
                if use_aligned:
                    f.write(f'Model Answer (Aligned): {dp[key]}\n')
                f.write(f'Model Answer: {dp[self.answer_key]}\n')
                f.write('-' * 20 + '\n\n')
        with open(os.path.join(out_dir, f'bottom_{k}_{score_key}_{self.action_name}.txt'), 'w') as f:
            for dp in sorted_data[:k]:
                f.write(f'{score_key}: {dp[score_key]:.4f}\n')
                f.write(f'id: {dp["id"]}\n')
                f.write(f'Question: {dp["question"]}\n')
                f.write(f'Comment: {dp.get("comment", "")}\n')
                f.write(f'GT Presuppositions: {"; ".join(dp["presuppositions"])}\n')
                if use_aligned:
                    f.write(f'Model Answer (Aligned): {dp[key]}\n')
                f.write(f'Model Answer: {dp[self.answer_key]}\n')
                f.write('-' * 20 + '\n\n')

class CREPEPresuppositionExtractionOperator(CREPEOperator):
    def __init__(self):
        self.action_name = "CREPE_Presupposition_Extraction"
        self.dataloader = None
        self.response_cls = CREPEPresuppositionExtractionResponse
        self.answer_key = "model_detected_presuppositions"

    def align_response(self, dp: dict, model_type: str = None, **kwargs) -> dict:
        presuppositions_gt = dp['presuppositions']
        presuppositions = dp.get(self.answer_key, {}).get('presuppositions', [])
        if len(presuppositions) == 0 or len(presuppositions_gt) == 0:
            dp['model_detected_presuppositions_aligned'] = []
            return dp
        aligned_map = dict()
        for p_gt in presuppositions_gt:
            best_p = None
            best_score = -1
            for p in presuppositions:
                score = bert_score_f1([p_gt], [p], model_type=model_type)
                if score > best_score:
                    best_score = score
                    best_p = p
            if best_p is not None:
                aligned_map[p_gt] = best_p
        dp['model_detected_presuppositions_aligned'] = list(aligned_map.values())
        return dp

    def add_data_module(self, file_dir: str = 'dataset', **kwargs):
        self.dataloader = instantiate_dataloader(dataset_name="CREPE", file_dir=file_dir)

    def load_data(self, split: str, k: int = None, **kwargs):
        dataset = self.dataloader.load_data(split=split)
        few_shot_ids = []
        with open(str(Path(__file__).resolve().parent.parent / 'data_gen' / 'CREPE' / 'few_shot.jsonl'), 'r') as f:
            for line in f:
                few_shot_id = json.loads(line.strip())['id']
                few_shot_ids.append(few_shot_id)
        few_shot_data = self.dataloader.load_data(split='train')
        few_shot_data = [data for data in few_shot_data if data['id'] in few_shot_ids]
        if k:
            few_shot_data = random.sample(few_shot_data, k)
        for data in dataset:
            data['few_shot_data'] = few_shot_data
        self.dataloader.save_data(dataset, split=split)
        return self.dataloader.load_data(split)

    def prepare_message(self, raw_dp: dict, system_role: str, json_format: bool = False, **kwargs) -> str:
        template = CREPEPresuppositionExtractionTemplate(**raw_dp, system_role=system_role, json_format=json_format)
        return template.generate()

class CREPEFeedbackActionOperator(CREPEOperator):
    def __init__(self):
        self.action_name = "CREPE_Feedback_Action"
        self.dataloader = None
        self.response_cls = CREPEFeedbackActionResponse
        self.answer_key = "model_feedback_action"
        
    def align_response(self, dp: dict, **kwargs) -> dict:
        return dp

    def add_data_module(self, model_name: str, file_dir: str = 'out', **kwargs):
        self.dataloader = instantiate_dataloader(dataset_name="CREPE", file_dir=file_dir, model_name=model_name)
        
    def load_data(self, **kwargs):
        return self.dataloader.load_data("CREPE_Presupposition_Extraction")

    def prepare_message(self, raw_dp: dict, system_role: str, json_format: bool = False, **kwargs) -> str:
        template = CREPEFeedbackActionTemplate(**raw_dp, system_role=system_role, json_format=json_format)
        return template.generate()