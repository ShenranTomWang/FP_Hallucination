import json
from pathlib import Path
from data_gen.template import CREPEPresuppositionExtractionTemplate, CREPEFeedbackActionTemplate, CREPEFinalAnswerTemplate, CREPEDirectQATemplate
from evaluator import CREPEPresuppositionExtractionEvaluator, CREPEFinalAnswerEvaluator
from .data_operator import DataOperator
import random, os
from response import CREPEPresuppositionExtractionResponse, CREPEFeedbackActionResponse, CREPEFinalAnswerResponse
from pydantic import BaseModel
from evaluator.utils import bert_score_f1

class CREPEOperator(DataOperator):
    answer_key: str
    action_name: str
    
    def __init__(self):
        self.exclude_domains = ['reddit.com']
        
    def parse_response_openai(self, response: dict | str, save_dp: dict, **kwargs) -> dict:
        if isinstance(response, str):
            save_dp[self.answer_key] = response
        else:
            save_dp[self.answer_key] = response['response']['body']['choices'][0]['message']['content']
        save_dp[self.answer_key] = self.response_cls.model_validate_plain_text(save_dp[self.answer_key]).model_dump()
        return save_dp
    
    def parse_response_gemini(self, response: dict | str, save_dp: dict, **kwargs) -> dict:
        if isinstance(response, str):
            save_dp[self.answer_key] = response
        else:
            save_dp[self.answer_key] = response['response']['text']
        save_dp[self.answer_key] = self.response_cls.model_validate_plain_text(save_dp[self.answer_key]).model_dump()
        return save_dp
    
    def parse_response_transformers(self, response: BaseModel, save_dp: dict, **kwargs) -> dict:
        save_dp[self.answer_key] = response.model_dump()
        return save_dp

    def load_data(self, file_path: str, k: int = None, **kwargs):
        with open(file_path, 'r') as f:
            dataset = []
            for line in f:
                dp = json.loads(line.strip())
                dataset.append(dp)
        if not dataset[0].get('few_shot_data'):
            few_shot_data = []
            with open(str(Path(__file__).resolve().parent.parent / 'data_gen' / 'CREPE' / 'few_shot.jsonl'), 'r') as f:
                for line in f:
                    few_shot_dp = json.loads(line.strip())
                    few_shot_data.append(few_shot_dp)
            if k:
                k = min(k, len(few_shot_data))
                few_shot_data = random.sample(few_shot_data, k)
            for data in dataset:
                data['few_shot_data'] = few_shot_data
            with open(file_path, 'w') as f:
                for dp in dataset:
                    f.write(json.dumps(dp) + '\n')
        return dataset

    def save_top_bottom_k(
        self,
        data: list,
        score_key: str,
        k: int,
        out_dir: str,
        use_aligned: str | None,
        fname: str = 'top_{}_{}_{}.txt'
    ):
        key = f'{self.answer_key}_aligned_{use_aligned}' if use_aligned and use_aligned == 'precision' else self.answer_key
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
                f.write(f'GT Presuppositions: {dp["presuppositions"]}\n')
                if use_aligned == 'recall':
                    f.write(f'GT Presuppositions (aligned): {dp.get("presuppositions_aligned_recall", [])}\n')
                if use_aligned == 'precision':
                    f.write(f'Model Answer (Aligned): {dp[key]}\n')
                f.write(f'Model Answer: {dp[self.answer_key]}\n')
                f.write('-' * 20 + '\n\n')
        with open(os.path.join(out_dir, f'bottom_{k}_{score_key}_{self.action_name}.txt'), 'w') as f:
            for dp in sorted_data[:k]:
                f.write(f'{score_key}: {dp[score_key]:.4f}\n')
                f.write(f'id: {dp["id"]}\n')
                f.write(f'Question: {dp["question"]}\n')
                f.write(f'Comment: {dp.get("comment", "")}\n')
                f.write(f'GT Presuppositions: {dp["presuppositions"]}\n')
                if use_aligned == 'recall':
                    f.write(f'GT Presuppositions (aligned): {dp.get("presuppositions_aligned_recall", [])}\n')
                if use_aligned == 'precision':
                    f.write(f'Model Answer (Aligned): {dp[key]}\n')
                f.write(f'Model Answer: {dp[self.answer_key]}\n')
                f.write('-' * 20 + '\n\n')

class CREPEPresuppositionExtractionOperator(CREPEOperator):
    def __init__(self):
        self.action_name = "CREPE_Presupposition_Extraction"
        self.response_cls = CREPEPresuppositionExtractionResponse
        self.answer_key = "model_detected_presuppositions"
        super().__init__()

    def evaluate(self, eval_dp: dict, run_bleurt: bool, run_bert_score: bool, use_aligned: str | None) -> tuple:
        evaluator = CREPEPresuppositionExtractionEvaluator(**eval_dp, use_aligned=use_aligned)
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

    def align_response(self, dp: dict, model_type: str = None, **kwargs) -> dict:
        presuppositions_gt = dp['presuppositions']
        presuppositions = dp.get(self.answer_key, {}).get('presuppositions', [])
        aligned_map = dict()
        if len(presuppositions) == 0:
            dp['model_detected_presuppositions_aligned_precision'] = []
            return dp
        for i, p_gt in enumerate(presuppositions_gt):
            best_p = None
            best_score = -1
            for p in presuppositions:
                score = bert_score_f1([p_gt], [p], model_type=model_type)
                if score > best_score:
                    best_score = score
                    best_p = p
            if best_p is not None:
                aligned_map[i] = best_p
        dp['model_detected_presuppositions_aligned_precision'] = list(aligned_map.values())
        aligned_map = dict()
        if len(presuppositions_gt) == 0:
            dp['presuppositions_aligned_recall'] = []
            return dp
        for i, p in enumerate(presuppositions):
            best_p_gt = None
            best_score = -1
            for p_gt in presuppositions_gt:
                score = bert_score_f1([p_gt], [p], model_type=model_type)
                if score > best_score:
                    best_score = score
                    best_p_gt = p_gt
            if best_p_gt is not None:
                aligned_map[i] = best_p_gt
        dp['presuppositions_aligned_recall'] = list(aligned_map.values())
        return dp

    def prepare_message(self, raw_dp: dict, **kwargs) -> str:
        template = CREPEPresuppositionExtractionTemplate(**raw_dp, **kwargs)
        return template.generate()

class CREPEFeedbackActionOperator(CREPEOperator):
    def __init__(self):
        self.action_name = "CREPE_Feedback_Action"
        self.response_cls = CREPEFeedbackActionResponse
        self.answer_key = "model_feedback_action"
        super().__init__()
        
    def evaluate(self, eval_dp: dict, **kwargs) -> tuple:
        raise NotImplementedError("Feedback Action evaluation is not implemented.")
        
    def align_response(self, dp: dict, **kwargs) -> dict:
        return dp

    def prepare_message(self, raw_dp: dict, **kwargs) -> str:
        template = CREPEFeedbackActionTemplate(**raw_dp, **kwargs)
        return template.generate()
    
class CREPEFinalAnswerOperator(CREPEOperator):
    def __init__(self):
        self.action_name = "CREPE_Final_Answer"
        self.response_cls = CREPEFinalAnswerResponse
        self.answer_key = "model_final_answer"
        super().__init__()
        
    def evaluate(self, eval_dp: dict, **kwargs) -> tuple:
        raise NotImplementedError("Feedback Action evaluation is not supported.")
    
    def align_response(self, dp: dict, **kwargs) -> dict:
        return dp

    def evaluate(self, eval_dp: dict, run_bleurt: bool = False, run_bert_score: bool = False, **kwargs) -> tuple:
        evaluator = CREPEFinalAnswerEvaluator(**eval_dp, model_final_answer=eval_dp[self.answer_key]['answer'])
        rouge1_f1 = evaluator.evaluate_rouge1_f1()
        rougeL_f1 = evaluator.evaluate_rougeL_f1()
        if run_bert_score:
            bert_score_f1 = evaluator.evaluate_bert_score_f1()
        else:
            bert_score_f1 = None
        if run_bleurt:
            bleurt_f1 = evaluator.evaluate_bleurt_f1()
        else:
            bleurt_f1 = None
        return rouge1_f1, rougeL_f1, bleurt_f1, bert_score_f1
    
    def prepare_message(self, raw_dp: dict, **kwargs) -> str:
        template = CREPEFinalAnswerTemplate(**raw_dp, **kwargs)
        return template.generate()
    
class CREPEDirectQAOperator(CREPEOperator):
    def __init__(self):
        self.action_name = "CREPE_Direct_QA"
        self.response_cls = CREPEFinalAnswerResponse
        self.answer_key = "model_answer"
        super().__init__()
        
    def evaluate(self, eval_dp: dict, run_bleurt: bool = False, run_bert_score: bool = False, **kwargs) -> tuple:
        evaluator = CREPEFinalAnswerEvaluator(**eval_dp, model_final_answer=eval_dp[self.answer_key]['answer'])
        rouge1_f1 = evaluator.evaluate_rouge1_f1()
        rougeL_f1 = evaluator.evaluate_rougeL_f1()
        if run_bert_score:
            bert_score_f1 = evaluator.evaluate_bert_score_f1()
        else:
            bert_score_f1 = None
        if run_bleurt:
            bleurt_f1 = evaluator.evaluate_bleurt_f1()
        else:
            bleurt_f1 = None
        return rouge1_f1, rougeL_f1, bleurt_f1, bert_score_f1
    
    def align_response(self, dp: dict, **kwargs) -> dict:
        return dp
    
    def prepare_message(self, raw_dp: dict, **kwargs) -> str:
        template = CREPEDirectQATemplate(**raw_dp, **kwargs)
        return template.generate()
