import json
from pathlib import Path
from data_gen.template import CREPEPresuppositionExtractionTemplate, CREPEFeedbackActionTemplate, CREPEFinalAnswerTemplate, CREPEDirectQATemplate, CREPEMiniCheckFinalAnswerTemplate
from evaluator import CREPEPresuppositionExtractionEvaluator, CREPEFinalAnswerEvaluator
from .data_operator import DataOperator
import random, os
from response import CREPEPresuppositionExtractionResponse, CREPEFeedbackActionResponse, CREPEFinalAnswerResponse, Response
from pydantic import BaseModel
from evaluator.utils import bert_score_f1
from typing import Dict, List
import numpy as np

def _avg_report(data: List[Dict], measure: str = '', run_bleurt: bool = False, run_bert_score: bool = False, run_fp_score: bool = False):
    _measure = f'_{measure}' if measure != '' else ''
    rouge1_f1_key = f'rouge1_f1{_measure}'
    rougeL_f1_key = f'rougeL_f1{_measure}'
    avg_rouge1 = np.mean([dp[rouge1_f1_key] for dp in data if dp.get(rouge1_f1_key) is not None])
    avg_rougeL = np.mean([dp[rougeL_f1_key] for dp in data if dp.get(rougeL_f1_key) is not None])
    print(f'Average ROUGE-1 F1 {measure.capitalize()}: {avg_rouge1:.4f}')
    print(f'Average ROUGE-L F1 {measure.capitalize()}: {avg_rougeL:.4f}')
    if run_bleurt:
        bleurt_key = f'bleurt_f1{_measure}'
        avg_bleurt = np.mean([dp[bleurt_key] for dp in data if dp.get(bleurt_key) is not None])
        print(f'Average BLEURT F1 {measure.capitalize()}: {avg_bleurt:.4f}')
    if run_bert_score:
        bert_score_key = f'bert_score_f1{_measure}'
        avg_bert_score = np.mean([dp[bert_score_key] for dp in data if dp.get(bert_score_key) is not None])
        print(f'Average BERTScore F1 {measure.capitalize()}: {avg_bert_score:.4f}')
    if run_fp_score:
        fp_score_key = f'fp_score'
        avg_fp_score = np.mean([dp[fp_score_key] for dp in data if dp.get(fp_score_key) is not None and "false presupposition" in dp['labels']])
        print(f'Average FP Score {measure.capitalize()}: {avg_fp_score:.4f}')

class CREPEOperator(DataOperator):
    answer_key: str
    
    def __init__(self, action_name: str, response_cls: Response, answer_key: str, exclude_domains: List[str] = ["reddit.com"]):
        super().__init__(action_name, response_cls, exclude_domains)
        self.answer_key = answer_key
    
    def RAG_retrieve(self, dp: Dict[str, any], **kwargs):
        # TODO: finalize the design choice
        return dp['passages']
    
    def __init__(self):
        self.exclude_domains = ['reddit.com']
        
    def parse_response_openai(self, response: Dict | str, save_dp: Dict, **kwargs) -> Dict:
        if isinstance(response, str):
            save_dp[self.answer_key] = response
        else:
            save_dp[self.answer_key] = response['response']['body']['choices'][0]['message']['content']
        save_dp[self.answer_key] = self.response_cls.model_validate_plain_text(save_dp[self.answer_key]).model_dump()
        return save_dp
    
    def parse_response_gemini(self, response: Dict | str, save_dp: Dict, **kwargs) -> Dict:
        if isinstance(response, str):
            save_dp[self.answer_key] = response
        else:
            save_dp[self.answer_key] = response['response']['text']
        save_dp[self.answer_key] = self.response_cls.model_validate_plain_text(save_dp[self.answer_key]).model_dump()
        return save_dp
    
    def parse_response_transformers(self, response: BaseModel, save_dp: Dict, **kwargs) -> Dict:
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

class CREPEPresuppositionExtractionOperator(CREPEOperator):
    def __init__(self, action_name: str = "CREPE_Presupposition_Extraction", response_cls: Response = CREPEPresuppositionExtractionResponse, answer_key: str = "model_detected_presuppositions"):
        super().__init__(action_name=action_name, response_cls=response_cls, answer_key=answer_key)
        
    def print_eval_result(data: List[Dict], run_bleurt: bool = False, run_bert_score: bool = False, **kwargs):
        _avg_report(data=data, measure='precision', run_bleurt=run_bleurt, run_bert_score=run_bert_score)
        _avg_report(data=data, measure='recall', run_bleurt=run_bleurt, run_bert_score=run_bert_score)
        
    def evaluate(self, eval_dp: Dict, run_bleurt: bool = False, run_bert_score: bool = False, **kwargs) -> Dict:
        if eval_dp.get(self.answer_key) is None:
            eval_dp['rouge1_f1_precision'], eval_dp['rouge1_f1_recall'] = 0, 0
            eval_dp['rougeL_f1_precision'], eval_dp['rougeL_f1_recall'] = 0, 0
            if run_bleurt:
                eval_dp['bleurt_f1_precision'], eval_dp['bleurt_f1_recall'] = 0, 0
            if run_bert_score:
                eval_dp['bert_score_f1_precision'], eval_dp['bert_score_f1_recall'] = 0, 0
            return eval_dp
        eval_dp['rouge1_f1_precision'], eval_dp['rougeL_f1_precision'], bleurt_f1_precision, bert_score_f1_precision = self._evaluate(eval_dp, run_bleurt=run_bleurt, run_bert_score=run_bert_score, use_aligned="precision")
        eval_dp['rouge1_f1_recall'], eval_dp['rougeL_f1_recall'], bleurt_f1_recall, bert_score_f1_recall = self._evaluate(eval_dp, run_bleurt=run_bleurt, run_bert_score=run_bert_score, use_aligned="recall")
        if run_bleurt:
            eval_dp['bleurt_f1_precision'] = bleurt_f1_precision
            eval_dp['bleurt_f1_recall'] = bleurt_f1_recall
        if run_bert_score:
            eval_dp['bert_score_f1_precision'] = bert_score_f1_precision
            eval_dp['bert_score_f1_recall'] = bert_score_f1_recall
        return eval_dp

    def _evaluate(self, eval_dp: Dict, run_bleurt: bool, run_bert_score: bool, use_aligned: str | None) -> tuple:
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
    
    def save_top_bottom_k(self, data: List[Dict], k: int, out_dir: str, run_bleurt: bool = False, run_bert_score: bool = False, **kwargs):
        for score_key in ['rouge1_f1_precision', 'rougeL_f1_precision'] + (['bleurt_f1_precision'] if run_bleurt else []) + (['bert_score_f1_precision'] if run_bert_score else []):
            self._save_top_bottom_k(data, score_key, k, out_dir, use_aligned='precision')
        for score_key in ['rouge1_f1_recall', 'rougeL_f1_recall'] + (['bleurt_f1_recall'] if run_bleurt else []) + (['bert_score_f1_recall'] if run_bert_score else []):
            self._save_top_bottom_k(data, score_key, k, out_dir, use_aligned='recall')
    
    def _save_top_bottom_k(
        self,
        data: List[Dict],
        score_key: str,
        k: int,
        out_dir: str,
        use_aligned: str | None = None,
        **kwargs
    ):
        key = f'{self.answer_key}_aligned_{use_aligned}' if use_aligned and use_aligned == 'precision' else self.answer_key
        sorted_data = sorted(
            [dp for dp in data if dp.get(key) is not None and dp.get(score_key) is not None],
            key=lambda x: x[score_key]
        )
        
        def pretty_print(dp: Dict, f, score_key: str):
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

        sorted_data = [dp for dp in sorted_data if 'false presupposition' in dp['labels']]
        with open(os.path.join(out_dir, f'top_{k}_{score_key}_{self.action_name}.txt'), 'w') as f:
            for dp in sorted_data[-k:][::-1]:
                pretty_print(dp, f, score_key)
        with open(os.path.join(out_dir, f'bottom_{k}_{score_key}_{self.action_name}.txt'), 'w') as f:
            for dp in sorted_data[:k]:
                pretty_print(dp, f, score_key)

    def align_response(self, dp: Dict, model_type: str = None, **kwargs) -> Dict:
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

    def prepare_message(self, raw_dp: Dict, **kwargs) -> str:
        template = CREPEPresuppositionExtractionTemplate(**raw_dp, **kwargs)
        return template.generate()

class CREPEFeedbackActionOperator(CREPEOperator):
    def __init__(self, action_name: str = "CREPE_Feedback_Action", response_cls: Response = CREPEFeedbackActionResponse, answer_key: str = "model_feedback_action"):
        super().__init__(action_name=action_name, response_cls=response_cls, answer_key=answer_key)
    
    @staticmethod
    def print_eval_result(data: List[Dict], **kwargs):
        raise NotImplementedError("Feedback Action evaluation is not supported.")
        
    def save_top_bottom_k(self, data: List[Dict], k: int, out_dir: str, **kwargs):
        raise NotImplementedError("Feedback Action evaluation is not supported.")
    
    def evaluate(self, eval_dp: Dict, **kwargs) -> Dict:
        raise NotImplementedError("Feedback Action evaluation is not supported.")
        
    def align_response(self, dp: Dict, **kwargs) -> Dict:
        return dp

    def prepare_message(self, raw_dp: Dict, **kwargs) -> str:
        template = CREPEFeedbackActionTemplate(**raw_dp, **kwargs)
        return template.generate()
    
class CREPEFinalAnswerOperator(CREPEOperator):
    def __init__(self, action_name: str = "CREPE_Final_Answer", response_cls: Response = CREPEFinalAnswerResponse, answer_key: str = "model_final_answer"):
        super().__init__(action_name=action_name, response_cls=response_cls, answer_key=answer_key)
        
    @staticmethod
    def print_eval_result(data: List[Dict], run_bleurt: bool = False, run_bert_score: bool = False, run_fp_score: bool = False, **kwargs):
        _avg_report(data=data, run_bleurt=run_bleurt, run_bert_score=run_bert_score, run_fp_score=run_fp_score)
        
    def save_top_bottom_k(self, data: List[Dict], k: int, out_dir: str, run_bleurt: bool = False, run_bert_score: bool = False, run_fp_score: bool = False, **kwargs):
        for score_key in ['rouge1_f1', 'rougeL_f1'] + (['bleurt_f1'] if run_bleurt else []) + (['bert_score_f1'] if run_bert_score else []) + (['fp_score'] if run_fp_score else []):
            self._save_top_bottom_k(data, score_key, k, out_dir)
    
    def _save_top_bottom_k(self, data: List[Dict], score_key: str, k: int, out_dir: str, **kwargs):
        key = self.answer_key
        sorted_data = sorted(
            [dp for dp in data if dp.get(key) is not None and dp.get(score_key) is not None],
            key=lambda x: x[score_key]
        )
        
        def pretty_print(dp: Dict, f, score_key: str):
            f.write(f'{score_key}: {dp[score_key]:.4f}\n')
            f.write(f'id: {dp["id"]}\n')
            f.write(f'Question: {dp["question"]}\n')
            f.write(f'Comment: {dp.get("comment", "")}\n')
            f.write(f'GT Presuppositions: {dp["presuppositions"]}\n')
            f.write(f'Model Detected Presuppositions: {dp["model_detected_presuppositions"]["presuppositions"]}\n')
            f.write(f'Model Action Feedback: {dp["model_feedback_action"]["feedback_action"]}\n')
            f.write(f'Model Answer: {dp[self.answer_key]}\n')
            f.write('-' * 20 + '\n\n')

        sorted_data = [dp for dp in sorted_data if 'false presupposition' in dp['labels']]
        with open(os.path.join(out_dir, f'top_{k}_{score_key}_{self.action_name}.txt'), 'w') as f:
            for dp in sorted_data[-k:][::-1]:
                pretty_print(dp, f, score_key)
        with open(os.path.join(out_dir, f'bottom_{k}_{score_key}_{self.action_name}.txt'), 'w') as f:
            for dp in sorted_data[:k]:
                pretty_print(dp, f, score_key)
    
    def align_response(self, dp: Dict, **kwargs) -> Dict:
        return dp

    def evaluate(self, eval_dp: Dict, run_bleurt: bool = False, run_bert_score: bool = False, run_fp_score: bool = False, **kwargs) -> Dict:
        evaluator = CREPEFinalAnswerEvaluator(
            question=eval_dp['question'],
            comment=eval_dp['comment'],
            model_final_answer=eval_dp[self.answer_key]['answer'],
            presuppositions=eval_dp['presuppositions'],
            few_shot_data=eval_dp['few_shot_data']
        )
        rouge1_f1 = evaluator.evaluate_rouge1_f1()
        rougeL_f1 = evaluator.evaluate_rougeL_f1()
        eval_dp['rouge1_f1'] = rouge1_f1
        eval_dp['rougeL_f1'] = rougeL_f1
        if run_bert_score:
            bert_score_f1 = evaluator.evaluate_bert_score_f1()
            eval_dp['bert_score_f1'] = bert_score_f1
        if run_bleurt:
            bleurt_f1 = evaluator.evaluate_bleurt_f1()
            eval_dp['bleurt_f1'] = bleurt_f1
        if run_fp_score:    
            fp_score = evaluator.evaluate_fp_score()
            eval_dp['fp_score_raw'] = fp_score
            eval_dp['fp_score'] = (int(fp_score) / len(eval_dp['presuppositions'])) if len(eval_dp['presuppositions']) > 0 else 0.0
        return eval_dp
    
    def prepare_message(self, raw_dp: Dict, **kwargs) -> str:
        template = CREPEFinalAnswerTemplate(**raw_dp, **kwargs)
        return template.generate()

class CREPEMiniCheckFinalAnswerOperator(CREPEFinalAnswerOperator):
    def __init__(self, action_name: str = "CREPE_MiniCheck_Final_Answer", response_cls: Response = CREPEFinalAnswerResponse, answer_key: str = "model_final_answer"):
        super().__init__(action_name=action_name, response_cls=response_cls, answer_key=answer_key)

    def prepare_message(self, raw_dp: Dict, **kwargs) -> str:
        template = CREPEMiniCheckFinalAnswerTemplate(**raw_dp, **kwargs)
        return template.generate()
    
class CREPEDirectQAOperator(CREPEOperator):
    def __init__(self, action_name: str = "CREPE_Direct_QA", response_cls: Response = CREPEFinalAnswerResponse, answer_key: str = "model_answer"):
        super().__init__(action_name=action_name, response_cls=response_cls, answer_key=answer_key)
    
    @staticmethod
    def print_eval_result(data: List[Dict], run_bleurt: bool = False, run_bert_score: bool = False, run_fp_score: bool = False, **kwargs):
        _avg_report(data=data, run_bleurt=run_bleurt, run_bert_score=run_bert_score, run_fp_score=run_fp_score)
        
    def save_top_bottom_k(self, data: List[Dict], k: int, out_dir: str, run_bleurt: bool = False, run_bert_score: bool = False, run_fp_score: bool = False, **kwargs):
        for score_key in ['rouge1_f1', 'rougeL_f1'] + (['bleurt_f1'] if run_bleurt else []) + (['bert_score_f1'] if run_bert_score else []) + (['fp_score'] if run_fp_score else []):
            self._save_top_bottom_k(data, score_key, k, out_dir)
    
    def _save_top_bottom_k(self, data: List[Dict], score_key: str, k: int, out_dir: str, **kwargs):
        key = self.answer_key
        sorted_data = sorted(
            [dp for dp in data if dp.get(key) is not None and dp.get(score_key) is not None],
            key=lambda x: x[score_key]
        )
        
        def pretty_print(dp: Dict, f, score_key: str):
            f.write(f'{score_key}: {dp[score_key]:.4f}\n')
            f.write(f'id: {dp["id"]}\n')
            f.write(f'Question: {dp["question"]}\n')
            f.write(f'Comment: {dp.get("comment", "")}\n')
            f.write(f'GT Presuppositions: {dp["presuppositions"]}\n')
            f.write(f'Model Answer: {dp[self.answer_key]}\n')
            f.write('-' * 20 + '\n\n')

        sorted_data = [dp for dp in sorted_data if 'false presupposition' in dp['labels']]
        with open(os.path.join(out_dir, f'top_{k}_{score_key}_{self.action_name}.txt'), 'w') as f:
            for dp in sorted_data[-k:][::-1]:
                pretty_print(dp, f, score_key)
        with open(os.path.join(out_dir, f'bottom_{k}_{score_key}_{self.action_name}.txt'), 'w') as f:
            for dp in sorted_data[:k]:
                pretty_print(dp, f, score_key)
    
    def evaluate(
        self,
        eval_dp: Dict,
        run_bleurt: bool = False,
        run_bert_score: bool = False,
        run_fp_score: bool = False,
        system_role: str = "system",
        model_role: str = "assistant",
        user_role: str = "user",
        **kwargs
    ) -> Dict:
        evaluator = CREPEFinalAnswerEvaluator(
            question=eval_dp['question'],
            comment=eval_dp['comment'],
            model_final_answer=eval_dp[self.answer_key]['answer'],
            presuppositions=eval_dp['presuppositions'],
            few_shot_data=eval_dp['few_shot_data']
        )
        rouge1_f1 = evaluator.evaluate_rouge1_f1()
        rougeL_f1 = evaluator.evaluate_rougeL_f1()
        eval_dp['rouge1_f1'] = rouge1_f1
        eval_dp['rougeL_f1'] = rougeL_f1
        if run_bert_score:
            bert_score_f1 = evaluator.evaluate_bert_score_f1()
            eval_dp['bert_score_f1'] = bert_score_f1
        if run_bleurt:
            bleurt_f1 = evaluator.evaluate_bleurt_f1()
            eval_dp['bleurt_f1'] = bleurt_f1
        if run_fp_score:
            fp_score = evaluator.evaluate_fp_score(system_role=system_role, model_role=model_role, user_role=user_role)
            eval_dp['fp_score_raw'] = fp_score
            eval_dp['fp_score'] = (int(fp_score) / len(eval_dp['presuppositions'])) if len(eval_dp['presuppositions']) > 0 else 0.0
        return eval_dp
    
    def align_response(self, dp: Dict, **kwargs) -> Dict:
        return dp
    
    def prepare_message(self, raw_dp: Dict, **kwargs) -> str:
        template = CREPEDirectQATemplate(**raw_dp, **kwargs)
        return template.generate()
