from typing import List, Dict
from .evaluator import Evaluator
from .utils import rouge1_f1, rougeL_f1, bleurt_score, bert_score_f1
from numpy import mean

class CREPEPresuppositionExtractionEvaluator(Evaluator):
    def __init__(self, presuppositions: List[str], raw_presuppositions: List[str], model_answer: Dict[str, List[str]] | List[str], use_aligned: bool, **kwargs):
        self.presuppositions = [p.strip() for p in presuppositions if p.strip() != '']
        self.raw_presuppositions = [rp.strip() for rp in raw_presuppositions if rp.strip() != '']
        self.model_answer = model_answer['presuppositions'] if isinstance(model_answer, dict) else model_answer
        self.use_aligned = use_aligned
    
    def evaluate_rouge1_f1(self) -> float:
        if len(self.presuppositions) == 0:
            return 0.0 if len(self.model_answer) == 0 else 1.0
        rouge1_f1s = []
        for i, answer in enumerate(self.model_answer):
            if self.use_aligned:
                rouge1_f1s.append(rouge1_f1(answer, [self.presuppositions[i]]))
            else:
                rouge1_f1s.append(rouge1_f1(answer, [" ".join(self.presuppositions).strip()]))
        return mean(rouge1_f1s) if len(rouge1_f1s) > 0 else 0.0

    def evaluate_rougeL_f1(self) -> float:
        if len(self.presuppositions) == 0:
            return 0.0 if len(self.model_answer) == 0 else 1.0
        rougeL_f1s = []
        for i, answer in enumerate(self.model_answer):
            if self.use_aligned:
                rougeL_f1s.append(rougeL_f1(answer, [self.presuppositions[i]]))
            else:
                rougeL_f1s.append(rougeL_f1(answer, [" ".join(self.presuppositions).strip()]))
        return mean(rougeL_f1s) if len(rougeL_f1s) > 0 else 0.0

    def evaluate_bleurt_f1(self) -> float:
        if len(self.presuppositions) == 0:
            return 0.0 if len(self.model_answer) == 0 else 1.0
        if self.use_aligned:
            bleurt_score_f1s = []
            for i, answer in enumerate(self.model_answer):
                bleurt_score_f1s.append(bleurt_score([answer], [self.presuppositions[i]]))
            return mean(bleurt_score_f1s) if len(bleurt_score_f1s) > 0 else 0.0
        return bleurt_score(self.model_answer, [" ".join(self.presuppositions).strip()])

    def evaluate_bert_score_f1(self) -> float:
        if len(self.presuppositions) == 0:
            return 0.0 if len(self.model_answer) == 0 else 1.0
        if self.use_aligned:
            bert_score_f1s = []
            for i, answer in enumerate(self.model_answer):
                bert_score_f1s.append(bert_score_f1([answer], [self.presuppositions[i]]))
            return mean(bert_score_f1s) if len(bert_score_f1s) > 0 else 0.0
        return bert_score_f1(self.model_answer, [presuppositions.strip() for presuppositions in self.presuppositions])