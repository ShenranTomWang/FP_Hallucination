from typing import List
from .evaluator import Evaluator
from .utils import rouge1_f1, rougeL_f1, bleurt_score, bert_score_f1

class CREPEPresuppositionExtractionEvaluator(Evaluator):
    def __init__(self, presuppositions: List[str], raw_presuppositions: List[str], model_answer: List[str], **kwargs):
        self.presuppositions = [p.strip() for p in presuppositions if p.strip() != '']
        self.raw_presuppositions = [rp.strip() for rp in raw_presuppositions if rp.strip() != '']
        self.model_answer = model_answer
    
    def evaluate_rouge1_f1(self) -> float:
        if len(self.raw_presuppositions) == 0:
            return 0.0
        rouge1_f1s = []
        for answer in self.model_answer:
            rouge1_f1s.append(rouge1_f1(answer, [" ".join(self.raw_presuppositions).strip()]))
        return max(rouge1_f1s)

    def evaluate_rougeL_f1(self) -> float:
        if len(self.raw_presuppositions) == 0:
            return 0.0
        rougeL_f1s = []
        for answer in self.model_answer:
            rougeL_f1s.append(rougeL_f1(answer, [" ".join(self.raw_presuppositions).strip()]))
        return max(rougeL_f1s)

    def evaluate_bleurt_f1(self) -> float:
        if len(self.raw_presuppositions) == 0:
            return 0.0
        return bleurt_score(self.model_answer, [" ".join(self.raw_presuppositions).strip()])

    def evaluate_bert_score_f1(self) -> float:
        if len(self.raw_presuppositions) == 0:
            return 0.0
        return bert_score_f1(self.model_answer, [presuppositions.strip() for presuppositions in self.raw_presuppositions])