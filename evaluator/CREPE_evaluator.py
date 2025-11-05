from typing import List
from .evaluator import Evaluator
from .utils import rouge1_f1, rougeL_f1, bleurt_score, bert_score

class CREPEEvaluator(Evaluator):
    def __init__(self, presuppositions: List[str], raw_presuppositions: List[str], model_answer: List[str], **kwargs):
        self.presuppositions = presuppositions
        self.raw_presuppositions = raw_presuppositions
        self.model_answer = model_answer
    
    def evaluate_rouge1_f1(self):
        if len(self.raw_presuppositions) == 0:
            return 0.0
        rouge1_f1s = []
        for answer in self.model_answer:
            rouge1_f1s.append(rouge1_f1(answer, [" ".join(self.raw_presuppositions)]))
        return max(rouge1_f1s)

    def evaluate_rougeL_f1(self):
        if len(self.raw_presuppositions) == 0:
            return 0.0
        rougeL_f1s = []
        for answer in self.model_answer:
            rougeL_f1s.append(rougeL_f1(answer, [" ".join(self.raw_presuppositions)]))
        return max(rougeL_f1s)

    def evaluate_bleurt_f1(self):
        if len(self.raw_presuppositions) == 0:
            return 0.0
        return bleurt_score(self.model_answer, [" ".join(self.raw_presuppositions)])
    
    def evaluate_bert_score_f1(self):
        if len(self.raw_presuppositions) == 0:
            return 0.0
        return bert_score(self.model_answer, self.raw_presuppositions)