from .evaluator import Evaluator
from .utils import rouge1_f1, rougeL_f1, bleurt_score

class CREPEEvaluator(Evaluator):
    def __init__(self, presuppositions: list, raw_presuppositions: list, model_answer: str, **kwargs):
        self.presuppositions = " ".join(presuppositions)
        self.raw_presuppositions = " ".join(raw_presuppositions)
        self.model_answer = model_answer
    
    def evaluate_rouge1_f1(self):
        return rouge1_f1(self.model_answer, [self.presuppositions])

    def evaluate_rougeL_f1(self):
        return rougeL_f1(self.model_answer, [self.presuppositions])

    def evaluate_bleurt_f1(self):
        return bleurt_score(self.model_answer, [self.presuppositions])
