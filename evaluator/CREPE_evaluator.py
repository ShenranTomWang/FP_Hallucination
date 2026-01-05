from typing import List, Dict
from .evaluator import Evaluator
from .utils import rouge1_f1, rougeL_f1, bleurt_score, bert_score_f1
from numpy import mean

class CREPEPresuppositionExtractionEvaluator(Evaluator):
    def __init__(
        self,
        presuppositions: List[str],
        raw_presuppositions: List[str],
        model_detected_presuppositions: Dict[str, List[str]] | List[str],
        use_aligned: str | None,
        presuppositions_aligned_recall: List[str] = None,
        model_detected_presuppositions_aligned_precision: List[str] = None,
        **kwargs
    ):
        self.presuppositions = [p.strip() for p in presuppositions if p.strip() != '']
        self.raw_presuppositions = [rp.strip() for rp in raw_presuppositions if rp.strip() != '']
        self.model_detected_presuppositions = model_detected_presuppositions['presuppositions'] if isinstance(model_detected_presuppositions, dict) else model_detected_presuppositions
        self.use_aligned = use_aligned
        self.presuppositions_aligned_recall = presuppositions_aligned_recall
        self.model_detected_presuppositions_aligned_precision = model_detected_presuppositions_aligned_precision
    
    def evaluate_rouge1_f1(self) -> float:
        rouge1_f1s = []
        if self.use_aligned and self.use_aligned == 'precision':
            if len(self.presuppositions) == 0:
                return 1.0 if len(self.model_detected_presuppositions_aligned_precision) == 0 else 0.0
            for i, answer in enumerate(self.model_detected_presuppositions_aligned_precision):
                rouge1_f1s.append(rouge1_f1(answer, [self.presuppositions[i]]))
        elif self.use_aligned and self.use_aligned == 'recall':
            if len(self.model_detected_presuppositions) == 0:
                return 1.0 if len(self.presuppositions_aligned_recall) == 0 else 0.0
            for i, answer in enumerate(self.presuppositions_aligned_recall):
                if len(self.model_detected_presuppositions) != len(self.presuppositions_aligned_recall):
                    breakpoint()
                rouge1_f1s.append(rouge1_f1(answer, [self.model_detected_presuppositions[i]]))
        else:
            for answer in self.model_detected_presuppositions:
                rouge1_f1s.append(rouge1_f1(answer, [" ".join(self.presuppositions)]))
        
        return mean(rouge1_f1s) if len(rouge1_f1s) > 0 else 0.0

    def evaluate_rougeL_f1(self) -> float:
        rougeL_f1s = []
        if self.use_aligned and self.use_aligned == 'precision':
            if len(self.presuppositions) == 0:
                return 1.0 if len(self.model_detected_presuppositions_aligned_precision) == 0 else 0.0
            for i, answer in enumerate(self.model_detected_presuppositions_aligned_precision):
                rougeL_f1s.append(rougeL_f1(answer, [self.presuppositions[i]]))
        elif self.use_aligned and self.use_aligned == 'recall':
            if len(self.model_detected_presuppositions) == 0:
                return 1.0 if len(self.presuppositions_aligned_recall) == 0 else 0.0
            for i, answer in enumerate(self.presuppositions_aligned_recall):
                rougeL_f1s.append(rougeL_f1(answer, [self.model_detected_presuppositions[i]]))
        else:
            for answer in self.model_detected_presuppositions:
                rougeL_f1s.append(rougeL_f1(answer, [" ".join(self.presuppositions)]))

        return mean(rougeL_f1s) if len(rougeL_f1s) > 0 else 0.0

    def evaluate_bleurt_f1(self) -> float:
        bleurt_f1s = []
        if self.use_aligned and self.use_aligned == 'precision':
            if len(self.presuppositions) == 0:
                return 1.0 if len(self.model_detected_presuppositions_aligned_precision) == 0 else 0.0
            for i, answer in enumerate(self.model_detected_presuppositions_aligned_precision):
                bleurt_f1s.append(bleurt_score([answer], [self.presuppositions[i]]))
        elif self.use_aligned and self.use_aligned == 'recall':
            if len(self.model_detected_presuppositions) == 0:
                return 1.0 if len(self.presuppositions_aligned_recall) == 0 else 0.0
            for i, answer in enumerate(self.presuppositions_aligned_recall):
                bleurt_f1s.append(bleurt_score([answer], [self.model_detected_presuppositions[i]]))
        else:
            return bleurt_score(self.model_detected_presuppositions, [presuppositions.strip() for presuppositions in self.presuppositions])
        
        return mean(bleurt_f1s) if len(bleurt_f1s) > 0 else 0.0

    def evaluate_bert_score_f1(self) -> float:
        bleurt_f1s = []
        if self.use_aligned and self.use_aligned == 'precision':
            if len(self.presuppositions) == 0:
                return 1.0 if len(self.model_detected_presuppositions_aligned_precision) == 0 else 0.0
            for i, answer in enumerate(self.model_detected_presuppositions_aligned_precision):
                bleurt_f1s.append(bert_score_f1([answer], [self.presuppositions[i]]))
        elif self.use_aligned and self.use_aligned == 'recall':
            if len(self.model_detected_presuppositions) == 0:
                return 1.0 if len(self.presuppositions_aligned_recall) == 0 else 0.0
            for i, answer in enumerate(self.presuppositions_aligned_recall):
                bleurt_f1s.append(bert_score_f1([answer], [self.model_detected_presuppositions[i]]))
        else:
            return bert_score_f1(self.model_detected_presuppositions, [presuppositions.strip() for presuppositions in self.presuppositions])
        
        return mean(bleurt_f1s) if len(bleurt_f1s) > 0 else 0.0
    
class CREPEFinalAnswerEvaluator(Evaluator):
    def __init__(
        self,
        comment: str,
        model_final_answer: str,
        **kwargs
    ):
        self.comment = comment.strip()
        self.model_final_answer = model_final_answer.strip()
    
    def evaluate_rouge1_f1(self):
        return rouge1_f1(self.model_final_answer, [self.comment])
    
    def evaluate_rougeL_f1(self):
        return rougeL_f1(self.model_final_answer, [self.comment])
    
    def evaluate_bleurt_f1(self):
        return bleurt_score([self.model_final_answer], [self.comment])
    
    def evaluate_bert_score_f1(self):
        return bert_score_f1([self.model_final_answer], [self.comment])