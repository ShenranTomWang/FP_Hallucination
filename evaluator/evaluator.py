from abc import ABC, abstractmethod

class Evaluator(ABC):
    @abstractmethod
    def evaluate_rouge1_f1(self) -> float:
        pass
    
    @abstractmethod
    def evaluate_rougeL_f1(self) -> float:
        pass
    
    @abstractmethod
    def evaluate_bleurt_f1(self) -> float:
        pass
    
    @abstractmethod
    def evaluate_bert_score_f1(self) -> float:
        pass
    
    @abstractmethod
    def evaluate_fp_score(self, system_role: str = "system", model_role: str = "assistant", user_role: str = "user") -> int:
        pass