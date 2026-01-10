from abc import ABC, abstractmethod
from typing import Dict
from minicheck.minicheck import MiniCheck

class MiniCheckOperator(ABC):
    model: MiniCheck
    
    def __init__(self, model_name: str, cache_dir: str):
        self.model = MiniCheck(model_name=model_name, cache_dir=cache_dir)
        
    @abstractmethod
    def check(self, dp: Dict[str, any], **kwargs) -> Dict[str, any]:
        pass