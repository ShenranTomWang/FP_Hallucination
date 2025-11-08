from abc import ABC, abstractmethod
from typing import List
from response import Response
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pydantic import BaseModel

class DataOperator(ABC):
    action_name: str
    response_cls: Response
    
    @abstractmethod
    def align_responses(self, dp: dict, **kwargs) -> dict:
        pass

    @abstractmethod
    def prepare_message(self, raw_dp: dict, **kwargs) -> str:
        pass
    
    @abstractmethod
    def run_transformer_model(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, messages: List[str], device: torch.DeviceObjType, **kwargs) -> Response:
        pass
    
    @abstractmethod
    def message2openai_request(self, id: str, model: str, messages: List[str], **kwargs) -> dict:
        pass

    @abstractmethod
    def parse_response_openai(self, response: dict, save_dp: dict, **kwargs) -> dict:
        pass
    
    @abstractmethod
    def parse_response_transformers(self, response: BaseModel, save_dp: dict, **kwargs) -> dict:
        pass 

    @abstractmethod
    def evaluate(self, eval_dp: dict, run_bleurt: bool) -> tuple:
        """Evaluate a data point with ROUGE1-F1, ROUGEL-F1 and BleuRT (optional)

        Args:
            eval_dp (dict): datapoint for evaluation
            run_bleurt (bool): whether to run BLEURT evaluation

        Returns:
            dict: evaluation results, ROUGE1-F1, ROUGEL-F1, BLEURT (if run_bleurt is True)
        """
        pass
    
    @abstractmethod
    def add_data_module(self, file_dir: str = None, **kwargs):
        pass
    
    @abstractmethod
    def load_data(self, **kwargs) -> List[dict]:
        pass

    @abstractmethod
    def save_top_bottom_k(self, data: list, score_key: str, k: int, out_dir: str):
        """Save the top and bottom k data points based on a specific score.

        Args:
            data (list): The dataset to process.
            score_key (str): The key to use for scoring, one of 'rouge1_f1', 'rougeL_f1', 'bleurt_f1'.
            k (int): The number of top/bottom entries to save.
            out_dir (str): The directory to save the results.
        """
        pass
