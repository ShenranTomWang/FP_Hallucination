from abc import ABC, abstractmethod
from typing import List, Dict
from response import Response
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
from pydantic import BaseModel

class DataOperator(ABC):
    action_name: str
    response_cls: Response
    exclude_domains: List[str] = []
    RAG_encoder: AutoModel = None
    RAG_tokenizer: AutoTokenizer = None
    
    def __init__(self, action_name: str, response_cls: Response, exclude_domains: List[str] = []):
        self.action_name = action_name
        self.response_cls = response_cls
        self.exclude_domains = exclude_domains
    
    @torch.inference_mode()
    def _RAG_compute_similarity(self, query: str, passages: List[str], **kwargs) -> List[float]:
        if not self.RAG_encoder or not self.RAG_tokenizer:
            self.RAG_encoder = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
            self.RAG_encoder.eval()
            self.RAG_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
        query_tokenized = self.RAG_tokenizer(query, return_tensors='pt', truncation=True, padding=True)
        passage_tokenized = self.RAG_tokenizer(passages, return_tensors='pt', truncation=True, padding=True)
        # TODO: finalize the design choice
        pass
    
    def RAG_retrieve(self, dp: Dict[str, any], **kwargs) -> List[str]:
        # TODO: finalize the design choice
        pass
    
    @staticmethod
    @abstractmethod
    def print_eval_result(data: List[dict], **kwargs):
        pass
    
    @abstractmethod
    def align_response(self, dp: dict, **kwargs) -> dict:
        pass

    @abstractmethod
    def prepare_message(self, raw_dp: dict, **kwargs) -> str:
        pass
    
    @torch.inference_mode()
    def run_transformers_model(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        messages: List[str],
        device: torch.DeviceObjType,
        **kwargs
    ) -> Response:
        # if json_format:
        #     if not self.transformer_model:
        #         self.transformer_model = outlines.from_transformers(
        #             model, tokenizer
        #         )
        #     prompt = tokenizer.apply_chat_template(messages)
        #     prompt = tokenizer.decode(prompt)
        #     response = self.transformer_model(prompt, self.response_cls, max_new_tokens=512)
        #     return self.response_cls.model_validate_json(response)
        # else:
        model = model.to(device)
        prompt = tokenizer.apply_chat_template(messages, return_tensors='pt').to(device)
        output_ids = model.generate(prompt, max_new_tokens=512)[:, prompt.shape[1]:]
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return self.response_cls.model_validate_plain_text(output_text)
    
    def message2openai_request(self, id: str, model: str, messages: List[Dict[str, str]], use_web_search: bool = False, **kwargs) -> dict:
        return {
            "custom_id": id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
                "tools": [
                    {
                        "type": "web_search",
                        "filters": {
                            "allowed_domains": [
                                "wikipedia.org"
                            ]
                        }
                    }
                ] if use_web_search else []
            }
        }
    
    def message2gemini_request(self, id: str, messages: List[Dict[str, str]], temperature: float = 0.0, use_web_search: bool = False, **kwargs) -> dict:
        return {
            "key": id,
            "request": {
                "contents": [
                    {
                        "role": message['role'],
                        "parts": [
                            {"text": message['content']}
                        ]
                    } for message in messages[1:]
                ],
                "config": {
                    "system_instruction": {
                        "parts": [
                            {"text": messages[0]['content']}
                        ]
                    },
                    "temperature": temperature,
                    "config": {
                        "tools": [
                            {"google_search": {"exclude_domains": self.exclude_domains}}
                        ] if use_web_search else []
                    }
                }
            }
        }

    @abstractmethod
    def parse_response_openai(self, response: dict, save_dp: dict, **kwargs) -> dict:
        pass
    
    @abstractmethod
    def parse_response_transformers(self, response: BaseModel, save_dp: dict, **kwargs) -> dict:
        pass 

    @abstractmethod
    def evaluate(self, eval_dp: dict, **kwargs) -> Dict:
        """Evaluate a data point with ROUGE1-F1, ROUGEL-F1

        Args:
            eval_dp (dict): datapoint for evaluation
            **kwargs: additional arguments

        Returns:
            dict: evaluation results stored as keys in eval_dp
        """
        pass
    
    @abstractmethod
    def load_data(self, **kwargs) -> List[dict]:
        pass

    @abstractmethod
    def save_top_bottom_k(self, data: list, k: int, out_dir: str, **kwargs):
        """Save the top and bottom k data points based on scores specific to dataset type.

        Args:
            data (list): The dataset to process.
            k (int): The number of top/bottom entries to save.
            out_dir (str): The directory to save the results.
            fname (str): The filename pattern to use for saving.
            **kwargs: Additional arguments.
        """
        pass
