from pydantic import BaseModel
from abc import ABC, abstractmethod

class Response(BaseModel):
    @classmethod
    @abstractmethod
    def model_validate_plain_text(cls, text: str):
        pass