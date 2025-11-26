from pydantic import BaseModel
from abc import ABC, abstractmethod

class Response(BaseModel):
    @abstractmethod
    def model_validate_plain_text(self, text: str):
        pass