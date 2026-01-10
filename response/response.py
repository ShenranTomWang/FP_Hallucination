from pydantic import BaseModel
from abc import abstractmethod

class Response(BaseModel):
    @classmethod
    @abstractmethod
    def model_validate_plain_text(cls, text: str):
        pass
    
    @abstractmethod
    def get(self):
        pass