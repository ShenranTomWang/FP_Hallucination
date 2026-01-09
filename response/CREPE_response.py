from typing import List
from .response import Response

class CREPEResponse(Response):
    pass

class CREPEPresuppositionExtractionResponse(CREPEResponse):
    presuppositions: List[str]
    
    @classmethod
    def model_validate_plain_text(cls, text: str, model_role: str = "assistant"):
        text = text.lower().replace(f"{model_role}\n", "")
        presuppositions = [line.strip() for line in text.split("\n")]
        return cls(presuppositions=presuppositions)
    
    def get(self):
        return self.presuppositions
    
class CREPEFeedbackActionResponse(CREPEResponse):
    feedback_action: str
    
    @classmethod
    def model_validate_plain_text(cls, text: str):
        return cls(feedback_action=text)
    
    def get(self):
        return self.feedback_action

class CREPEFinalAnswerResponse(CREPEResponse):
    answer: str
    
    @classmethod
    def model_validate_plain_text(cls, text: str, model_role: str = "assistant"):
        text = text.lower().replace(f"{model_role}\n", "")
        return cls(answer=text.strip())
    
    def get(self):
        return self.answer
    
class CREPEEntailmentCountingResponse(CREPEResponse):
    count: int | str
    
    @classmethod
    def model_validate_plain_text(cls, text: str):
        try:
            count = int(text.strip())
        except ValueError:
            count = text.strip()
        return cls(count=count)
    
    def get(self):
        return self.count