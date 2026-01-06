from typing import List
from .response import Response

class CREPEResponse(Response):
    pass

class CREPEPresuppositionExtractionResponse(CREPEResponse):
    presuppositions: List[str]
    
    @classmethod
    def model_validate_plain_text(cls, text: str):
        text = text.lower().replace("assistant\n", "")
        presuppositions = [line.strip() for line in text.split("\n")]
        return cls(presuppositions=presuppositions)
    
class CREPEFeedbackActionResponse(CREPEResponse):
    feedback_action: str
    
    @classmethod
    def model_validate_plain_text(cls, text: str):
        return cls(feedback_action=text)

class CREPEFinalAnswerResponse(CREPEResponse):
    answer: str
    
    @classmethod
    def model_validate_plain_text(cls, text: str):
        text = text.lower().replace("assistant\n", "")
        return cls(answer=text.strip())