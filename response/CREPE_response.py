from typing import List
from .response import Response

class CREPEResponse(Response):
    pass

class CREPEPresuppositionExtractionResponse(CREPEResponse):
    presuppositions: List[str]
    
    @classmethod
    def model_validate_plain_text(cls, text: str):
        text = text.replace("assistant\n", "")
        presuppositions = [line.strip() for line in text.split("\n")]
        return cls(presuppositions=presuppositions)
    
class CREPEFeedbackActionResponse(CREPEResponse):
    feedback: str
    action: str
    
    @classmethod
    def model_validate_plain_text(cls, text: str):
        lines = text.split("\n")
        feedback = ""
        action = ""
        for line in lines:
            if line.startswith("Feedback:"):
                feedback = line[len("Feedback:"):].strip()
            elif line.startswith("Action:"):
                action = line[len("Action:"):].strip()
        return cls(feedback=feedback, action=action)