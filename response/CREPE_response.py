from typing import List
from .response import Response

class CREPEResponse(Response):
    pass

class CREPEPresuppositionExtractionResponse(CREPEResponse):
    presuppositions: List[str]
    
    def model_validate_plain_text(self, text: str):
        presuppositions = [line.strip() for line in text.split("\n")]
        return CREPEPresuppositionExtractionResponse(presuppositions=presuppositions)
    
class CREPEFeedbackActionResponse(CREPEResponse):
    feedback: str
    action: str
    
    def model_validate_plain_text(self, text: str):
        lines = text.split("\n")
        feedback = ""
        action = ""
        for line in lines:
            if line.startswith("Feedback:"):
                feedback = line[len("Feedback:"):].strip()
            elif line.startswith("Action:"):
                action = line[len("Action:"):].strip()
        return CREPEFeedbackActionResponse(feedback=feedback, action=action)