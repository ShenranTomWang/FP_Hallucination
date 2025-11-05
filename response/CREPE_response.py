from typing import List
from .response import Response

class CREPEResponse(Response):
    pass

class CREPEPresuppositionExtractionResponse(CREPEResponse):
    presuppositions: List[str]
    
class CREPEFeedbackActionResponse(CREPEResponse):
    feedback: str
    action: str