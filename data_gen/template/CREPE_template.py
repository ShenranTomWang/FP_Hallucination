from .template import PresuppositionExtractionTemplate, FeedbackActionTemplate
from typing import List
        
class CREPEPresuppositionExtractionTemplate(PresuppositionExtractionTemplate):
    def __init__(self, question: str, few_shot_data: List[str], **kwargs):
        super().__init__(question=question, few_shot_data=few_shot_data)

class CREPEFeedbackActionTemplate(FeedbackActionTemplate):
    def __init__(self, question: str, model_detected_presuppositions: str, few_shot_data: List[str], **kwargs):
        super().__init__(question=question, model_detected_presuppositions=model_detected_presuppositions, few_shot_data=few_shot_data)
