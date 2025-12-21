from .template import PresuppositionExtractionTemplate, FeedbackActionTemplate, FinalAnswerTemplate
from typing import List, Dict
        
class CREPEPresuppositionExtractionTemplate(PresuppositionExtractionTemplate):
    def __init__(self, question: str, few_shot_data: List[Dict], **kwargs):
        super().__init__(question=question, few_shot_data=few_shot_data)

class CREPEFeedbackActionTemplate(FeedbackActionTemplate):
    def __init__(self, question: str, model_detected_presuppositions: str, few_shot_data: List[Dict], **kwargs):
        super().__init__(question=question, model_detected_presuppositions=model_detected_presuppositions, few_shot_data=few_shot_data)

class CREPEFinalAnswerTemplate(FinalAnswerTemplate):
    def __init__(self, question: str, model_feedback_action: str, few_shot_data: List[Dict], **kwargs):
        super().__init__(question=question, model_feedback_action=model_feedback_action, few_shot_data=few_shot_data)