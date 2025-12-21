from abc import ABC, abstractmethod
from typing import List, Dict

class Template(ABC):
    @abstractmethod
    def generate(self, **kwargs) -> str | List[Dict]:
        pass
    
class PresuppositionExtractionFewShotExample(Template):
    def __init__(self, question: str, presuppositions: List[str], **kwargs):
        self.question = question
        self.presuppositions = presuppositions

    def generate(self, **kwargs) -> List[Dict]:
        content = "\n".join(self.presuppositions) + "\n"
        return [
            {
                "role": "user",
                "content": self.question
            },
            {
                "role": "assistant",
                "content": content
            }
        ]

class PresuppositionExtractionTemplate(ABC, Template):
    question: str
    few_shot_data: List[PresuppositionExtractionFewShotExample]
    
    def __init__(self, question: str, few_shot_data: List[Dict], system_role: str = "system", **kwargs):
        self.question = question
        self.system_role = system_role
        self.few_shot_data = []
        for dp in few_shot_data:
            self.few_shot_data += PresuppositionExtractionFewShotExample(**dp).generate()
    
    def generate(self, **kwargs) -> List[Dict]:
        messages = [
            {
                "role": self.system_role,
                "content": f"""
                    You are a helpful assistant that analyzes the given question.
                    Your task is to extract presuppositions in the given question.
                    Notice that the presuppositions in a question could be true or false, and may be explicit or implicit.
                    There could be multiple presuppositions in a question, but there will always be at least one presupposition in the question.
                    Format your response as a list of presuppositions, separated by newlines.
                """
            },
            *self.few_shot_data,
            {"role": "user", "content": self.question}
        ]
        return messages

class FeedbackActionFewShotExample(Template):
    def __init__(self, question: str, presuppositions: List[str], raw_corrections: str, **kwargs):
        self.question = question
        self.presuppositions = presuppositions
        self.raw_corrections = "; ".join(raw_corrections)

    def generate(self, **kwargs) -> List[Dict]:
        presuppositions = self.presuppositions
        presuppositions.append("There is a clear and single answer to the question.")
        content = "\n".join(presuppositions) + "\n"
        feedback = f"The question contains false presuppositions that {self.presuppositions}."
        action = f"Correct the false assumptions that {self.presuppositions} and respond based on the corrected assumption."
        content += f"Feedback: {feedback}\nAction: {action}\n"
        return [
            {
                "role": "user",
                "content": self.question
            },
            {
                "role": "assistant",
                "content": content
            }
        ]

class FeedbackActionTemplate(Template):
    question: str
    model_detected_presuppositions: List[str]
    few_shot_data: List[FeedbackActionFewShotExample]
    
    def __init__(self, question: str, model_detected_presuppositions: str, few_shot_data: List[Dict], system_role: str = "system", **kwargs):
        self.question = question
        self.system_role = system_role
        self.model_detected_presuppositions = model_detected_presuppositions
        self.few_shot_data = []
        for dp in few_shot_data:
            self.few_shot_data += FeedbackActionFewShotExample(**dp).generate()

    def generate(self, **kwargs) -> List[Dict]:
        messages = [
            {
                "role": self.system_role,
                "content": f"""
                    You are a helpful assistant that provides feedback on the question and a guideline for answering the question.
                    You will be given a question and the assumptions that are implicit in the question.
                    Your task is to first, provide feedback on the question based on whether it contains any false assumptions and then provide a guideline for answering the question.
                """},
            *self.few_shot_data,
            {"role": "user", "content": f"Question: {self.question}\nPresuppositions: {self.model_detected_presuppositions}\n"}
        ]
        return messages
    
class FinalAnswerFewShotExample(Template):
    def __init__(self, question: str, feedback_action: str, answer: str, **kwargs):
        self.question = question
        self.feedback_action = feedback_action
        self.answer = answer
        
    def generate(self, **kwargs) -> Dict[str]:
        return [
            {
                "role": "user",
                "content": f"""
                    Question: {self.question}\n
                    Feedback: {self.feedback_action}\n
                """
            },
            {
                "role": "assistant",
                "content": self.answer
            }
        ]
    
class FinalAnswerTemplate(Template):
    question: str
    model_feedback_action: str
    few_shot_data: List[FinalAnswerFewShotExample]
    
    def __init__(self, question: str, model_feedback_action: str, few_shot_data: List[Dict], system_role: str = "system", **kwargs):
        self.question = question
        self.model_feedback_action = model_feedback_action
        self.system_role = system_role
        self.few_shot_data = []
        for dp in few_shot_data:
            self.few_shot_data += FinalAnswerFewShotExample(**dp).generate()
        
    def generate(self, **kwargs) -> List[Dict]:
        messages = [
            {
                "role": self.system_role,
                "content": f"""
                    You are a helpful assistant that provides a response to a 
                """},
            *self.few_shot_data,
            {"role": "user", "content": f"Question: {self.question}\nPresuppositions: {self.model_detected_presuppositions}\n"}
        ]
        return messages