from abc import ABC, abstractmethod
from typing import List, Dict

class Template(ABC):
    @abstractmethod
    def generate(self, **kwargs) -> str | List[Dict]:
        pass
    
class PresuppositionExtractionFewShotExample(Template):
    question: str
    presuppositions: List[str]
    user_role: str
    model_role: str
    
    def __init__(self, question: str, presuppositions: List[str], user_role: str = "user", model_role: str = "assistant", **kwargs):
        self.question = question
        self.presuppositions = presuppositions
        self.user_role = user_role
        self.model_role = model_role

    def generate(self, **kwargs) -> List[Dict]:
        content = "\n".join(self.presuppositions) + "\n"
        return [
            {
                "role": self.user_role,
                "content": self.question
            },
            {
                "role": self.model_role,
                "content": content
            }
        ]

class PresuppositionExtractionTemplate(Template):
    question: str
    few_shot_data: List[PresuppositionExtractionFewShotExample]
    system_role: str
    model_role: str
    user_role: str
    
    def __init__(self, question: str, few_shot_data: List[Dict], system_role: str = "system", model_role: str = "assistant", user_role: str = "user", **kwargs):
        self.question = question
        self.system_role = system_role
        self.model_role = model_role
        self.user_role = user_role
        self.few_shot_data = []
        for dp in few_shot_data:
            self.few_shot_data += PresuppositionExtractionFewShotExample(**dp, user_role=self.user_role, model_role=self.model_role).generate()
    
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
            {
                "role": self.user_role,
                "content": self.question
            }
        ]
        return messages

class FeedbackActionFewShotExample(Template):
    question: str
    presuppositions: List[str]
    raw_corrections: str
    user_role: str
    model_role: str
    
    def __init__(self, question: str, presuppositions: List[str], raw_corrections: str, user_role: str = "user", model_role: str = "assistant", **kwargs):
        self.question = question
        self.presuppositions = presuppositions
        self.raw_corrections = "; ".join(raw_corrections)
        self.user_role = user_role
        self.model_role = model_role

    def generate(self, **kwargs) -> List[Dict]:
        presuppositions = self.presuppositions
        presuppositions.append("There is a clear and single answer to the question.")
        content = "\n".join(presuppositions) + "\n"
        feedback = f"The question contains false presuppositions that {self.presuppositions}."
        action = f"Correct the false assumptions that {self.presuppositions} and respond based on the corrected assumption."
        content += f"Feedback: {feedback}\nAction: {action}\n"
        return [
            {
                "role": self.user_role,
                "content": self.question
            },
            {
                "role": self.model_role,
                "content": content
            }
        ]

class FeedbackActionTemplate(Template):
    question: str
    model_detected_presuppositions: List[str]
    few_shot_data: List[FeedbackActionFewShotExample]
    system_role: str
    user_role: str
    model_role: str
    
    def __init__(self, question: str, model_detected_presuppositions: str, few_shot_data: List[Dict], system_role: str = "system", user_role: str = "user", model_role: str = "assistant", **kwargs):
        self.question = question
        self.system_role = system_role
        self.user_role = user_role
        self.model_role = model_role
        self.model_detected_presuppositions = model_detected_presuppositions
        self.few_shot_data = []
        for dp in few_shot_data:
            self.few_shot_data += FeedbackActionFewShotExample(**dp, user_role=self.user_role, model_role=self.model_role).generate()

    def generate(self, **kwargs) -> List[Dict]:
        messages = [
            {
                "role": self.system_role,
                "content": f"""
                    You are a helpful assistant that provides feedback on the question and a guideline for answering the question.
                    You will be given a question and the assumptions that are implicit in the question.
                    Your task is to first, provide feedback on the question based on whether it contains any false assumptions and then provide a guideline for answering the question.
                """
            },
            *self.few_shot_data,
            {
                "role": self.user_role,
                "content": f"Question: {self.question}\nPresuppositions: {self.model_detected_presuppositions}\n"
            }
        ]
        return messages
    
class FinalAnswerFewShotExample(Template):
    question: str
    feedback_action: str
    answer: str
    user_role: str
    model_role: str
    
    def __init__(self, question: str, feedback_action: str, answer: str, user_role: str = "user", model_role: str = "assistant", **kwargs):
        self.question = question
        self.feedback_action = feedback_action
        self.answer = answer
        self.user_role = user_role
        self.model_role = model_role
        
    def generate(self, **kwargs) -> List[Dict]:
        return [
            {
                "role": self.user_role,
                "content": f"""
                    Question: {self.question}\n
                    Feedback: {self.feedback_action}\n
                """
            },
            {
                "role": self.model_role,
                "content": self.answer
            }
        ]
    
class FinalAnswerTemplate(Template):
    question: str
    model_feedback_action: str
    few_shot_data: List[FinalAnswerFewShotExample]
    system_role: str
    model_role: str
    user_role: str
    
    def __init__(self, question: str, model_feedback_action: str, few_shot_data: List[Dict], system_role: str = "system", user_role: str = "user", model_role: str = "assistant", **kwargs):
        self.question = question
        self.model_feedback_action = model_feedback_action
        self.system_role = system_role
        self.user_role = user_role
        self.model_role = model_role
        self.few_shot_data = []
        for dp in few_shot_data:
            self.few_shot_data += FinalAnswerFewShotExample(**dp, user_role=self.user_role, model_role=self.model_role).generate()
        
    def generate(self, **kwargs) -> List[Dict]:
        messages = [
            {
                "role": self.system_role,
                "content": f"""
                    You are a helpful assistant that provides a response to a 
                """},
            *self.few_shot_data,
            {
                "role": self.user_role,
                "content": f"Question: {self.question}\nPresuppositions: {self.model_detected_presuppositions}\n"
            }
        ]
        return messages