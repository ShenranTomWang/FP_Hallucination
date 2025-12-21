from abc import ABC, abstractmethod
from typing import List

class Template(ABC):
    @abstractmethod
    def generate(self, **kwargs):
        pass
    
class DirectFPQATemplate(Template):
    def __init__(self, question: str, **kwargs):
        self.question = question
        
    def generate(self, system_role: str = "system", **kwargs):
        messages = [
            {
                "role": system_role,
                "content": f"""
                    You are a helpful assistant that answer questions based on your knowledge.
                    The user will ask a question, and you need to provide the answer to that question.
                """
            },
            {"role": "user", "content": self.question}
        ]
        return messages
    
class PresuppositionExtractionFewShotExample(Template):
    def __init__(self, question: str, presuppositions: List[str], **kwargs):
        self.question = question
        self.presuppositions = presuppositions

    def generate(self, **kwargs):
        content = "\n".join(self.presuppositions) + "\n"
        return content

class PresuppositionExtractionTemplate(ABC, Template):
    question: str
    few_shot_data: List[PresuppositionExtractionFewShotExample]
    
    def __init__(self, question: str, few_shot_data: List[dict], **kwargs):
        self.question = question
        self.few_shot_data = [PresuppositionExtractionFewShotExample(**dp) for dp in few_shot_data]
    
    def generate(self, system_role: str = "system", **kwargs):
        few_shot = []
        for dp in self.few_shot_data:
            few_shot.append({"role": "user", "content": dp.question})
            few_shot.append({"role": "assistant", "content": dp.generate()})
        messages = [
            {
                "role": system_role,
                "content": f"""
                    You are a helpful assistant that analyzes the given question.
                    Your task is to extract presuppositions in the given question.
                    Notice that the presuppositions in a question could be true or false, and may be explicit or implicit.
                    There could be multiple presuppositions in a question, but there will always be at least one presupposition in the question.
                    Format your response as a list of presuppositions, separated by newlines.
                """
            },
            *few_shot,
            {"role": "user", "content": self.question}
        ]
        return messages

class FeedbackActionFewShotExample(Template):
    def __init__(self, question: str, presuppositions: List[str], raw_corrections: str, **kwargs):
        self.question = question
        self.presuppositions = presuppositions
        self.raw_corrections = "; ".join(raw_corrections)

    def generate(self, **kwargs):
        presuppositions = self.presuppositions
        presuppositions.append("There is a clear and single answer to the question.")
        content = "\n".join(presuppositions) + "\n"
        feedback = f"The question contains false presuppositions that {self.presuppositions}."
        action = f"Correct the false assumptions that {self.presuppositions} and respond based on the corrected assumption."
        content += f"Feedback: {feedback}\nAction: {action}\n"
        return content

class FeedbackActionTemplate(Template):
    question: str
    model_detected_presuppositions: List[str]
    few_shot_data: List[FeedbackActionFewShotExample]
    
    def __init__(self, question: str, model_detected_presuppositions: str, few_shot_data: List[dict], **kwargs):
        self.question = question
        self.model_detected_presuppositions = model_detected_presuppositions
        self.few_shot_data = [FeedbackActionFewShotExample(**dp) for dp in few_shot_data]

    def generate(self, system_role: str = "system", **kwargs):
        few_shot = []
        for dp in self.few_shot_data:
            few_shot.append({"role": "user", "content": dp.question})
            few_shot.append({"role": "assistant", "content": dp.generate()})
        messages = [
            {
                "role": system_role,
                "content": f"""
                    You are a helpful assistant that provides feedback on the question and a guideline for answering the question.
                    You will be given a question and the assumptions that are implicit in the question.
                    Your task is to first, provide feedback on the question based on whether it contains any false assumptions nad then provide a guideline for answering the question.
                """},
            *few_shot,
            {"role": "user", "content": f"Question: {self.question}\nPresuppositions: {self.model_detected_presuppositions}\n"}
        ]
        return messages