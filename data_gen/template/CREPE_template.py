from .template import Template
from typing import List

class PresuppositionExtractionFewShotExample(Template):
    def __init__(self, question: str, presuppositions: List[str], **kwargs):
        self.question = question
        self.presuppositions = presuppositions

    def generate(self, **kwargs):
        content = f"""
            Question: {self.question}
            Presuppositions:
                
        """
        for i, p in enumerate(self.presuppositions, 1):
            content += f"    ({i}) {p}\n"
        return content
        
class CREPEPresuppositionExtractionTemplate(Template):
    def __init__(self, question: str, few_shot_data: List[str], **kwargs):
        self.question = question
        self.few_shot_data = [PresuppositionExtractionFewShotExample(**dp) for dp in few_shot_data]

    def generate(self, system_role: str = "system", **kwargs):
        few_shot_content = "\n".join([dp.generate() for dp in self.few_shot_data])
        messages = [
            {
                "role": system_role,
                "content": f"""
                    You are a helpful assistant that analyzes the following question.
                    Your task is to extract assumptions implicit in a given question.
                    You must notice that considering the intention of the question will be helpful to extract a hidden assumption of the given question.
                    {few_shot_content}
                """},
            {"role": "user", "content": self.question}
        ]
        return messages
    
class FeedbackActionFewShotExample(Template):
    def __init__(self, question: str, presuppositions: str, raw_corrections: str, **kwargs):
        self.question = question
        self.presuppositions = presuppositions
        self.raw_corrections = "; ".join(raw_corrections)

    def generate(self, **kwargs):
        content = f"""
            Question: {self.question}
            Presuppositions: 
        """
        for i, p in enumerate(self.presuppositions, 1):
            content += f"    ({i}) {p}\n"
        content += f"   ({i}) There is a clear and single answer to the question.\n"
        if self.raw_corrections != "":
            content += f"    Feedback: The question contains a false presupposition that {self.presuppositions[0]}.\n"
            content += f"    Action: correct the false assumption that {self.presuppositions[0]} and respond based on the corrected assumption.\n"
        else:
            content += f"    Feedback: The question contains a false presupposition that there is a clear and single answer to the question.\n"
            content += f"    Action: correct the false assumption that there is a clear and single answer to the question and respond based on the corrected assumption.\n"
        return content

class CREPEFeedbackActionTemplate(Template):
    def __init__(self, question: str, model_detected_presuppositions: str, few_shot_data: List[str], **kwargs):
        self.question = question
        self.model_detected_presuppositions = model_detected_presuppositions
        self.few_shot_data = [FeedbackActionFewShotExample(**dp) for dp in few_shot_data]

    def generate(self, system_role: str = "system", **kwargs):
        few_shot_content = "\n".join([dp.generate() for dp in self.few_shot_data])
        messages = [
            {
                "role": system_role,
                "content": f"""
                    You are a helpful assistant that provides feedback on the question and a guideline for answering the question.
                    You will be given a question and the assumptions that are implicit in the question.
                    Your task is to first, provide feedback on the question based on whether it contains any false assumptions nad then provide a guideline for answering the question.
                    {few_shot_content}
                """},
            {"role": "user", "content": f"Question: {self.question}\nPresuppositions: {self.model_detected_presuppositions}\n"}
        ]
        return messages