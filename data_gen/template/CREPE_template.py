from .template import Template
from typing import List
from response import CREPEPresuppositionExtractionResponse, CREPEFeedbackActionResponse

class PresuppositionExtractionFewShotExample(Template):
    def __init__(self, question: str, presuppositions: List[str], **kwargs):
        self.question = question
        self.presuppositions = presuppositions

    def generate(self, **kwargs):
        content = f"""
            {CREPEPresuppositionExtractionResponse(presuppositions=self.presuppositions).model_dump_json()}\n
        """
        return content
        
class CREPEPresuppositionExtractionTemplate(Template):
    def __init__(self, question: str, few_shot_data: List[str], **kwargs):
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
                    Format your response as a JSON object with a single key "presuppositions" whose value is a list of presuppositions.
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
        content = f"""
            {CREPEPresuppositionExtractionResponse(presuppositions=presuppositions).model_dump_json()}\n
        """
        if self.raw_corrections != "":
            feedback = f"The question contains a false presupposition that {self.presuppositions[0]}."
            action = f"Correct the false assumptions that {self.raw_corrections} and respond based on the corrected assumption."
        else:
            content += f"The question contains a false presupposition that there is a clear and single answer to the question.\n"
            content += f"Correct the false assumption that there is a clear and single answer to the question and respond based on the corrected assumption.\n"
        content += f"{CREPEFeedbackActionResponse(feedback=feedback, action=action).model_dump_json()}\n"
        return content

class CREPEFeedbackActionTemplate(Template):
    def __init__(self, question: str, model_detected_presuppositions: str, few_shot_data: List[str], **kwargs):
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