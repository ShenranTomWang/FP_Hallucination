from abc import ABC, abstractmethod

class Template(ABC):
    @abstractmethod
    def generate(self):
        pass

class FewShotExample(Template):
    def __init__(self, question: str, presuppositions: list, **kwargs):
        self.question = question
        self.presuppositions = presuppositions

    def generate(self):
        content = f"""
            Question: {self.question}
            Presuppositions:
                
        """
        for i, p in enumerate(self.presuppositions, 1):
            content += f"    ({i}) {p}\n"
        return content
        
class PresuppositionExtractionTemplate(Template):
    def __init__(self, question: str, few_shot_data: list, **kwargs):
        self.question = question
        self.few_shot_data = [FewShotExample(**dp) for dp in few_shot_data]

    def generate(self):
        few_shot_content = "\n".join([dp.generate() for dp in self.few_shot_data])
        messages = [
            {
                "role": "system",
                "content": f"""
                    You are a helpful assistant that analyzes the following question.
                    Your task is to extract assumptions implicit in a given question.
                    You must notice that considering the intention of the question will be helpful to extract a hidden assumption of the given question.
                    {few_shot_content}
                """},
            {"role": "user", "content": self.question}
        ]
        return messages
