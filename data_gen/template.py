from abc import ABC, abstractmethod

class Template(ABC):
    @abstractmethod
    def generate(self):
        pass

class PresuppositionExtractionTemplate(Template):
    def __init__(self, question: str, **kwargs):
        self.question = question

    def generate(self):
        messages = [
            {
                "role": "system",
                "content": """
                    You are a helpful assistant that analyzes the following question. \
                    Your task is to extract assumptions implicit in a given question. \
                    You must notice that considering the intention of the question will be helpful to extract a hidden assumption of the given question.
                    e.g. Question : When did the great depression began before world war 1?
                        Presuppositions :
                        (1) There was a period called the Great Depression.
                        (2) There was a conflict called World War 1.
                        (3) The Great Depression began before World War 1.
                    There is a clear and single answer to the question.
                """},
            {"role": "user", "content": self.question}
        ]
        return messages
