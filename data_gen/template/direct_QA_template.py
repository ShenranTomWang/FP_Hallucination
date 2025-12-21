from .template import Template

class DirectQATemplate(Template):
    def __init__(self, question: str, system_role: str = "system" **kwargs):
        self.question = question
        self.system_role = system_role
        
    def generate(self, **kwargs):
        messages = [
            {
                "role": self.system_role,
                "content": f"""
                    You are a helpful assistant that answer questions based on your knowledge.
                    The user will ask a question, and you need to provide the answer to that question.
                """
            },
            {"role": "user", "content": self.question}
        ]
        return messages