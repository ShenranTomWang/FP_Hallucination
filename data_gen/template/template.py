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
    passages: List[str]
    presuppositions: List[str]
    user_role: str
    model_role: str
    
    def __init__(self, question: str, passages: List[str], presuppositions: List[str], is_normal: bool, user_role: str = "user", model_role: str = "assistant", **kwargs):
        self.question = question
        self.passages = passages
        self.presuppositions = presuppositions
        self.is_normal = is_normal
        self.user_role = user_role
        self.model_role = model_role

    def generate(self, **kwargs) -> List[Dict]:
        presuppositions = self.presuppositions
        presuppositions.append("There is a clear and single answer to the question.")
        if self.is_normal:
            feedback = "The question is valid and does not contain false presuppositions."
            action = "Answer the question directly based on the presuppositions."
        else:
            feedback = f"The question contains false presuppositions that {'; '.join(self.presuppositions)}."
            action = f"Correct the false assumptions that {'; '.join(self.presuppositions)} and respond based on the corrected assumption."
        content = f"Feedback: {feedback}\nAction: {action}"
        user_content = f"Question: {self.question}\nPresuppositions: {'; '.join(self.presuppositions)}"
        if len(self.passages) > 0:
            user_content += f"\nAdditional Information: {' ||| '.join(self.passages)}"
        return [
            {
                "role": self.user_role,
                "content": user_content
            },
            {
                "role": self.model_role,
                "content": content
            }
        ]

class FeedbackActionTemplate(Template):
    question: str
    passages: List[str]
    model_detected_presuppositions: List[str]
    few_shot_data: List[FeedbackActionFewShotExample]
    system_role: str
    user_role: str
    model_role: str
    
    def __init__(self, question: str, passages: List[str], model_detected_presuppositions: List[str], few_shot_data: List[Dict], system_role: str = "system", user_role: str = "user", model_role: str = "assistant", **kwargs):
        self.question = question
        self.passages = passages
        self.system_role = system_role
        self.user_role = user_role
        self.model_role = model_role
        self.model_detected_presuppositions = model_detected_presuppositions
        self.few_shot_data = []
        for dp in few_shot_data:
            self.few_shot_data += FeedbackActionFewShotExample(**dp, user_role=self.user_role, model_role=self.model_role).generate()

    def generate(self, **kwargs) -> List[Dict]:
        user_content = f"Question: {self.question}\nPresuppositions: {'; '.join(self.model_detected_presuppositions)}"
        if len(self.passages) > 0:
            user_content += f"\nAdditional Information: {' ||| '.join(self.passages)}"
        messages = [
            {
                "role": self.system_role,
                "content": f"""
                    You are a helpful assistant that provides feedback on the question and a guideline for answering the question.
                    You will be given a question and the assumptions that are implicit in the question.
                    Your task is to first, provide feedback on the question based on whether it contains any false assumptions and then provide a guideline for answering the question.
                    Separate your feedback and action with a newline, and format your response as:
                    Feedback: <your feedback>\nAction: <your action>
                """
            },
            *self.few_shot_data,
            {
                "role": self.user_role,
                "content": user_content
            }
        ]
        return messages
    
class FinalAnswerFewShotExample(Template):
    question: str
    feedback_action: str
    answer: str
    user_role: str
    model_role: str
    
    def __init__(self, question: str, presuppositions: List[str], is_normal: bool, answer: str, user_role: str = "user", model_role: str = "assistant", **kwargs):
        self.question = question
        self.presuppositions = presuppositions
        self.is_normal = is_normal
        self.answer = answer
        self.user_role = user_role
        self.model_role = model_role
        
    def generate(self, **kwargs) -> List[Dict]:
        if self.is_normal:
            feedback = "The question is valid and does not contain false presuppositions."
            action = "Answer the question directly based on the presuppositions."
        else:
            feedback = f"The question contains false presuppositions that {'; '.join(self.presuppositions)}."
            action = f"Correct the false assumptions that {'; '.join(self.presuppositions)} and respond based on the corrected assumption."
        return [
            {
                "role": self.user_role,
                "content": f"""
                    Question: {self.question}\nFeedback: {feedback}\nAction: {action}
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
                    You are a helpful assistant that provides a response to a question based on the feedback and action guideline.
                    You will be given a question and feedback and action guideline on how to answer the question.
                    Your task is to provide a final answer to the question based on the feedback and action guideline
                """},
            *self.few_shot_data,
            {
                "role": self.user_role,
                "content": f"Question: {self.question}\n{self.model_feedback_action}\n"
            }
        ]
        return messages
    
class MiniCheckFinalAnswerTemplate(Template):
    question: str
    minicheck_results: List[int]
    presuppositions: List[str]
    few_shot_data: List[FinalAnswerFewShotExample]
    system_role: str
    model_role: str
    user_role: str
    
    def __init__(self, question: str, minicheck_results: List[int], presuppositions: List[str], few_shot_data: List[Dict], system_role: str = "system", user_role: str = "user", model_role: str = "assistant", **kwargs):
        self.question = question
        self.minicheck_results = minicheck_results
        self.presuppositions = presuppositions
        self.system_role = system_role
        self.user_role = user_role
        self.model_role = model_role
        self.few_shot_data = []
        for dp in few_shot_data:
            self.few_shot_data += FinalAnswerFewShotExample(**dp, user_role=self.user_role, model_role=self.model_role).generate()
        
    def generate(self, **kwargs) -> List[Dict]:
        false_presuppositions = [self.presuppositions[i] for i, res in enumerate(self.minicheck_results) if res == 0]
        if len(false_presuppositions) == 0:
            feedback_action = "Feedback: The question is valid and does not contain false presuppositions.\nAction: Answer the question directly based on the presuppositions."
        else:
            feedback_action = f"Feedback: The question contains false presuppositions that {'; '.join(false_presuppositions)}.\nAction: Correct the false assumptions that {'; '.join(false_presuppositions)} and respond based on the corrected assumption."
        messages = [
            {
                "role": self.system_role,
                "content": f"""
                    You are a helpful assistant that provides a response to a question based on the feedback and action guideline.
                    You will be given a question and feedback and action guideline on how to answer the question.
                    Your task is to provide a final answer to the question based on the feedback and action guideline
                """},
            *self.few_shot_data,
            {
                "role": self.user_role,
                "content": f"Question: {self.question}\n{feedback_action}\n"
            }
        ]
        return messages
    
class DirectQAFewShotExample(Template):
    question: str
    passages: List[str]
    answer: str
    user_role: str
    model_role: str
    
    def __init__(self, question: str, passages: List[str], answer: str, user_role: str = "user", model_role: str = "assistant", **kwargs):
        self.question = question
        self.passages = passages
        self.answer = answer
        self.user_role = user_role
        self.model_role = model_role
        
    def generate(self, **kwargs) -> List[Dict]:
        user_content = self.question
        if len(self.passages) > 0:
            user_content += f"\nAdditional Information: {' ||| '.join(self.passages)}"
        return [
            {
                "role": self.user_role,
                "content": user_content
            },
            {
                "role": self.model_role,
                "content": self.answer
            }
        ]

class DirectQATemplate(Template):
    def __init__(self, question: str, passages: List[str], few_shot_data: List[Dict], system_role: str = "system", user_role: str = "user", model_role: str = "assistant", **kwargs):
        self.question = question
        self.passages = passages
        self.system_role = system_role
        self.user_role = user_role
        self.model_role = model_role
        self.few_shot_data = []
        for dp in few_shot_data:
            self.few_shot_data += DirectQAFewShotExample(**dp, user_role=self.user_role, model_role=self.model_role).generate()
        
    def generate(self, **kwargs):
        user_content = self.question
        if len(self.passages) > 0:
            user_content += f"\nAdditional Information: {' ||| '.join(self.passages)}"
        messages = [
            {
                "role": self.system_role,
                "content": f"""
                    You are a helpful assistant that answer questions based on your knowledge.
                    The user will ask a question, and you need to provide the answer to that question.
                """
            },
            *self.few_shot_data,
            {"role": self.user_role, "content": user_content}
        ]
        return messages
    
class FPScorePresuppositionExtractionFewShotExample(Template):
    question: str
    answer: str
    presuppositions: List[str]
    user_role: str
    model_role: str
    
    def __init__(self, question: str, answer: str, presuppositions: List[str], user_role: str = "user", model_role: str = "assistant", **kwargs):
        self.question = question
        self.answer = answer
        self.presuppositions = presuppositions
        self.user_role = user_role
        self.model_role = model_role

    def generate(self, **kwargs) -> List[Dict]:
        content = "\n".join(self.presuppositions)
        return [
            {
                "role": self.user_role,
                "content": f"Question: {self.question}\nResponse: {self.answer}"
            },
            {
                "role": self.model_role,
                "content": content
            }
        ]

class FPScorePresuppositionExtractionTemplate(Template):
    def __init__(self, question: str, model_final_answer: str, few_shot_data: List[str], system_role: str = "system", user_role: str = "user", model_role: str = "assistant", **kwargs):
        self.question = question
        self.model_final_answer = model_final_answer
        self.system_role = system_role
        self.user_role = user_role
        self.model_role = model_role
        self.few_shot_data = []
        for dp in few_shot_data:
            self.few_shot_data += FPScorePresuppositionExtractionFewShotExample(**dp, user_role=self.user_role, model_role=self.model_role).generate()

    def generate(self, **kwargs) -> List[Dict]:
        return [
            {
                "role": self.system_role,
                "content": f"""
                    You are a helpful assistant that does the following task:
                    You will be given a question and a response that addresses some false presuppositions in the question.
                    Your task is to extract the presuppositions being addressed in the response.
                    Format your response as a list of presuppositions, separated by newlines.
                """
            },
            *self.few_shot_data,
            {
                "role": self.user_role,
                "content": f"Question: {self.question}\nResponse: {self.model_final_answer}"
            }
        ]

class FPScoreEntailmentCountingFewShotExample(Template):
    presuppositions: List[str]
    user_role: str
    model_role: str
    
    def __init__(self, presuppositions: List[str], user_role: str = "user", model_role: str = "assistant", **kwargs):
        self.presuppositions = presuppositions
        self.user_role = user_role
        self.model_role = model_role

    def generate(self, **kwargs) -> List[Dict]:
        content = f"{len(self.presuppositions)}"
        return [
            {
                "role": self.user_role,
                "content": f"""
                    Presuppositions from the question: {self.presuppositions}
                    Presuppositions from model final answer: {self.presuppositions}
                """
            },
            {
                "role": self.model_role,
                "content": content
            }
        ]

class FPScoreEntailmentCountingTemplate(Template):
    def __init__(self, presuppositions: List[str], answer_extracted_presuppositions: List[str], few_shot_data: List[str], system_role: str = "system", user_role: str = "user", model_role: str = "assistant", **kwargs):
        self.presuppositions = presuppositions
        self.answer_extracted_presuppositions = answer_extracted_presuppositions
        self.system_role = system_role
        self.user_role = user_role
        self.model_role = model_role
        self.few_shot_data = []
        for dp in few_shot_data:
            self.few_shot_data += FPScoreEntailmentCountingFewShotExample(**dp, user_role=self.user_role, model_role=self.model_role).generate()

    def generate(self, **kwargs) -> List[Dict]:
        return [
            {
                "role": self.system_role,
                "content": f"""
                    You are a helpful assistant that does the following task:
                    You will be given two lists of claims:
                    1. A list of presuppositions extracted from a question.
                    2. A list of presuppositions addressed in the model final answer.
                    Your task is to count how many presuppositions from the first list are being addressed in the second list.
                    Format your response as an integer indicating the number of presuppositions being addressed. Do not return any other text.
                """
            },
            *self.few_shot_data,
            {
                "role": self.user_role,
                "content": f"""
                    Presuppositions from the question: {self.presuppositions}
                    Presuppositions from model final answer: {self.answer_extracted_presuppositions}
                """
            }
        ]