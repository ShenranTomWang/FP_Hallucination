from .template import PresuppositionExtractionTemplate, FeedbackActionTemplate, FinalAnswerTemplate, MiniCheckFinalAnswerTemplate, DirectQATemplate, FPScorePresuppositionExtractionTemplate, FPScoreEntailmentCountingTemplate
from typing import List, Dict
        
class CREPEPresuppositionExtractionTemplate(PresuppositionExtractionTemplate):
    def __init__(
        self,
        question: str,
        few_shot_data: List[Dict],
        system_role: str = "system",
        model_role: str = "assistant",
        user_role: str = "user",
        **kwargs
    ):
        super().__init__(
            question=question,
            few_shot_data=few_shot_data,
            system_role=system_role,
            model_role=model_role,
            user_role=user_role
        )

class CREPEFeedbackActionTemplate(FeedbackActionTemplate):
    def __init__(
        self,
        question: str,
        model_detected_presuppositions: Dict[str, str],
        few_shot_data: List[Dict],
        passages: List[str],
        RAG_retrieval: int = 0,
        system_role: str = "system",
        model_role: str = "assistant",
        user_role: str = "user",
        **kwargs
    ):
        few_shot_data = [
            {
                **dp,
                "presuppositions": dp["presuppositions"],
                "is_normal":  "false presupposition" not in dp["labels"]
            } for dp in few_shot_data
        ]
        if RAG_retrieval:
            passages = passages[:RAG_retrieval]
            few_shot_data = [
                {
                    **dp,
                    "passages": dp['passages'][:RAG_retrieval]
                } for dp in few_shot_data
            ]
        
        model_detected_presuppositions = model_detected_presuppositions["presuppositions"]
        super().__init__(
            question=question,
            passages=passages,
            model_detected_presuppositions=model_detected_presuppositions,
            few_shot_data=few_shot_data,
            system_role=system_role,
            model_role=model_role,
            user_role=user_role
        )

class CREPEFinalAnswerTemplate(FinalAnswerTemplate):
    def __init__(
        self,
        question: str,
        few_shot_data: List[Dict],
        model_feedback_action: Dict[str, str],
        system_role: str = "system",
        model_role: str = "assistant",
        user_role: str = "user",
        **kwargs
    ):
        few_shot_data = [
            {
                **dp,
                "answer": dp["comment"],
                "presuppositions": dp["presuppositions"],
                "is_normal": "false presupposition" not in dp["labels"]
            } for dp in few_shot_data
        ]
        model_feedback_action = model_feedback_action["feedback_action"]
        super().__init__(
            question=question,
            model_feedback_action=model_feedback_action,
            few_shot_data=few_shot_data,
            system_role=system_role,
            model_role=model_role,
            user_role=user_role
        )

class CREPEMiniCheckFinalAnswerTemplate(MiniCheckFinalAnswerTemplate):
    def __init__(
        self,
        question: str,
        few_shot_data: List[Dict],
        minicheck_results: List[int],
        presuppositions: List[str],
        system_role: str = "system",
        model_role: str = "assistant",
        user_role: str = "user",
        **kwargs
    ):
        few_shot_data = [
            {
                **dp,
                "answer": dp["comment"],
                "presuppositions": dp["presuppositions"],
                "is_normal": "false presupposition" not in dp["labels"]
            } for dp in few_shot_data
        ]
        super().__init__(
            question=question,
            minicheck_results=minicheck_results,
            presuppositions=presuppositions,
            few_shot_data=few_shot_data,
            system_role=system_role,
            model_role=model_role,
            user_role=user_role
        )

class CREPEDirectQATemplate(DirectQATemplate):
    def __init__(
        self,
        question: str,
        passages: List[str],
        few_shot_data: List[Dict],
        RAG_retrieval: int = 0,
        system_role: str = "system",
        model_role: str = "assistant",
        user_role: str = "user",
        **kwargs
    ):
        few_shot_data = [{**dp, "answer": dp["comment"]} for dp in few_shot_data]
        if RAG_retrieval:
            passages = passages[:RAG_retrieval]
            few_shot_data = [
                {
                    **dp,
                    "passages": dp['passages'][:RAG_retrieval]
                } for dp in few_shot_data
            ]
        super().__init__(
            question=question,
            passages=passages,
            few_shot_data=few_shot_data,
            system_role=system_role,
            model_role=model_role,
            user_role=user_role
        )
        
class CREPEFPScorePresuppositionExtractionTemplate(FPScorePresuppositionExtractionTemplate):
    def __init__(
        self,
        question: str,
        model_final_answer: Dict[str, str],
        few_shot_data: List[Dict],
        system_role: str = "system",
        model_role: str = "assistant",
        user_role: str = "user",
        **kwargs
    ):
        few_shot_data = [{**dp, "answer": dp["comment"]} for dp in few_shot_data if "false presupposition" in dp["labels"]]
        super().__init__(
            question=question,
            model_final_answer=model_final_answer,
            few_shot_data=few_shot_data,
            system_role=system_role,
            model_role=model_role,
            user_role=user_role
        )
        
class CREPEFPScoreEntailmentCountingTemplate(FPScoreEntailmentCountingTemplate):
    def __init__(
        self,
        answer_extracted_presuppositions: Dict[str, str],
        presuppositions: List[str],
        few_shot_data: List[Dict],
        system_role: str = "system",
        model_role: str = "assistant",
        user_role: str = "user",
        **kwargs
    ):
        few_shot_data = [dp for dp in few_shot_data if "false presupposition" in dp["labels"]]
        super().__init__(
            answer_extracted_presuppositions=answer_extracted_presuppositions,
            presuppositions=presuppositions,
            few_shot_data=few_shot_data,
            system_role=system_role,
            model_role=model_role,
            user_role=user_role
        )