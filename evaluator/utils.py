from typing import List, Iterable, Dict, Type
import evaluate
from bert_score import BERTScorer
from ignite.metrics import RougeL, RougeN
from google import genai
from google.genai import types
from data_gen.template import CREPEFPScoreEntailmentCountingTemplate, CREPEFPScorePresuppositionExtractionTemplate
from response import Response, CREPEPresuppositionExtractionResponse, CREPEEntailmentCountingResponse

__CACHE__ = {
    "bleurt_models": None,
    "bert_scorer": {}
}

def _parse_response_gemini(response_cls: Type[Response], response: Dict | str) -> Dict:
    if not isinstance(response, str):
        response = response['response']['text']
    return response_cls.model_validate_plain_text(response).model_dump()

def f1(prec: float, rec: float, eps: float = 1e-12) -> float:
    return 2 * prec * rec / (prec + rec + eps)

def rouge1_f1(
    candidate: str,
    references: Iterable[str],
) -> float:
    """
    ROUGE-1 F1 over possibly multiple references (take the best ref).
    """
    m = RougeN(ngram=1, multiref='best', alpha=0.5)
    m.update(([candidate.split()], [[r.split() for r in references]]))
    return m.compute()['Rouge-1-F']

def rougeL_f1(
    candidate: str,
    references: Iterable[str]
) -> float:
    """
    ROUGE-L F1 using Lin (2004) LCS-based precision/recall.
    Multiple refs: take the best ref.
    """
    m = RougeL(multiref='best', alpha=0.5)
    m.update(([candidate.split()], [[r.split() for r in references]]))
    return m.compute()['Rouge-L-F']

def bleurt_score(
    candidates: Iterable[str],
    references: Iterable[str],
    model_name: str = "Elron/bleurt-large-512",
    reduction: str = "max"
) -> float:
    """
    Compute BLEURT for candidate vs multiple references.
    reduction: "max" or "mean"
    """
    bleurt = __CACHE__.get("bleurt_models")
    if bleurt is None:
        __CACHE__["bleurt_models"] = evaluate.load("bleurt", module_type="metric", config_name=model_name)
        bleurt = __CACHE__["bleurt_models"]
    scores = []
    for ref in references:
        _scores = []
        for candidate in candidates:
            res = bleurt.compute(predictions=[candidate], references=[ref], return_details=True)
            _scores.append(res["scores"][0])
        scores.append(max(_scores))
    if reduction == "mean":
        return sum(scores) / len(scores)
    else:
        return max(scores)

def bert_score_f1(
    candidates: List[str],
    references: List[str],
    model_type: str = None
) -> float:
    """
    Compute BERTScore F1 for lists of candidates and references.
    """
    bert_scorer = __CACHE__.get("bert_scorer", {}).get(model_type, None)
    if bert_scorer is None:
        bert_scorer = BERTScorer(model_type=model_type, lang="en")
        __CACHE__["bert_scorer"][model_type] = bert_scorer
    _, _, F1 = bert_scorer.score(candidates, references)
    if F1.item() < -1: breakpoint()
    return F1.mean().item()

def fp_score(
    question: str,
    model_final_answer: str,
    presuppositions: List[str],
    few_shot_data: List[Dict],
    system_role: str = "system",
    model_role: str = "assistant",
    user_role: str = "user"
) -> int:
    """
    Compute the percentage of false presuppositions being identified by the model final answer.
    """
    client = __CACHE__.get("genai_client", None)
    if not client:
        client = genai.Client(vertexai=True)
        __CACHE__["genai_client"] = client
    messages = CREPEFPScorePresuppositionExtractionTemplate(
        question=question,
        model_final_answer=model_final_answer,
        few_shot_data=few_shot_data,
        system_role=system_role,
        model_role=model_role,
        user_role=user_role
    ).generate()
    response1 = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[{"role": message["role"], "parts": [{"text": message["content"]}]} for message in messages[1:]],
        config=types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=messages[0]['content']
        )
    )
    presuppositions = _parse_response_gemini(CREPEPresuppositionExtractionResponse, response1)
    messages = CREPEFPScoreEntailmentCountingTemplate(
        answer_extracted_presuppositions=presuppositions['answer_extracted_presuppositions'],
        presuppositions=presuppositions['presuppositions'],
        few_shot_data=few_shot_data,
        system_role=system_role,
        model_role=model_role,
        user_role=user_role
    ).generate()
    response2 = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[{"role": message["role"], "parts": [{"text": message["content"]}]} for message in messages[1:]],
        config=types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=messages[0]['content']
        )
    )
    entailment_counting = _parse_response_gemini(CREPEEntailmentCountingResponse, response2)
    return entailment_counting['count'] / len(presuppositions['presuppositions'])