from typing import List, Iterable, Callable
import evaluate
from bert_score import BERTScorer
from ignite.metrics import RougeL, RougeN

__CACHE__ = {
    "bleurt_models": None,
    "bert_scorer": {}
}

def default_tok(s: str) -> List[str]:
    return s.lower().strip().split()

def f1(prec: float, rec: float, eps: float = 1e-12) -> float:
    return 2 * prec * rec / (prec + rec + eps)

def rouge1_f1(
    candidate: str,
    references: Iterable[str],
    tok: Callable[[str], List[str]] = default_tok
) -> float:
    """
    ROUGE-1 F1 over possibly multiple references (take the best ref).
    """
    m = RougeN(ngram=1, multiref='best')
    c_tokens = tok(candidate)
    m.update(([c_tokens], [tok(r) for r in references]))
    return m.compute()['Rouge-1-F']

def rougeL_f1(
    candidate: str,
    references: Iterable[str],
    tok: Callable[[str], List[str]] = default_tok
) -> float:
    """
    ROUGE-L F1 using Lin (2004) LCS-based precision/recall.
    Multiple refs: take the best ref.
    """
    c_tokens = tok(candidate)
    m = RougeL(multiref='best')
    m.update(([c_tokens], [tok(r) for r in references]))
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
        bert_scorer = BERTScorer(model_type=model_type, lang="en", rescale_with_baseline=True)
        __CACHE__["bert_scorer"][model_type] = bert_scorer
    _, _, F1 = bert_scorer.score(candidates, references)
    return F1.mean()