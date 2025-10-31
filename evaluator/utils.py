from typing import List, Iterable, Tuple, Callable
from collections import Counter

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
    c_tokens = tok(candidate)
    c_counts = Counter(c_tokens)

    def rouge1_pair(ref: str) -> float:
        r_tokens = tok(ref)
        r_counts = Counter(r_tokens)
        overlap = sum((c_counts & r_counts).values())
        prec = overlap / max(len(c_tokens), 1)
        rec  = overlap / max(len(r_tokens), 1)
        return f1(prec, rec)

    return max(rouge1_pair(r) for r in references)

def _lcs_len(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = tmp
    return dp[m]

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

    def rougeL_pair(ref: str) -> float:
        r_tokens = tok(ref)
        lcs = _lcs_len(c_tokens, r_tokens)
        prec = lcs / max(len(c_tokens), 1)
        rec  = lcs / max(len(r_tokens), 1)
        return f1(prec, rec)

    return max(rougeL_pair(r) for r in references)

def bleurt_score(
    candidate: str,
    references: Iterable[str],
    model_name: str = "Elron/bleurt-large-512",
    reduction: str = "max"
) -> float:
    """
    Compute BLEURT for candidate vs multiple references.
    reduction: "max" or "mean"
    """
    try:
        import evaluate
    except ImportError:
        raise RuntimeError(
            "Please install: pip install evaluate transformers tensorflow"
        )
    bleurt = evaluate.load("bleurt", module_type="metric", config_name=model_name)
    scores = []
    for ref in references:
        res = bleurt.compute(predictions=[candidate], references=[ref])
        scores.append(res["scores"][0])
    if reduction == "mean":
        return sum(scores) / len(scores)
    return max(scores)