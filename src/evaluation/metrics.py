"""
Evaluation metrics: answer recall, F1, ROUGE-L.
Based on SQuAD 6.1 (Rajpurkar et al.).
"""

import re
from typing import List, Optional, Tuple


def normalize_answer(s: str) -> str:
    """Lowercase, remove articles, punctuation, extra whitespace."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def tokenize(s: str) -> List[str]:
    """Whitespace tokenization."""
    return normalize_answer(s).split()


def compute_f1(pred: str, gold: str) -> float:
    """
    Token-level F1: 2 * P * R / (P + R).
    P = |pred_tokens ∩ gold_tokens| / |pred_tokens|
    R = |pred_tokens ∩ gold_tokens| / |gold_tokens|
    """
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = len(set(pred_tokens) & set(gold_tokens))
    if common == 0:
        return 0.0
    p = common / len(pred_tokens)
    r = common / len(gold_tokens)
    return 2 * p * r / (p + r)


def compute_exact_match(pred: str, gold: str) -> float:
    """Exact match (1.0 if normalized pred == normalized gold else 0)."""
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def compute_recall(pred: str, gold: str) -> float:
    """
    Recall: |pred_tokens ∩ gold_tokens| / |gold_tokens|.
    For multiple gold answers, take max over golds.
    """
    pred_tokens = set(tokenize(pred))
    golds = gold if isinstance(gold, list) else [gold]
    best = 0.0
    for g in golds:
        gold_tokens = set(tokenize(g))
        if not gold_tokens:
            best = max(best, 1.0 if not pred_tokens else 0.0)
            continue
        common = len(pred_tokens & gold_tokens)
        best = max(best, common / len(gold_tokens))
    return best


def lcs_length(a: List[str], b: List[str]) -> int:
    """Longest common subsequence length."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(pred: str, gold: str) -> float:
    """
    ROUGE-L F1: based on longest common subsequence.
    R_lcs = LCS(pred, gold) / len(gold)
    P_lcs = LCS(pred, gold) / len(pred)
    F_lcs = 2 * R * P / (R + P)
    """
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    r = lcs / len(gold_tokens)
    p = lcs / len(pred_tokens)
    return 2 * r * p / (r + p)


def evaluate_batch(
    predictions: dict,
    references: dict,
    metrics: Optional[List[str]] = None,
) -> dict:
    """
    Evaluate predictions against references.
    predictions: {id: "answer"}
    references: {id: "answer"} or {id: ["ans1", "ans2"]}
    """
    metrics = metrics or ["f1", "exact_match", "rouge_l"]
    ids = set(predictions.keys()) & set(references.keys())
    scores = {m: [] for m in metrics}
    for qid in ids:
        pred = predictions.get(qid, "")
        ref = references[qid]
        refs = ref if isinstance(ref, list) else [ref]
        for m in metrics:
            if m == "f1":
                scores[m].append(max(compute_f1(pred, r) for r in refs))
            elif m == "exact_match":
                scores[m].append(max(compute_exact_match(pred, r) for r in refs))
            elif m == "rouge_l":
                scores[m].append(max(compute_rouge_l(pred, r) for r in refs))
            elif m == "recall":
                scores[m].append(compute_recall(pred, refs))
    return {m: sum(v) / len(v) if v else 0.0 for m, v in scores.items()}
