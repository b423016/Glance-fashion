"""Rank-aware retrieval metrics — pure functions (numpy only, no models).

Kept separate from the evaluation *orchestration* so the metric maths can be unit
tested in isolation. Relevance is binary (an image satisfies the query or not).
"""
from __future__ import annotations

import numpy as np


def precision_at_k(rels: list[int], k: int) -> float:
    return sum(rels[:k]) / k


def ndcg_at_k(rels: list[int], support: int, k: int) -> float:
    """nDCG with an ideal ranking of ``min(support, k)`` relevant items on top."""
    idcg = sum(1 / np.log2(i + 2) for i in range(min(support, k))) or 1.0
    dcg = sum(rels[i] / np.log2(i + 2) for i in range(min(k, len(rels))))
    return dcg / idcg


def average_precision_at_k(rels: list[int], support: int, k: int) -> float:
    hits, running = 0, 0.0
    for i, r in enumerate(rels[:k]):
        if r:
            hits += 1
            running += hits / (i + 1)
    return running / (min(support, k) or 1)


def metrics(rels, support: int, k_p: int = 8, k_ndcg: int = 10, k_ap: int = 20) -> dict:
    """Summary of ``{precision@k_p, nDCG@k_ndcg, AP@k_ap}`` for one ranking."""
    rels = [1 if x else 0 for x in rels]
    return {
        "p8": round(precision_at_k(rels, k_p), 3),
        "ndcg10": round(ndcg_at_k(rels, support, k_ndcg), 3),
        "ap20": round(average_precision_at_k(rels, support, k_ap), 3),
    }
