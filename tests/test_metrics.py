"""Rank-aware retrieval metrics."""
from evaluation.metrics import metrics


def test_perfect_ranking():
    m = metrics([1] * 8, support=8)
    assert m["p8"] == 1.0
    assert m["ndcg10"] > 0.99
    assert m["ap20"] > 0.99


def test_empty_ranking():
    m = metrics([0] * 8, support=5)
    assert m["p8"] == 0.0
    assert m["ap20"] == 0.0


def test_rank_awareness():
    front = metrics([1, 1, 0, 0, 0, 0, 0, 0], support=2)
    back = metrics([0, 0, 0, 0, 0, 0, 1, 1], support=2)
    assert front["ndcg10"] > back["ndcg10"]   # relevant-early ranks higher
