"""Compositional binding: Hungarian distinct-region assignment + soft-MIN."""
import numpy as np

from retriever import bind


def _region(vec):
    return {"vector": np.asarray(vec, float), "lab": [50.0, 0.0, 0.0],
            "achromatic": False, "category": "x", "color_name": "x"}


def test_hungarian_assigns_distinct_regions():
    clause_vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
    regions = [_region([1, 0]), _region([0, 1])]
    agg, trace = bind.bind_score(clause_vecs, [None, None], regions)
    assert len({id(t[1]) for t in trace}) == 2   # each clause bound a different region
    assert agg > 0


def test_missing_region_is_penalised():
    clause_vecs = np.array([[1.0, 0.0]])
    present, _ = bind.bind_score(clause_vecs, [None], [_region([1, 0])])
    absent, _ = bind.bind_score(clause_vecs, [None], [])
    assert absent < present


def test_soft_min_pulls_below_mean():
    # scores ~ [1.0, 0.1]; mean 0.55, soft-MIN (alpha .7) should sit well below it
    clause_vecs = np.array([[1.0, 0.0], [1.0, 0.0]])
    regions = [_region([1, 0]), _region([0.1, 0])]
    agg, _ = bind.bind_score(clause_vecs, [None, None], regions)
    assert agg < 0.55
