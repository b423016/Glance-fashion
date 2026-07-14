"""Compositional binding — the core mechanism that beats vanilla CLIP.

For each parsed clause (e.g. "red tie", "white shirt") we score every garment
region by ``cosine(garment_text, region_embedding) x colour_gate``. Then:

  * **Hungarian assignment** forces each clause onto a *distinct* region, so
    "red tie AND white shirt" cannot both be satisfied by one red garment — this
    is what kills the "red tie ~ red shirt" swap that fools a global embedding.
  * **soft-MIN** across clauses (``alpha*min + (1-alpha)*mean``) requires *all*
    clauses to be present: one unsatisfied clause drags the score down, but a
    single weak region degrades gracefully instead of hard-failing.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from fashionlib import colors, config


def clause_region_matrix(clause_vecs: np.ndarray, clause_colors, regions) -> np.ndarray:
    """(C x R) matrix of clause↔region compatibility = embed-sim x colour-gate."""
    C, R = len(clause_colors), len(regions)
    if R == 0 or C == 0:
        return np.zeros((C, R))
    reg_vecs = np.asarray([r["vector"] for r in regions], dtype=np.float32)
    sims = reg_vecs @ clause_vecs.T                     # (R, C)
    M = np.zeros((C, R))
    for ci, col in enumerate(clause_colors):
        for ri, reg in enumerate(regions):
            gate = 1.0 if not col else colors.color_gate(
                np.asarray(reg["lab"]), bool(reg["achromatic"]), col)
            M[ci, ri] = sims[ri, ci] * gate
    return M


def bind_score(clause_vecs, clause_colors, regions, alpha: float = None):
    """Return ``(aggregate_score, trace)``. ``trace`` is a list of
    ``(clause_idx, matched_region_or_None, score)`` for explainability."""
    alpha = config.BIND_ALPHA if alpha is None else alpha
    C = len(clause_colors)
    if C == 0:
        return 0.0, []
    if len(regions) == 0:
        return -1.0, [(ci, None, 0.0) for ci in range(C)]

    M = clause_region_matrix(clause_vecs, clause_colors, regions)
    rows, cols = linear_sum_assignment(-M)              # maximise; handles C!=R
    assigned = dict(zip(rows.tolist(), cols.tolist()))

    scores, trace = [], []
    for ci in range(C):
        if ci in assigned:
            ri = assigned[ci]
            scores.append(M[ci, ri])
            trace.append((ci, regions[ri], float(M[ci, ri])))
        else:  # more clauses than regions
            scores.append(0.0)
            trace.append((ci, None, 0.0))
    s = np.asarray(scores)
    agg = float(alpha * s.min() + (1 - alpha) * s.mean())
    return agg, trace
