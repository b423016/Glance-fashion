"""Part B core — two-stage retrieval.

  Stage 1 (recall) : ANN over the global image embedding — sublinear, the layer
                     that scales to 1M images.
  Stage 2 (rerank) : compositional binding + scene/formality, fused with the
                     global score. All component scores are min-max calibrated
                     within the candidate set before weighting, so the fusion
                     weights are meaningful despite different score scales.

``mode="global"`` disables stage 2 (pure fashion-CLIP retrieval) — this is the
ablation baseline the full pipeline is measured against.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fashionlib import config, store
from fashionlib.scene import SceneTagger

from . import bind as bind_mod
from . import parse as parse_mod


def _minmax(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


@dataclass
class Result:
    image_id: int
    file_name: str
    path: str
    score: float
    scene: str
    formality: str
    components: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)


class Retriever:
    def __init__(self, db, enc, tagger=None):
        self.db, self.enc = db, enc
        self.tagger = tagger or SceneTagger(enc)

    def search(self, text, k=None, mode="full", recall_n=None, trace=False):
        k = k or config.TOP_K
        recall_n = recall_n or config.RECALL_N
        q = parse_mod.parse(text)
        qvec = self.enc.encode_text(text)

        if mode == "global":
            return self._global(qvec, k), q

        cand = store.recall(self.db, qvec, recall_n)
        if not cand:
            return [], q
        img_vecs = np.asarray([c["vector"] for c in cand], dtype=np.float32)
        global_s = img_vecs @ qvec

        clause_colors = [c.color for c in q.clauses]
        clause_vecs = (self.enc.encode_texts([c.garment for c in q.clauses])
                       if q.clauses else np.zeros((0, config.EMBED_DIM), np.float32))
        regions_map = store.regions_for(self.db, [c["image_id"] for c in cand])

        bind_s, traces = [], []
        for c in cand:
            agg, tr = bind_mod.bind_score(clause_vecs, clause_colors, regions_map.get(c["image_id"], []))
            bind_s.append(agg)
            traces.append(tr)
        bind_s = np.asarray(bind_s)

        scene_s = self.tagger.score_for(img_vecs, "scene", q.scene) if q.scene else np.zeros(len(cand))
        form_s = self.tagger.score_for(img_vecs, "formality", q.formality) if q.formality else np.zeros(len(cand))

        gN, bN, sN, fN = _minmax(global_s), _minmax(bind_s), _minmax(scene_s), _minmax(form_s)
        w_g = config.W_GLOBAL
        w_b = config.W_BIND if q.clauses else 0.0
        w_s = config.W_SCENE if q.scene else 0.0
        w_f = config.W_FORM if q.formality else 0.0
        final = (w_g * gN + w_b * bN + w_s * sN + w_f * fN) / (w_g + w_b + w_s + w_f)

        order = np.argsort(-final)[:k]
        out = []
        for i in order:
            c = cand[i]
            comp = dict(final=float(final[i]), global_n=float(gN[i]), bind_n=float(bN[i]),
                        scene_n=float(sN[i]), form_n=float(fN[i]),
                        global_raw=float(global_s[i]), bind_raw=float(bind_s[i]))
            out.append(Result(c["image_id"], c["file_name"], c["path"], float(final[i]),
                              c["scene"], c["formality"], comp, self._fmt_trace(traces[i], q) if trace else []))
        return out, q

    def _global(self, qvec, k):
        rows = store.all_images(self.db)
        vecs = np.asarray([r["vector"] for r in rows], dtype=np.float32)
        s = vecs @ qvec
        order = np.argsort(-s)[:k]
        return [Result(rows[i]["image_id"], rows[i]["file_name"], rows[i]["path"], float(s[i]),
                       rows[i]["scene"], rows[i]["formality"], {"global_raw": float(s[i])}, [])
                for i in order]

    @staticmethod
    def _fmt_trace(tr, q):
        out = []
        for ci, reg, sc in tr:
            clause = q.clauses[ci]
            if reg is None:
                out.append(f"{clause}: UNMATCHED")
            else:
                out.append(f"{clause} -> {reg['category']}({reg['color_name']}) {sc:.3f}")
        return out
