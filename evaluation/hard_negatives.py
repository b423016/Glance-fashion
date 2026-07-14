"""Hard-negative binding test — the rigorous proof that the pipeline *binds*
attributes to objects, not merely filters colours.

For each colour-swap pair (e.g. "black jacket + white top" vs "white jacket +
black top") a bag-of-words model retrieves ~the same images for both — it cannot
tell them apart. A binding model separates them. We report, for each mode, the
**discrimination gap** = P@8(query judged on its OWN spec) − P@8(query judged on
the SWAPPED spec). global ≈ 0 (conflated); full large (bound).

Run (after building the index):  python -m eval.hard_negatives
"""
from __future__ import annotations

import json

from evaluation.run_eval import OUTER, SHIRTS, TOPS, build_judge, satisfies
from fashionlib import config, models, store, viz
from fashionlib.scene import SceneTagger
from retriever.search import Retriever

PAIRS = [
    ("black_jacket_white_top",
     ("a black jacket and a white top", [("black", OUTER), ("white", TOPS)]),
     ("a white jacket and a black top", [("white", OUTER), ("black", TOPS)])),
    ("white_top_black_pants",
     ("a white top and black pants", [("white", TOPS), ("black", {"pants"})]),
     ("a black top and white pants", [("black", TOPS), ("white", {"pants"})])),
    ("red_tie_white_shirt",
     ("a red tie and a white shirt", [("red", {"tie"}), ("white", SHIRTS)]),
     ("a white tie and a red shirt", [("white", {"tie"}), ("red", SHIRTS)])),
]
K = 8


def p_at(res, spec, per_img, k=K):
    rel = [satisfies(per_img.get(x.image_id, []), spec) for x in res[:k]]
    return sum(rel) / k


def main():
    enc = models.Encoder()
    db = store.connect()
    r = Retriever(db, enc, SceneTagger(enc))
    per_img = build_judge(db)

    print(f"{'pair':24s} {'mode':6s} |  own P@8 | swap P@8 | discrimination")
    print("-" * 66)
    rows = []
    for name, (qa, sa), (qb, sb) in PAIRS:
        for mode in ("global", "full"):
            resA, _ = r.search(qa, k=K, mode=mode, recall_n=150)
            resB, _ = r.search(qb, k=K, mode=mode, recall_n=150)
            own = (p_at(resA, sa, per_img) + p_at(resB, sb, per_img)) / 2
            swap = (p_at(resA, sb, per_img) + p_at(resB, sa, per_img)) / 2
            print(f"{name:24s} {mode:6s} |   {own:.2f}   |   {swap:.2f}   |   {own - swap:+.2f}")
            rows.append(dict(pair=name, mode=mode, own=round(own, 3), swap=round(swap, 3),
                             discrimination=round(own - swap, 3)))
            if mode == "full":
                viz.board([(x.path, f"{i}. {x.score:.2f}") for i, x in enumerate(resA, 1)],
                          config.OUTPUT_DIR / f"HN_{name}_A.png", title=f"[FULL] {qa}")
                viz.board([(x.path, f"{i}. {x.score:.2f}") for i, x in enumerate(resB, 1)],
                          config.OUTPUT_DIR / f"HN_{name}_B.png", title=f"[FULL] {qb}")

    (config.OUTPUT_DIR / "hard_negatives.json").write_text(json.dumps(rows, indent=2))
    g = [r["discrimination"] for r in rows if r["mode"] == "global"]
    f = [r["discrimination"] for r in rows if r["mode"] == "full"]
    print("-" * 66)
    print(f"mean discrimination gap:  global {sum(g)/len(g):+.2f}   →   full {sum(f)/len(f):+.2f}")
    print("A near-zero gap for global = it cannot bind (swap looks identical);")
    print("a large gap for full = attributes are correctly bound to their garments.")


if __name__ == "__main__":
    main()
