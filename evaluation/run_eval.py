"""Evaluation — the ablation that answers "is this better than vanilla CLIP?"

Compares two retrieval modes on the same gallery:
  * ``global`` : rank by the global FashionSigLIP embedding only (a strong
                 fashion-CLIP baseline — already beats OpenAI CLIP on fashion).
  * ``full``   : add compositional binding + scene/formality (our pipeline).

Fashionpedia has no query->image relevance labels, so we synthesise a relevance
judge FROM the annotations (GT garment category + our extracted colour) — used
only for *evaluation*, never for retrieval, and identical for both modes.
Reported per query: precision@8, nDCG@10, and Average Precision@20 (rank-aware).

Run (after building the index):  python -m eval.run_eval
"""
from __future__ import annotations

import json

import numpy as np

from evaluation.metrics import metrics
from fashionlib import colors, config, models, store, viz
from fashionlib.scene import SceneTagger
from retriever.search import Retriever

OUTER = {"jacket", "coat"}
TOPS = {"top, t-shirt, sweatshirt", "shirt, blouse", "sweater", "cardigan"}
SHIRTS = {"shirt, blouse", "top, t-shirt, sweatshirt"}

QUERIES = [
    dict(name="Q1_yellow_raincoat", text="A person in a bright yellow raincoat.", judge=[("yellow", OUTER)]),
    dict(name="Q2_office_business", text="Professional business attire inside a modern office.", judge=None),
    dict(name="Q3_blue_shirt_park", text="Someone wearing a blue shirt sitting on a park bench.",
         judge=[("blue", TOPS)]),
    dict(name="Q4_casual_city_walk", text="Casual weekend outfit for a city walk.", judge=None),
    dict(name="Q5_red_tie_white_shirt", text="A red tie and a white shirt in a formal setting.",
         judge=[("red", {"tie"}), ("white", SHIRTS)]),
    dict(name="S1_yellow_jacket", text="a yellow jacket", judge=[("yellow", OUTER)]),
    dict(name="S2_blue_top", text="a blue top", judge=[("blue", TOPS)]),
    dict(name="S3_red_dress", text="a red dress", judge=[("red", {"dress"})]),
    dict(name="S4_white_top_black_pants", text="a white top and black pants",
         judge=[("white", TOPS), ("black", {"pants"})]),
    dict(name="S5_black_jacket_white_top", text="a black jacket and a white top",
         judge=[("black", {"jacket", "coat"}), ("white", TOPS)]),
]

K_BOARD = 12
K_EVAL = 30
RECALL_N = 150


def build_judge(db):
    per_img = {}
    for r in store.all_regions(db):
        per_img.setdefault(r["image_id"], []).append((r["category"], r["color_name"]))
    return per_img


def satisfies(garments, clauses):
    used = set()
    for tcolor, cats in clauses:
        hit = False
        for i, (cat, cname) in enumerate(garments):
            if i in used:
                continue
            if cat in cats and (tcolor is None or colors.family(cname) == colors.family(tcolor)):
                used.add(i)
                hit = True
                break
        if not hit:
            return False
    return True


def support(per_img, clauses):
    return sum(1 for g in per_img.values() if satisfies(g, clauses))


def main():
    enc = models.Encoder()
    db = store.connect()
    r = Retriever(db, enc, SceneTagger(enc))
    per_img = build_judge(db)
    n = len(store.all_images(db))
    print(f"gallery: {n} images\n")
    print(f"{'query':26s} {'sup':>4s} | {'P@8 g→f':>14s} | {'nDCG@10 g→f':>16s} | {'AP@20 g→f':>14s}")

    summary, agg = [], {"p8": [[], []], "ndcg10": [[], []], "ap20": [[], []]}
    for spec in QUERIES:
        judge = spec["judge"]
        sup = support(per_img, judge) if judge else None
        row = {"query": spec["name"], "text": spec["text"], "support": sup}
        for mi, mode in enumerate(("global", "full")):
            res, q = r.search(spec["text"], k=K_EVAL, mode=mode, recall_n=RECALL_N, trace=(mode == "full"))
            if judge is not None:
                rels = [satisfies(per_img.get(x.image_id, []), judge) for x in res]
                m = metrics(rels, sup)
                row[mode] = m
                for key in agg:
                    agg[key][mi].append(m[key])
                marks = ["[+] " if x else "" for x in rels]
            else:
                marks = [""] * len(res)
            board = [(x.path, f"{mk}{rank}. {x.score:.2f} {x.scene}")
                     for rank, (x, mk) in enumerate(zip(res[:K_BOARD], marks[:K_BOARD]), 1)]
            viz.board(board, config.OUTPUT_DIR / f"{spec['name']}__{mode}.png",
                      title=f"[{mode.upper()}] {spec['text']}  (support={sup})")
        summary.append(row)
        if judge is not None:
            g, f = row["global"], row["full"]
            print(f"{spec['name']:26s} {sup:>4d} | {g['p8']:.2f} → {f['p8']:.2f}   | "
                  f"{g['ndcg10']:.2f} → {f['ndcg10']:.2f}     | {g['ap20']:.2f} → {f['ap20']:.2f}")
        else:
            print(f"{spec['name']:26s}    - | (scene-only, qualitative board)")

    print("\nMean over judgeable queries (global → full):")
    for key, lab in [("p8", "P@8"), ("ndcg10", "nDCG@10"), ("ap20", "AP@20")]:
        g, f = np.mean(agg[key][0]), np.mean(agg[key][1])
        print(f"  {lab:8s}: {g:.3f} → {f:.3f}  ({(f-g)/max(g,1e-9)*100:+.0f}% rel)")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nboards + eval_summary.json -> {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
