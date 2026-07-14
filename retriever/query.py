"""Part B entrypoint — accept a natural-language query, return the top-k images.

Examples:
  python -m retriever.query "a red tie and a white shirt in a formal setting"
  python -m retriever.query "a bright yellow raincoat" --k 8 --render
  python -m retriever.query "business attire in an office" --mode global
"""
from __future__ import annotations

import argparse

from fashionlib import config, models, store, viz
from fashionlib.scene import SceneTagger
from retriever.search import Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", type=str)
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--mode", choices=["full", "global"], default="full")
    ap.add_argument("--render", action="store_true", help="save a result board PNG")
    args = ap.parse_args()

    enc = models.Encoder()
    db = store.connect()
    r = Retriever(db, enc, SceneTagger(enc))
    results, q = r.search(args.query, k=args.k, mode=args.mode, trace=True)

    print(f'\nQuery: "{args.query}"   [{args.mode}]')
    print(f"Parsed: {q.summary()}\n")
    for rank, res in enumerate(results, 1):
        print(f"{rank:2d}. {res.score:.3f}  {res.file_name[:14]}  scene={res.scene:7s} form={res.formality}")
        for line in res.trace:
            print(f"        · {line}")

    if args.render:
        items = [(res.path, f"{rank}. {res.score:.2f} {res.scene}")
                 for rank, res in enumerate(results, 1)]
        out = config.OUTPUT_DIR / f"query_{args.mode}.png"
        viz.board(items, out, title=f'{args.query}  [{args.mode}]')
        print(f"\nboard -> {out}")


if __name__ == "__main__":
    main()
