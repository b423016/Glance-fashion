"""Part A — The Indexer.

Processes Fashionpedia val images into a searchable LanceDB index:
  per image  -> global embedding + scene/formality tags        (``images`` table)
  per garment-> padded-crop embedding + CIELab dominant colour  (``regions`` table)

Run:  python -m indexer.build_index               # full val gallery (~1,158)
      python -m indexer.build_index --limit 60    # quick smoke build

At 1M scale the ONLY change is swapping the Fashionpedia GT masks for an
open-vocab detector (Grounding DINO / YOLO-World) in ``iter_garments`` — every
downstream step is identical.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from PIL import Image

from fashionlib import config, data, models, regions, store
from fashionlib.scene import SceneTagger


def _embed_images(enc, paths, batch=32):
    out = []
    for i in range(0, len(paths), batch):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        out.append(enc.encode_images(ims, batch_size=batch))
        for im in ims:
            im.close()
    return np.concatenate(out, 0)


def _embed_regions(enc, recs, chunk=256):
    """Stream garment crops through the encoder in chunks (bounded memory).
    Returns list of (image_id, garment, color_dict, embedding)."""
    rows, buf_imgs, buf_meta = [], [], []

    def flush():
        if not buf_imgs:
            return
        emb = enc.encode_images(buf_imgs, batch_size=32)
        for e, (iid, g, col) in zip(emb, buf_meta):
            rows.append((iid, g, col, e))
        for im in buf_imgs:
            im.close()
        buf_imgs.clear()
        buf_meta.clear()

    for r in recs:
        im = Image.open(r["path"]).convert("RGB")
        for g in r["garments"]:
            buf_imgs.append(regions.region_crop(im, g))
            buf_meta.append((r["image_id"], g, regions.dominant_color(im, g)))
            if len(buf_imgs) >= chunk:
                flush()
        im.close()
    flush()
    return rows


def build(limit=None, batch=32, supplement=False, n_random=1500):
    t0 = time.time()
    recs, cats = data.load_val(limit=limit)
    if supplement:
        sup, n_tie = data.load_supplement(n_random=n_random)
        recs = recs + sup
        print(f"[index] + train supplement: {len(sup)} imgs ({n_tie} contain a tie)")
    print(f"[index] {len(recs)} images, {sum(len(r['garments']) for r in recs)} garments")

    enc = models.Encoder()
    tagger = SceneTagger(enc)

    print("[index] embedding full images + tagging scenes ...")
    img_emb = _embed_images(enc, [r["path"] for r in recs], batch)
    tags = tagger.tag(img_emb)

    image_rows = [{
        "image_id": r["image_id"], "file_name": r["file_name"], "path": r["path"],
        "scene": tags["scene"][i], "formality": tags["formality"][i],
        "vector": img_emb[i].tolist(),
    } for i, r in enumerate(recs)]

    print("[index] embedding garment regions + extracting colours ...")
    reg = _embed_regions(enc, recs)
    region_rows, rid = [], 0
    for iid, g, col, emb in reg:
        if col is None:
            continue
        region_rows.append({
            "rid": rid, "image_id": iid, "category": g["category"],
            "color_name": col["name"], "achromatic": col["achromatic"],
            "lab": col["lab"], "bbox": [float(v) for v in g["bbox"]],
            "vector": emb.tolist(),
        })
        rid += 1

    print(f"[index] writing {len(image_rows)} image rows + {len(region_rows)} region rows to LanceDB ...")
    db = store.connect()
    store.create_tables(db, image_rows, region_rows)
    print(f"[index] DONE in {time.time()-t0:.0f}s -> {config.INDEX_DIR}")


def cli():
    ap = argparse.ArgumentParser(description="Build the fashion retrieval index (Part A).")
    ap.add_argument("--limit", type=int, default=None, help="index only the first N val images")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--supplement", action="store_true", help="add the Fashionpedia train supplement")
    ap.add_argument("--n-random", type=int, default=1500, dest="n_random",
                    help="size of the random train sample added alongside all tie images")
    build(**vars(ap.parse_args()))


if __name__ == "__main__":
    cli()
