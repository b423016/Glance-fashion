"""Fashionpedia loaders.

The searchable gallery is the **val** split (1,158 images with public masks +
attributes), optionally augmented with a targeted **train** supplement so the
evaluation queries have real positives (train adds ~1,455 tie images for the
"red tie + white shirt" query, plus environment variety). Both splits are the
same dataset with the same COCO-format masks, so the pipeline is identical — the
masks are used only as *geometry*; every scored signal is computed fresh, so
retrieval stays zero-shot.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import config


def _records(ann: dict, img_dir: Path, keep_ids=None, names=None, limit=None, id_offset=0):
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    by_img: dict[int, list] = defaultdict(list)
    for a in ann["annotations"]:
        by_img[a["image_id"]].append(a)
    nameset = set(names) if names else None
    out = []
    for im in ann["images"]:
        if keep_ids is not None and im["id"] not in keep_ids:
            continue
        if nameset is not None and im["file_name"] not in nameset:
            continue
        garments = [{
            "category_id": a["category_id"],
            "category": cats[a["category_id"]],
            "bbox": a["bbox"],
            "area": a["area"],
            "segmentation": a["segmentation"],
        } for a in by_img[im["id"]] if a["area"] >= config.MIN_AREA]
        out.append({
            "image_id": im["id"] + id_offset,   # offset keeps train ids distinct from val
            "file_name": im["file_name"],
            "path": str(img_dir / im["file_name"]),
            "width": im["width"],
            "height": im["height"],
            "garments": garments,
            "split": "train" if img_dir == config.TRAIN_IMAGES_DIR else "val",
        })
        if limit and len(out) >= limit:
            break
    return out, cats


def load_val(limit: int | None = None):
    ann = json.loads(config.VAL_ANN.read_text())
    return _records(ann, config.IMAGES_DIR, limit=limit)


def load_supplement(n_random: int = 1500, seed: int = 0):
    """Targeted train supplement: every tie image (for the "red tie + white shirt"
    query) + a deterministic random sample for environment/attribute variety.
    Returns records only (categories are identical to val)."""
    ann = json.loads(config.TRAIN_ANN.read_text())
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    tie_id = next(i for i, n in cats.items() if n == "tie")
    by_img: dict[int, list] = defaultdict(list)
    for a in ann["annotations"]:
        by_img[a["image_id"]].append(a)
    tie_ids = [iid for iid, anns in by_img.items() if any(a["category_id"] == tie_id for a in anns)]
    rng = np.random.default_rng(seed)
    others = [iid for iid in by_img if iid not in set(tie_ids)]
    rand_ids = rng.choice(others, size=min(n_random, len(others)), replace=False).tolist()
    keep = set(tie_ids) | set(int(i) for i in rand_ids)
    recs, _ = _records(ann, config.TRAIN_IMAGES_DIR, keep_ids=keep, id_offset=10_000_000)
    return recs, len(tie_ids)
