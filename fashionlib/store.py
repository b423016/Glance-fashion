"""LanceDB wrapper — the only module that talks to the vector store.

We use LanceDB because it is embedded (no server for the demo) yet the exact same
API scales on-disk / on-S3 to ~1B vectors, and it supports vector ANN + metadata
filters in one query — which is precisely the recall + structured-rerank pattern.
"""
from __future__ import annotations

import lancedb
import numpy as np

from . import config
from .schema import ImageRow, RegionRow


def connect():
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(config.INDEX_DIR))


def create_tables(db, image_rows: list[dict], region_rows: list[dict]):
    for name in ("images", "regions"):
        if name in db.table_names():
            db.drop_table(name)
    it = db.create_table("images", schema=ImageRow)
    it.add(image_rows)
    rt = db.create_table("regions", schema=RegionRow)
    rt.add(region_rows)
    return it, rt


def images(db):
    return db.open_table("images")


def regions(db):
    return db.open_table("regions")


def recall(db, query_vec: np.ndarray, n: int) -> list[dict]:
    """Stage-1 ANN recall over the image table (cosine on normalised vectors)."""
    return (images(db).search(query_vec.astype(np.float32))
            .metric("cosine").limit(n).to_list())


def regions_for(db, image_ids: list[int]) -> dict[int, list[dict]]:
    """Fetch region rows for a set of candidate images (metadata filter)."""
    if not image_ids:
        return {}
    id_list = ",".join(str(int(i)) for i in image_ids)
    rows = regions(db).search().where(f"image_id IN ({id_list})").limit(1_000_000).to_list()
    out: dict[int, list[dict]] = {}
    for r in rows:
        out.setdefault(r["image_id"], []).append(r)
    return out


def all_images(db) -> list[dict]:
    """Every image row (used by the global-only baseline in the ablation)."""
    return images(db).search().limit(1_000_000).to_list()


def all_regions(db) -> list[dict]:
    """Every region row (used by the evaluation's relevance judge)."""
    return regions(db).to_arrow().to_pylist()
