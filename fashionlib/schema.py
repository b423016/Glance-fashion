"""LanceDB row schemas — the single definition of what the index stores.

Two tables (this separation is what makes compositional binding possible):
  * ``images``  — one row per image: global embedding + scene/formality tags.
                  Stage-1 ANN recall runs here (the sublinear, 1M-scale layer).
  * ``regions`` — one row per garment instance: region embedding + pixel colour.
                  Stage-2 compositional binding runs over the recalled images'
                  regions only.
"""
from __future__ import annotations

from lancedb.pydantic import LanceModel, Vector

from . import config


class ImageRow(LanceModel):
    image_id: int
    file_name: str
    path: str
    scene: str
    formality: str
    vector: Vector(config.EMBED_DIM)      # global FashionSigLIP image embedding


class RegionRow(LanceModel):
    rid: int
    image_id: int
    category: str
    color_name: str
    achromatic: bool
    lab: list[float]                       # CIELab of dominant colour (len 3)
    bbox: list[float]                      # [x, y, w, h]
    vector: Vector(config.EMBED_DIM)      # region (padded-crop) embedding
