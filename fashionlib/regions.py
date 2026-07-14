"""Turn a Fashionpedia garment annotation into two things, from two different
uses of the mask geometry (this split is the key correction from the bake-off):

  * ``region_crop`` — a PADDED bbox crop *with surrounding context pixels* for
    embedding. NOT a background-zeroed cutout: the SigLIP encoder was trained on
    natural images, so a masked cutout is out-of-distribution and hurts.
  * ``dominant_color`` — colour from the ERODED exact mask in CIELab (erosion
    kills skin/background bleed at the garment edge).

Segmentation is polygon for ~96% of instances (rasterised with PIL); the ~4% RLE
instances fall back to the bbox — no ``pycocotools`` needed.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw
from skimage.color import rgb2lab
from sklearn.cluster import KMeans

from . import colors, config

REGION_PAD = 0.12  # fraction of bbox size added as context on each side


def polygon_mask(seg, w: int, h: int) -> np.ndarray:
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for poly in seg:
        if len(poly) >= 6:
            d.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)], fill=1)
    return np.asarray(m, dtype=bool)


def mask_or_none(seg, w: int, h: int):
    return polygon_mask(seg, w, h) if isinstance(seg, list) else None


def region_crop(img: Image.Image, garment, pad: float = REGION_PAD) -> Image.Image:
    """Padded bbox crop with natural context (in-distribution for the encoder)."""
    w, h = img.size
    x, y, bw, bh = garment["bbox"]
    px, py = bw * pad, bh * pad
    box = (max(0, int(x - px)), max(0, int(y - py)),
           min(w, int(x + bw + px)), min(h, int(y + bh + py)))
    return img.crop(box).convert("RGB")


def dominant_color(img: Image.Image, garment, k: int | None = None, erode: int = 2):
    """Return ``{lab, rgb, name, achromatic}`` for the garment's dominant colour,
    from k-means over eroded-mask pixels in CIELab. ``None`` if too few pixels."""
    k = k or config.COLOR_K
    w, h = img.size
    arr = np.asarray(img.convert("RGB"))
    mask = mask_or_none(garment["segmentation"], w, h)
    if mask is not None and mask.sum() > 60:
        if erode > 0:
            er = ndi.binary_erosion(mask, iterations=erode)
            if er.sum() > 40:
                mask = er
        pix = arr[mask]
    else:
        x, y, bw, bh = (int(round(v)) for v in garment["bbox"])
        pix = arr[y:y + bh, x:x + bw].reshape(-1, 3)
    if len(pix) < 10:
        return None
    rng = np.random.default_rng(0)
    if len(pix) > 4000:
        pix = pix[rng.choice(len(pix), 4000, replace=False)]
    lab_pix = rgb2lab(pix.reshape(-1, 1, 3).astype(float) / 255.0).reshape(-1, 3)
    km = KMeans(n_clusters=min(k, len(lab_pix)), n_init=4, random_state=0).fit(lab_pix)
    counts = np.bincount(km.labels_, minlength=km.n_clusters)
    lab = km.cluster_centers_[int(counts.argmax())]
    return {
        "lab": [float(v) for v in lab],
        "rgb": list(colors.lab_to_rgb(lab)),
        "name": colors.name_from_lab(lab),
        "achromatic": bool(colors.is_achromatic(lab)),
    }
