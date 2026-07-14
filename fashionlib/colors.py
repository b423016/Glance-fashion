"""CIELab-based colour naming and query→region colour gating.

Colour is derived from *pixels* (never the encoder — VLMs are ~11% exact on
colour). We work in CIELab so distances are perceptual: a "red" query matches a
maroon garment (ΔE≈19) but not a blue one (ΔE≈44), and white/black/gray are
matched by *lightness*, not hue (the "achromatic" path).
"""
from __future__ import annotations

import numpy as np
from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab

# Compact, query-shaped palette (the colour words a fashion query actually uses).
PALETTE = {
    "black": (20, 20, 20), "white": (245, 245, 245), "gray": (128, 128, 128),
    "red": (200, 30, 30), "maroon": (110, 20, 30), "orange": (235, 120, 20),
    "yellow": (240, 215, 40), "gold": (200, 165, 60),
    "green": (40, 150, 60), "olive": (110, 115, 40), "teal": (30, 140, 140),
    "blue": (40, 80, 205), "navy": (25, 35, 90), "light blue": (120, 175, 225),
    "purple": (120, 50, 160), "pink": (235, 130, 175),
    "brown": (120, 75, 40), "beige": (225, 205, 165), "cream": (245, 240, 220),
}
ACHROMATIC = {"black", "white", "gray"}
SYNONYMS = {"grey": "gray", "navy blue": "navy", "sky blue": "light blue", "off-white": "cream"}

# coarse families so a "red" query/label folds "maroon", "blue" folds "navy", etc.
COLOR_FAMILY = {
    "maroon": "red", "gold": "yellow", "navy": "blue", "light blue": "blue",
    "cream": "white", "beige": "brown", "olive": "green", "teal": "blue",
}


def family(name: str | None) -> str | None:
    if not name:
        return None
    return COLOR_FAMILY.get(name, name)


def _lab(rgb) -> np.ndarray:
    return rgb2lab(np.asarray(rgb, float).reshape(1, 1, 3) / 255.0).reshape(3)


PALETTE_LAB = {n: _lab(v) for n, v in PALETTE.items()}
_ACHRO_L = {"black": 8.0, "gray": 54.0, "white": 96.5}


def lab_to_rgb(lab) -> tuple:
    rgb = lab2rgb(np.asarray(lab, float).reshape(1, 1, 3)).reshape(3)
    return tuple(int(round(c * 255)) for c in np.clip(rgb, 0, 1))


def chroma(lab) -> float:
    return float(np.hypot(lab[1], lab[2]))


def is_achromatic(lab, thr: float = 12.0) -> bool:
    return chroma(lab) < thr


def _ciede(a, b) -> float:
    return float(deltaE_ciede2000(np.asarray(a).reshape(1, 1, 3), np.asarray(b).reshape(1, 1, 3))[0, 0])


def name_from_lab(lab) -> str:
    if is_achromatic(lab):
        L = lab[0]
        return "black" if L < 25 else "white" if L > 80 else "gray"
    best, bd = "gray", 1e9
    for n, ref in PALETTE_LAB.items():
        if n in ACHROMATIC:
            continue
        d = _ciede(lab, ref)
        if d < bd:
            bd, best = d, n
    return best


def normalize_color(word: str | None) -> str | None:
    if not word:
        return None
    w = word.lower().strip()
    return SYNONYMS.get(w, w)


def color_gate(region_lab, region_achromatic: bool, query_color: str,
               dE0: float = 10.0, span: float = 40.0) -> float:
    """Soft [0,1] score that a region's colour satisfies the query colour.

    Chromatic query -> CIEDE2000 falloff (full credit <=ΔE 10, zero by ~ΔE 50).
    Achromatic query (white/black/gray) -> lightness match, and a chromatic
    region is rejected (a red garment is not "white")."""
    q = normalize_color(query_color)
    if q in ACHROMATIC:
        if not region_achromatic:
            return 0.12
        dL = abs(float(region_lab[0]) - _ACHRO_L.get(q, 50.0))
        return float(np.clip(1.0 - dL / 45.0, 0.0, 1.0))
    ref = PALETTE_LAB.get(q)
    if ref is None:
        return 0.5  # unknown colour word -> neutral, don't penalise
    if region_achromatic:
        return 0.15  # a gray region isn't "red"/"blue"/...
    dE = _ciede(region_lab, ref)
    return float(np.clip(1.0 - (dE - dE0) / span, 0.0, 1.0))
