"""Parse a natural-language query into structured intent.

Deterministic, explainable lexicon parser: extract (colour, garment) clauses,
a scene, and a formality. It is intentionally thin — the *matching* it feeds is
embedding-based (garment text -> region vector) and pixel-based (colour gate), so
retrieval stays zero-shot even for garment words outside the lexicon: an
unrecognised noun still becomes a clause and is matched by embedding similarity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from fashionlib import colors

# garment word -> canonical noun used as the region-embedding query text
GARMENTS = {
    "raincoat": "raincoat", "coat": "coat", "overcoat": "coat", "trench": "trench coat",
    "jacket": "jacket", "blazer": "blazer", "parka": "parka", "windbreaker": "windbreaker",
    "tie": "tie", "necktie": "tie", "bowtie": "bow tie",
    "shirt": "shirt", "blouse": "blouse", "button-down": "button-down shirt",
    "t-shirt": "t-shirt", "tshirt": "t-shirt", "tee": "t-shirt", "top": "top",
    "sweater": "sweater", "sweatshirt": "sweatshirt", "hoodie": "hoodie", "cardigan": "cardigan",
    "dress": "dress", "gown": "gown", "skirt": "skirt", "jumpsuit": "jumpsuit",
    "pants": "pants", "trousers": "trousers", "jeans": "jeans", "chinos": "chinos",
    "shorts": "shorts", "suit": "suit", "vest": "vest", "waistcoat": "vest",
    "scarf": "scarf", "hat": "hat", "cap": "cap", "shoe": "shoes", "shoes": "shoes",
    "boots": "boots", "sneakers": "sneakers", "bag": "bag", "handbag": "handbag",
}

MULTIWORD_COLORS = ["light blue", "navy blue", "sky blue", "off-white", "dark blue"]
COLOR_WORDS = set(colors.PALETTE) | set(colors.SYNONYMS) | {"grey"}
INTENSITY = {"bright", "dark", "deep", "pale", "vivid", "rich", "muted", "light"}

SCENE_WORDS = [
    (("office", "workplace", "corporate", "meeting room", "desk"), "office"),
    (("park", "bench", "garden", "grass", "outdoor park"), "park"),
    (("street", "city", "urban", "sidewalk", "road", "downtown"), "street"),
    (("home", "living room", "bedroom", "indoors at home", "house", "couch"), "home"),
]
FORMAL_WORDS = {"formal", "business", "professional", "elegant", "corporate", "suit", "office"}
CASUAL_WORDS = {"casual", "weekend", "relaxed", "streetwear", "everyday", "laid-back"}


@dataclass
class Clause:
    garment: str
    color: str | None = None

    def __repr__(self):
        return f"{self.color+' ' if self.color else ''}{self.garment}"


@dataclass
class Query:
    raw: str
    clauses: list[Clause] = field(default_factory=list)
    scene: str | None = None
    formality: str | None = None

    def summary(self) -> str:
        parts = [f"clauses={self.clauses}"]
        if self.scene:
            parts.append(f"scene={self.scene}")
        if self.formality:
            parts.append(f"formality={self.formality}")
        return " | ".join(parts)


def parse(text: str) -> Query:
    q = Query(raw=text)
    low = " " + text.lower() + " "
    for mw in MULTIWORD_COLORS:
        low = low.replace(mw, mw.replace(" ", "_"))
    tokens = re.findall(r"[a-z][a-z\-_']*", low)

    pending_color: str | None = None
    for tok in tokens:
        word = tok.replace("_", " ")
        if word in COLOR_WORDS or word in {c.replace(" ", "_") for c in MULTIWORD_COLORS} or "_" in tok:
            cand = colors.normalize_color(word)
            if cand in colors.PALETTE or cand in COLOR_WORDS:
                pending_color = cand
                continue
        if tok in INTENSITY:
            continue
        if tok in GARMENTS:
            q.clauses.append(Clause(garment=GARMENTS[tok], color=pending_color))
            pending_color = None
    # scene
    for words, label in SCENE_WORDS:
        if any(w in low for w in words):
            q.scene = label
            break
    # formality
    if any(w in low for w in FORMAL_WORDS):
        q.formality = "formal"
    elif any(w in low for w in CASUAL_WORDS):
        q.formality = "casual"
    return q
