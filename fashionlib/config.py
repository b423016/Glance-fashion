"""Central configuration: paths, model ids, device, and retrieval knobs.

Keeping every tunable here is what lets the rest of the code stay *logic only*
(no hard-coded paths, no magic numbers scattered around) — data lives under
``data/``/``index/``, logic lives in ``fashionlib/``.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "test"                    # val + test jpgs share this folder
VAL_ANN = DATA_DIR / "instances_attributes_val2020.json"
TRAIN_IMAGES_DIR = DATA_DIR / "train"             # train supplement images
TRAIN_ANN = DATA_DIR / "instances_attributes_train2020.json"
INDEX_DIR = PROJECT_ROOT / "index"                # LanceDB table dir
OUTPUT_DIR = PROJECT_ROOT / "outputs"             # rendered result boards

# --- encoder ---------------------------------------------------------------
FASHION_MODEL = "hf-hub:Marqo/marqo-fashionSigLIP"  # SOTA open fashion CLIP/SigLIP
EMBED_DIM = 768


def get_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --- retrieval knobs -------------------------------------------------------
TOP_K = 8
RECALL_N = 60          # stage-1 candidates handed to the compositional re-rank
COLOR_K = 4            # k-means clusters for dominant-colour extraction
MIN_AREA = 24 * 24     # drop tiny garment instances (logos, buttons, etc.)

# compositional binding + fusion
BIND_ALPHA = 0.7       # soft-MIN interpolation: alpha*min + (1-alpha)*mean over clauses
W_GLOBAL = 0.4         # weight of global-semantic score (always active)
W_BIND = 0.6           # weight of compositional binding score (when clauses present)
W_SCENE = 0.5          # weight of scene match (when a scene is parsed)
W_FORM = 0.25          # weight of formality match (when formality is parsed)

# --- scene taxonomy (zero-shot, matched on the FULL image) -----------------
SCENE_LABELS = {
    "office": ["a modern office interior", "an office workspace", "a corporate meeting room"],
    "street": ["a city street", "an urban sidewalk", "a busy street scene"],
    "park": ["a park with grass and trees", "a green outdoor park", "a garden"],
    "home": ["a home interior", "a living room at home", "indoors at home"],
    "studio": ["a plain studio backdrop", "a fashion studio shot", "a neutral background"],
}

# --- formality cue prompts (zero-shot) -------------------------------------
FORMALITY_LABELS = {
    "formal": ["formal business attire", "a professional suit and tie", "elegant formalwear"],
    "casual": ["casual everyday clothing", "relaxed weekend outfit", "streetwear"],
}
