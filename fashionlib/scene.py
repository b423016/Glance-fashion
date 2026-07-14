"""Zero-shot scene + formality tagging on the FULL image.

Scene is a global property, so it is matched on the whole image, never on garment
regions. We build a prototype vector per label by ensembling prompt templates
(templating measurably beats bare label strings). Validated on real park/street
images; the dataset's studio skew is a data-quantity issue, not a model issue.
"""
from __future__ import annotations

import numpy as np

from . import config

SCENE_TEMPLATES = ("a photo taken in {}", "a photo of {}", "{}")
FORMALITY_TEMPLATES = ("a person wearing {}", "a photo of {}", "{}")


class SceneTagger:
    def __init__(self, encoder):
        self.enc = encoder
        self.scene_names, self.scene_vecs = self._prototypes(config.SCENE_LABELS, SCENE_TEMPLATES)
        self.form_names, self.form_vecs = self._prototypes(config.FORMALITY_LABELS, FORMALITY_TEMPLATES)

    def _prototypes(self, label_map: dict, templates):
        names, vecs = [], []
        for name, labels in label_map.items():
            prompts = [t.format(label) for label in labels for t in templates]
            v = self.enc.encode_texts(prompts).mean(0)
            vecs.append(v / (np.linalg.norm(v) + 1e-8))
            names.append(name)
        return names, np.asarray(vecs, dtype=np.float32)

    def tag(self, img_vecs: np.ndarray):
        """img_vecs: (N, D) L2-normalised. Returns dict of parallel arrays:
        scene label + full score matrix, formality label + full score matrix."""
        scene_sim = img_vecs @ self.scene_vecs.T
        form_sim = img_vecs @ self.form_vecs.T
        return {
            "scene": [self.scene_names[i] for i in scene_sim.argmax(1)],
            "scene_names": self.scene_names,
            "scene_sim": scene_sim,
            "formality": [self.form_names[i] for i in form_sim.argmax(1)],
            "formality_names": self.form_names,
            "formality_sim": form_sim,
        }

    def score_for(self, img_vecs: np.ndarray, kind: str, label: str) -> np.ndarray:
        """Similarity of each image to one specific scene/formality label
        (used as a soft boost/filter at query time)."""
        names = self.scene_names if kind == "scene" else self.form_names
        vecs = self.scene_vecs if kind == "scene" else self.form_vecs
        if label not in names:
            return np.zeros(len(img_vecs), dtype=np.float32)
        return img_vecs @ vecs[names.index(label)]
