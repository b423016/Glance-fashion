"""Thin wrapper around Marqo-FashionSigLIP (via open_clip).

One place that knows how to turn images / garment crops / text into L2-normalised
768-d vectors. Everything downstream works in this shared embedding space, which
is what keeps text->image retrieval zero-shot (no per-label training).
"""
from __future__ import annotations

import numpy as np
import open_clip
import torch
from PIL import Image

from . import config


class Encoder:
    def __init__(self, model_name: str = config.FASHION_MODEL, device: str | None = None):
        self.device = device or config.get_device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def encode_images(self, imgs: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(imgs), batch_size):
            batch = [self.preprocess(im.convert("RGB")) for im in imgs[i:i + batch_size]]
            t = torch.stack(batch).to(self.device)
            f = self.model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.float().cpu().numpy())
        return np.concatenate(out, 0) if out else np.zeros((0, config.EMBED_DIM), np.float32)

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            tok = self.tokenizer(texts[i:i + batch_size]).to(self.device)
            f = self.model.encode_text(tok)
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.float().cpu().numpy())
        return np.concatenate(out, 0) if out else np.zeros((0, config.EMBED_DIM), np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]

    def encode_image(self, img: Image.Image) -> np.ndarray:
        return self.encode_images([img])[0]
