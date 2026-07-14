"""Render a top-k result board (contact sheet) as a PNG — used by the CLI and
the evaluation to produce human-inspectable evidence."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def board(items, out_path, cols=4, thumb=256, pad=8, title=None, header=28):
    """items: list of (image_path, caption). Saves a grid PNG."""
    items = list(items)
    rows = (len(items) + cols - 1) // cols
    cap_h = 34
    cell = thumb + cap_h
    W = cols * thumb + (cols + 1) * pad
    top = header + pad if title else pad
    H = top + rows * cell + rows * pad
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(canvas)
    if title:
        d.text((pad, pad), title, fill=(20, 20, 20))
    for i, (path, cap) in enumerate(items):
        r, c = divmod(i, cols)
        x = pad + c * (thumb + pad)
        y = top + r * (cell + pad)
        try:
            im = Image.open(path).convert("RGB")
            im.thumbnail((thumb, thumb))
        except Exception:
            im = Image.new("RGB", (thumb, thumb), (200, 200, 200))
        ox = x + (thumb - im.width) // 2
        canvas.paste(im, (ox, y))
        for j, line in enumerate(str(cap).split("\n")[:2]):
            d.text((x + 2, y + thumb + 2 + j * 14), line[:46], fill=(30, 30, 30))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path
