"""fashionlib — shared, reusable logic for the fashion retrieval engine.

Kept deliberately separate from the two *workflow* entrypoints (``indexer/`` and
``retriever/``) and from the *data* (``data/``, ``index/``) so that logic can be
tested and reused independently. See docs/design.md for the architecture.
"""
__all__ = ["config", "data", "regions", "colors"]
