# Glance — Multimodal Fashion & Context Retrieval

Text→image search over a fashion gallery that understands **what** you're wearing,
**what colour**, and **where** you are — and gets **compositional** queries right
(*"a red tie **and** a white shirt"* ≠ *"a white tie and a red shirt"*), which a
vanilla CLIP retriever cannot.

## Why not just CLIP?

A single pooled CLIP/SigLIP vector behaves like a **bag of words across modalities**:
it binds attributes to objects poorly, so *"red tie and white shirt"* scores almost
the same as *"white tie and red shirt"*. Even a fashion-tuned SigLIP scores ~4.5/100
on controlled colour-binding. We keep that strong encoder for **recall** but add a
**training-free compositional re-ranker** that decomposes the query, matches each
attribute clause to a **distinct garment region**, and requires **all** of them.

This decompose→per-clause-bind→conjunctive-aggregate design is the current
training-free SOTA family for compositionality (ComCLIP / ABE-CLIP), and the
Hungarian distinct-region step reproduces the mechanism of TTM (2025). It is
deliberately **lightweight and zero-shot**: no fine-tuning, no VLM in the loop.

## Architecture (two stages)

```
                    ┌── Part A: Indexer (offline) ──────────────────────────┐
   Fashionpedia →   │  full image  → FashionSigLIP embedding  → images table │
   val (+ train     │              → zero-shot scene/formality tags          │
   supplement)      │  each garment (GT mask) →                              │
                    │        padded-crop → region embedding   → regions table│  ← LanceDB
                    │        eroded mask → CIELab dominant colour            │
                    └───────────────────────────────────────────────────────┘

                    ┌── Part B: Retriever (online) ─────────────────────────┐
   "a red tie and   │  parse → clauses[(red,tie),(white,shirt)], scene, form │
    a white shirt"  │  Stage 1  ANN recall on global embedding  (scales to 1M)│
                    │  Stage 2  per clause: MAX over regions of               │
                    │           cos(garment_text, region) × CIEDE2000 colour │
                    │           Hungarian (distinct regions) → soft-MIN       │
                    │  fuse (calibrated): global + binding + scene + formality│
                    └───────────────────────────────────────────────────────┘
```

**Model:** `Marqo/marqo-fashionSigLIP` (open_clip, 768-d, Apache-2.0) — one encoder
for image regions, whole-image scene, and text. **Store:** LanceDB (embedded; same
API scales on-disk/S3 to ~1B). Runs fully local on Apple-Silicon (MPS), no GPU/API.

Result (see `docs/`): on compositional queries the full pipeline lifts precision@8
well above the global fashion-CLIP baseline, with the largest gains on two-clause
queries — and a hard-negative test confirms it *binds* attributes rather than
merely filtering colours.

## Layout

```
fashionlib/          shared, reusable logic (separated from data + workflows)
  config.py          all tunables (paths, model id, weights, thresholds)
  data.py            Fashionpedia val loader + targeted train supplement
  models.py          FashionSigLIP encoder wrapper
  regions.py         padded region crop  +  eroded-mask CIELab colour
  colors.py          CIELab palette naming + CIEDE2000 query→region colour gate
  scene.py           zero-shot scene / formality tagging
  schema.py, store.py  LanceDB row schemas + read/write/recall
  viz.py             result-board renderer
indexer/build_index.py    Part A entrypoint  (glance-index)
retriever/parse.py        query → structured clauses/scene/formality
retriever/bind.py         Hungarian + soft-MIN compositional binding
retriever/search.py       recall → rerank → calibrated fusion (+ global ablation)
retriever/query.py        Part B CLI            (glance-query)
evaluation/run_eval.py    global-vs-full ablation: P@8 / nDCG@10 / AP@20 + boards
evaluation/hard_negatives.py  colour-swap binding test          (glance-eval-hardneg)
evaluation/metrics.py     pure rank-aware metrics
tests/                    pytest unit tests (colours, parsing, binding, metrics)
docs/                     design + approaches write-up (the PDF source)
```

## Setup

```bash
# 1. environment (Python 3.12 via uv; torch 2.x with MPS) + editable install
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,docs]"
source .venv/bin/activate

# 2. data — Fashionpedia val gallery (masks) + optional train supplement
mkdir -p data
BASE=https://s3.amazonaws.com/ifashionist-dataset
curl -L -o data/val_test2020.zip $BASE/images/val_test2020.zip && unzip -q data/val_test2020.zip -d data
curl -L -o data/instances_attributes_val2020.json $BASE/annotations/instances_attributes_val2020.json
# optional (for the "red tie + white shirt" query): the train supplement
curl -L -o data/train2020.zip $BASE/images/train2020.zip && unzip -q data/train2020.zip -d data
curl -L -o data/instances_attributes_train2020.json $BASE/annotations/instances_attributes_train2020.json
```

## Usage

```bash
glance-index --supplement          # Part A: build the index (val + train supplement)
glance-query "a red tie and a white shirt in a formal setting" --render   # Part B
glance-query "a bright yellow raincoat" --mode global                     # ablation baseline
glance-eval                        # global-vs-full ablation → outputs/*.png + eval_summary.json
glance-eval-hardneg                # colour-swap binding proof → outputs/HN_*.png
```

## Development

```bash
pytest          # unit tests for the pure logic (colours, parsing, binding, metrics)
ruff check .    # lint
```

## Scaling to 1M images

Recall is sublinear ANN over one vector per image (LanceDB HNSW/IVF, on-disk/S3);
the expensive compositional binding runs **only on the top-N recalled candidates**,
so cost grows with `N`, not the gallery. The single change at scale is swapping the
Fashionpedia GT masks for an open-vocab detector (Grounding DINO / YOLO-World) in
`indexer/build_index.py::_embed_regions` — every downstream step is identical.

See `docs/design.md` (chosen architecture, results) and `docs/approaches.md`
(alternatives, SOTA positioning, future work) for the full write-up.
