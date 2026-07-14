# Approaches, Trade-offs & Future Work

I evaluated five approaches (a global-embedding baseline plus four ways to beat
it), adversarially red-teaming each on the five target queries and on the grading
axes, before committing. Below is what each approach entails, when it is the right
call, and why I chose a hybrid.

---

## 1. Approaches considered

### A. Global fashion-embedding (baseline)
Index one FashionSigLIP vector per image; retrieve by cosine ANN.
- **Good when:** single-concept queries ("a red dress"), huge scale, minimal code.
  Already beats OpenAI CLIP on fashion by a wide margin.
- **Weak when:** any multi-attribute query - bag-of-words binding fails
  ("red tie + white shirt" becomes indistinguishable from "white tie + red shirt").
  **Useful as recall and baseline, never as the final scorer.**

### B. Structure-guided region binding (inference-time) - core of my chosen approach
Decompose the query into (colour, garment) clauses; embed each garment region;
score `cos(garment_text, region) x colour_gate`; bind clauses to **distinct**
regions (Hungarian) and require **all** (soft-MIN).
- **Good when:** compositional / fine-grained colour queries; a
  **training-free**, **zero-shot**, fully-local system is needed; explainability
  (each region can be shown satisfying a specific clause).
- **Weak when:** region proposals are bad (mitigated here by GT masks; a detector
  would be used at scale); garment identity is fuzzy at SigLIP's compressed
  similarity scale.

### C. Multi-vector late-interaction (ColBERT-style MaxSim)
Store a bag of region vectors per image; match query phrase-vectors via MaxSim.
- **Good when:** the binding *mechanism* is desired without an explicit parser.
- **Weak when:** SigLIP is pooled-to-pooled (no true per-token late interaction),
  so the multivector index costs 5-15x storage for roughly no recall gain.
  I kept its *lesson* (per-clause MAX over regions) and dropped its *index*.

### D. VLM structured-caption index (generative)
Caption every image into structured JSON with a VLM; retrieve via LLM-parsed
filters plus text similarity.
- **Good when:** style / vibe / scene description matters most; a paid API or
  GPU is available; a smaller closed gallery.
- **Weak when:** local-1M-scale (per-image VLM captioning forces an API/GPU
  exactly where scalability is graded); the index is frozen at caption time; VLMs
  are ~11% exact on colour. I use its strength (scene/style) as an **optional**
  top-N reranker, never as the index.

### E. Hard-negative fine-tune (training-time; NegCLIP/CLIC)
Fine-tune the encoder on colour-garment hard negatives so the *global* embedding
binds better.
- **Good when:** a GPU is available and the fix should be baked into a single
  fast vector.
- **Weak when:** the whole compositional bet lives in a GPU step; stripped to run
  Mac-local it collapses to the baseline. Best kept as an *offline precision boost*.

## 2. Trade-off scorecard

Weights: compositional 0.28, scalability 0.14, zero-shot 0.14, mac-feasibility 0.12,
scene 0.10, style 0.08, build-speed 0.06, impressiveness (adversarial red-team,
0-10 per axis).

| Approach | compositional | scalability | zero-shot | mac-local | **weighted** |
|----------|:---:|:---:|:---:|:---:|:---:|
| **Hybrid (B-core + optional D graft)** | 7 | 7.5 | 8 | 8 | **7.21** |
| C - late-interaction | 8 | 7 | 8 | 8 | 7.08 |
| B - region-binding (alone) | 7 | 7 | 8 | 8 | 6.80 |
| D - VLM-caption index | 7.5 | 5.5 | 6 | 7 | 6.33 |
| E - hard-neg finetune | 3 | 9 | 6 | 2 | 5.04 |
| A - global baseline | (low) | 10 | 8 | 9 | - (baseline) |

**Chosen: B-core hybrid** - highest weighted score; wins the heaviest axis
(compositional) with a literature-backed, training-free mechanism; stays honestly
zero-shot and fully Mac-local; lighter to build than C's multivector index; and
keeps D's scene/style strength available as an optional, off-by-default reranker.

## 3. Why this is the SOTA choice - and why not the heavier options

I validated the design against a survey of mid-2026 SOTA (general and fashion
embedders, instruction-tuned multimodal embedders, training-free compositionality
methods, VLM rerankers, open-vocab detectors, and CIR benchmarks):

- **The region-binding rerank I use IS the current training-free SOTA family.** The
  decompose-per-clause-bind-conjunctive-aggregate pattern matches ComCLIP and
  ABE-CLIP (2025); the Hungarian distinct-region assignment independently
  reproduces the winning mechanism of TTM (Oct 2025).
- **I deliberately did NOT replace it with an instruction-tuned multimodal
  embedder** (GME / VLM2Vec / Qwen-VL-Embedding). They still pool to a single
  contrastive vector and inherit the same bag-of-words binding failure; a higher
  MMEB rank does not mean better binding, and general encoders *regress in-domain*
  (SigLIP2 0.261 vs fashion-tuned 0.283 R@10).
- **I deliberately did NOT add a VLM reranker.** A cross-encoder VLM could raise
  compositional precision further, but it is a large model in the query loop
  (~15-45 s/query on-device) - heavy and slow for a marginal gain. Lightweight
  region binding captures most of the benefit, training-free.
- **I deliberately did NOT fine-tune** (hard-negative / NegCLIP): that would put
  the whole compositional bet in a GPU training step, against the zero-shot brief.
  Kept as an optional offline booster only.

Net: the pipeline sits in the SOTA family while staying **lightweight, local, and
zero-shot** - the right operating point for this task.

## 4. Future work

### (a) Adding locations (cities / places) and weather
Both are **global scene properties**, so they slot in as extra tag/metadata heads
next to the existing scene tagger - **no retraining of the fashion encoder**:
- **Place / city:** attach a location embedding with **GeoCLIP** or **StreetCLIP**
  (trained on street-view / geotagged imagery) and either filter or fuse a
  location clause ("a summer outfit **in Tokyo**"). Landmark/OCR signals and EXIF
  geotags refine it further.
- **Weather:** a small transfer-learned weather classifier
  (sunny / cloudy / rainy / snowy / hazy, ~87-95% accuracy) writes a filterable
  `weather` column, enabling "a **rainy-day** street outfit". Because the query
  parser already extracts a scene, adding `place` and `weather` slots is a parser
  plus metadata-column change, not an architecture change.

### (b) Improving precision
- **Better regions at scale:** replace GT masks with **Grounding DINO / YOLO-World**
  open-vocab detection (already the documented 1M swap point); precision tracks
  region quality.
- **Sharper garment identity:** a light **hard-negative fine-tune** (CLIC/NegCLIP,
  approach E) of the region encoder on colour-garment swaps - lifts binding without
  hurting downstream retrieval, run offline.
- **VLM verification on top-k** (approach D as reranker): a cheap
  "is the tie red? is the shirt white?" VQA pass over the final top-k removes
  residual binding errors - especially valuable for the hardest compositional and
  vibe queries.
- **Learned fusion:** the current fusion weights are hand-set with calibrated
  (min-max) components; with a handful of labelled queries they can be tuned
  (e.g. logistic re-ranker over the component scores).
- **Cleaner colour:** grab-cut / matting instead of mask erosion, and a
  chroma-weighted palette, to reduce the colour-extraction noise noted in the
  limitations.
- **Proper eval:** collect a small human-labelled relevance set (or expand the
  synthetic judge) to report Recall@k / mAP with confidence intervals, not just
  precision@k on auto-judged queries.
