# Design & Chosen Approach - Multimodal Fashion & Context Retrieval

---

## 1. The problem, and why a vanilla CLIP baseline is not enough

The objective is text-to-image retrieval over a fashion gallery that resolves
**garment x colour x environment x vibe**, handles **compositional** queries,
and works **zero-shot** (no query-to-image training labels).

A single pooled CLIP/SigLIP embedding behaves like a **bag of words across
modalities**: it aligns the *set* of concepts in a caption with the *set* in an
image but binds attributes to objects poorly. So *"a red tie and a white shirt"*
lands almost on top of *"a white tie and a red shirt"* - even a fashion-tuned
SigLIP scores ~4.5/100 on controlled colour-binding. I confirmed this on my own
gallery: with a global embedding, the query *"red tie + white shirt"* actually
retrieves **more** of the *swapped* "white tie + red shirt" images than correct
ones (negative discrimination - see Section 4).

**Design principle:** keep a strong fashion encoder for cheap *recall*, but add a
**training-free compositional re-ranker** that makes attribute-to-object binding
**explicit** - decompose the query, match each clause to a **distinct garment
region**, and require **all** clauses. This decompose-bind-conjunctive-aggregate
pattern is the current training-free SOTA family (ComCLIP / ABE-CLIP), and the
distinct-region step reproduces the mechanism of TTM (2025); see `approaches.md`.

## 2. Architecture

One encoder throughout: **`Marqo/marqo-fashionSigLIP`** (open_clip, 768-d,
Apache-2.0) - SOTA open fashion text-to-image, runs on Apple-Silicon MPS. Store:
**LanceDB** (embedded; same API scales on-disk / S3 to ~1B vectors).

**Gallery:** Fashionpedia **val** (1,158 imgs) + a targeted **train supplement**
(every one of the 1,455 tie images - so the "red tie + white shirt" query has real
positives - plus a random sample for environment variety) = **4,113 images,
31,170 garment regions**. Same dataset, same COCO masks, so the pipeline is
identical; masks are used only as *geometry*.

### Part A - Indexer (`indexer/build_index.py`)

Per image: **global embedding** (whole image) + zero-shot **scene/formality tags**.
Per garment instance (region from the GT mask):
- **Region embedding** - FashionSigLIP on a **padded bbox crop with context
  pixels** (NOT a background-zeroed cutout: a masked cutout is out-of-distribution
  for the encoder and hurts - the red-teamed failure of the naive version).
- **Dominant colour** - k-means in **CIELab** over the **eroded** mask, stored as a
  Lab triple + an **achromatic flag** (white/black/grey matched by lightness).

The mask has two consumers - *padded bbox for embedding*, *eroded mask for colour* -
and nothing scored comes from Fashionpedia's label vocabulary, so retrieval stays
zero-shot.

### Part B - Retriever (`retriever/`)

```
query --> parse --> { clauses:[(red,tie),(white,shirt)], scene, formality }
        |-- Stage 1  RECALL: ANN over the global embedding -> top-N   (scales to 1M)
        +-- Stage 2  RERANK (top-N only):
             per clause  s = MAX over regions of  cos(garment_text, region) x colour_gate
             Hungarian assignment (distinct regions)   <-- clauses can't share a garment
             soft-MIN across clauses (alpha*min + (1-alpha)*mean)   <-- require ALL clauses
             + scene / formality soft score
        +-- FUSE (min-max calibrated components): w_global*g + w_bind*b + w_scene*s + w_form*f
```

The three mechanisms that do the compositional work: a **soft CIEDE2000 colour
gate** (a penalty, not a hard filter), **Hungarian assignment** (the step that
kills the "red tie = red shirt" swap), and **soft-MIN** (AND-semantics that
degrade gracefully). `mode="global"` disables Stage 2 for the ablation baseline.

## 3. How it answers the five evaluation queries

| # | Query | Mechanism |
|---|-------|-----------|
| 1 | *bright yellow raincoat* | one clause; CIELab colour gate on the outerwear region |
| 2 | *business attire in a modern office* | no bindable attribute, so scene(office)+formality(formal) over the global embedding |
| 3 | *blue shirt on a park bench* | clause (blue, shirt) via region MAX + colour gate; scene(park) boost |
| 4 | *casual weekend, city walk* | no clause, so scene(street)+formality(casual) + global vibe |
| 5 | *red tie **and** white shirt, formal* | two clauses; achromatic-white gate, **Hungarian** forces tie != shirt, soft-MIN requires both |

## 4. Results - does it beat vanilla CLIP?

Fashionpedia has no query-to-image relevance labels, so I synthesised a relevance
judge **from the annotations** (GT garment category + the extracted colour), used
only for *evaluation*, never for retrieval, and identical for both modes. *(Caveat:
the judge shares its colour reading with the retriever's gate, so it measures
consistency-with-extraction; the visual boards and the hard-negative test serve as
the independent checks.)* `glance-eval` produces the table and boards.

**Global (vanilla fashion-CLIP) vs full (my pipeline), 4,113-image gallery:**

| Query | positives | P@8 g to f | nDCG@10 g to f | AP@20 g to f |
|-------|:---:|:---:|:---:|:---:|
| Q1 yellow raincoat | 11 | 0.38 to **0.62** | 0.47 to **0.68** | 0.26 to **0.54** |
| Q3 blue shirt (+park) | 245 | 0.50 to **0.62** | 0.51 to 0.47 | 0.19 to **0.28** |
| **Q5 red tie + white shirt** | 51 | **0.25 to 0.62** | 0.32 to **0.55** | 0.13 to **0.35** |
| S2 blue top | 245 | 0.75 to **1.00** | 0.72 to **0.93** | 0.44 to **0.59** |
| S3 red dress | 122 | 1.00 to 1.00 | 0.93 to **1.00** | 0.91 to **1.00** |
| S4 white top + black pants | 894 | 0.75 to **1.00** | 0.85 to **0.93** | 0.56 to **0.76** |
| S1 yellow jacket / S5 black jacket+white top | 11 / 778 | 0.88 / 0.50 (flat) | - | - |

**Mean over judgeable queries: P@8 0.63 to 0.78 (+25%), nDCG@10 +15%, AP@20 +35%.**
The largest lifts are on the compositional and rarer-positive queries; where
positives are abundant (S5, 778 of 4,113) both modes already fill the top-8, so
full is flat - expected, not a regression.

**The binding proof (`glance-eval-hardneg`).** Precision alone can be satisfied by
colour filtering; to prove that I actually *bind* attributes, I run colour-swap
pairs and report the discrimination gap = P@8(query on its OWN spec) - P@8(query
on the SWAPPED spec):

| Swap pair | global | **full** |
|-----------|:---:|:---:|
| **red tie + white shirt** vs *white tie + red shirt* | **-0.06** | **+0.06** |
| white top + black pants vs *black top + white pants* | +0.44 | **+0.56** |
| black jacket + white top vs *white jacket + black top* | +0.31 | +0.31 |

On the marquee query the **global baseline is negative** - it cannot tell the swap
apart and actually prefers the wrong binding - while **full flips positive**. The
boards make it visible: for *"white top + black pants"* the global baseline mixes
in wrong-colour tops and black shorts, while full returns clean white-top/black-pants
results (P@8 0.75 to 1.00), and the *"red tie + white shirt"* board now surfaces
genuine suited red-tie/white-shirt looks. The hard-negative queries deliberately
strip the scene/formality cue to isolate **pure colour-garment binding** - which
is why Q5's binding-only own-P@8 here (0.25) is below its full-query 0.62 in
Section 4's table, which also benefits from the *"formal"* filter.

## 5. Honest limitations

- **Binding margin is modest on abundant-positive queries.** When a colour+garment
  combination is common (S5: 778 positives), a global embedding already fills the
  top-8, so binding adds little there; the gain concentrates on rarer/compositional
  queries. This is inherent to precision@k on a common attribute, not a bug.
- **Garment identity from region embeddings is soft** at SigLIP's compressed
  similarity scale, so the colour gate carries much of the binding signal; a
  zero-shot garment-type prior would sharpen it (kept out to stay label-free).
- **Colour extraction is noisy** on textured/low-light garments - one reason the
  gate is soft, not hard.
- **Scene axis** relies on the whole-image embedding; it is validated
  (park/street tag correctly) but the environment labels are coarse.

## 6. Scalability & zero-shot (grading criteria)

- **To 1M images:** recall is sublinear ANN over **one vector per image**
  (LanceDB HNSW/IVF, on-disk/S3); the expensive binding runs **only on the top-N
  recalled candidates**, so cost scales with `N`, not the gallery. The single
  change needed is swapping GT masks for an open-vocab detector (Grounding DINO /
  YOLO-World) in `_embed_regions`; every downstream step remains unchanged.
- **Zero-shot:** every scored signal is an embedding (garment text vs region) or a
  pixel measurement (colour) - no per-label training, no fine-tuning. Unrecognised
  garment/colour words fall back gracefully. Nothing is keyword-matched against
  Fashionpedia's 46 categories / 294 attributes.

## 7. Codebase

**GitHub:** [https://github.com/b423016/Glance-fashion](https://github.com/b423016/Glance-fashion)

Installable package (`pip install -e .`) with console scripts
(`glance-index/query/eval/eval-hardneg`), a pytest unit suite (colours, parsing,
binding, metrics), and ruff-clean source. Modular layout separates **logic**
(`fashionlib/`) from the two **workflows** (`indexer/`, `retriever/`), the
**evaluation** (`evaluation/`), and the **data** (`data/`, `index/`).
