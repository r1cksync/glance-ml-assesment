# Approaches Considered

The assignment defines the failure mode to beat explicitly: vanilla CLIP retrieval
"fails at compositionality ('red shirt with blue pants' vs 'blue shirt with red pants')
and at fine-grained fashion attributes." Every approach below is therefore judged
against the same six axes: **compositional binding**, **fine-grained attribute
fidelity**, **scene/context awareness**, **zero-shot robustness**, **query latency**,
and **infra cost**. The five acceptance queries exercise all of them — one is purely
attributive ("bright yellow raincoat"), one purely contextual ("inside a modern
office"), one is a garment+scene conjunction ("blue shirt … park bench"), one is
style inference ("casual weekend outfit"), and one is a two-garment compositional
binding ("a red tie AND a white shirt in a formal setting").

---

## a) Vanilla CLIP, whole-image zero-shot (the baseline)

**How it works.** Each image is encoded once by CLIP's image tower into a single
512-d vector (ViT-B/32); the query is encoded by the text tower into the same space;
retrieval is cosine top-k over one flat index. No detection, no metadata, no rerank.

**Strengths.**
- Zero training, zero labeling, one model, one index — deployable in an afternoon.
- Genuinely good at coarse semantics and "vibe": scene type, overall formality,
  broad color dominance. It would do respectably on "casual weekend outfit for a
  city walk" because that query *is* a holistic gestalt.
- Millisecond latency on CPU; the whole 1K-image index is ~2 MB.

**Weaknesses — and why it fails, specifically.**
- **Bag-of-words text encoder.** CLIP's contrastive objective only requires
  separating matched image–caption pairs from *other captions in the batch*. In-batch
  negatives almost never differ from the positive by word order alone, so the text
  encoder never receives gradient pressure to encode syntax. The result (documented
  by Winoground, ARO, SugarCrepe) is that `emb("red tie and white shirt")` ≈
  `emb("white tie and red shirt")` — near-chance performance on attribute–object
  binding. The compositional acceptance query is unanswerable *by construction*.
- **Single-vector bottleneck.** One 512-d point must simultaneously encode every
  garment, every color, the scene, the lighting, and the person. Attribute–object
  bindings are superposed and interfere; there is no slot structure to keep "red"
  attached to "tie".
- **Small-object dilution.** A tie occupies ~2–5% of image pixels. Global pooling
  over ViT patches means large regions (background, coat, wall) dominate the
  embedding; accessory-level queries ("watch", "belt", "tie") have weak signal.
- **Fine-grained fashion vocabulary.** LAION-style web pretraining underrepresents
  garment taxonomy. "Cardigan" vs "pullover" vs "sweatshirt", "chinos" vs "slacks",
  material words — all noisy directions in embedding space.
- **No negation, no filtering.** "Casual outfit, no denim" typically *increases*
  similarity to denim images, because "denim" is present in the text and the encoder
  has no mechanism for exclusion.

**When it IS the right call.** Coarse thematic retrieval, mood boards, near-duplicate
detection, single-subject product shots, or as a first prototype to bound the problem.
It is also the correct *scene channel* inside a larger system — which is exactly how
this codebase uses it (`embedding.scene_backend: clip-vit-b32`).

**Cost/complexity.** Trivial. ~150 MB model, CPU-only, one flat FAISS index.

---

## b) FashionCLIP / fashion-domain dual encoders, whole-image

**How it works.** Same dual-encoder architecture and same serving path as (a), but the
encoder is contrastively fine-tuned on fashion catalog data — `patrickjohncyh/fashion-clip`
(~700K Farfetch product image–caption pairs) or `Marqo/marqo-fashionSigLIP` (SigLIP
objective, fashion-specific multi-part loss). Drop-in vector swap.

**Strengths.**
- Substantially better garment-type and attribute alignment: "raincoat", "blazer",
  "turtleneck", color–garment phrases all point in cleaner directions. This measurably
  lifts the attribute-specific acceptance query.
- Identical infra cost to (a) — the improvement is free at serving time.

**Weaknesses.**
- **Fixes the vocabulary problem, not the structural one.** It is still one vector per
  image, still a bag-of-words text tower trained with in-batch negatives. Compositional
  binding fails the same way as (a); the ablation ladder in `eval/ablations.py` runs
  this exact variant (`fashionclip-whole`) to demonstrate it.
- **Catalog domain shift.** Training data is clean studio/product photography.
  On in-the-wild Fashionpedia street scenes, the gain shrinks.
- **Scene understanding gets worse, not better.** Fine-tuning on garment-centric
  captions trades away general visual knowledge; "modern office" and "park bench" are
  weaker in FashionCLIP space than in vanilla CLIP space. This is precisely why the
  built system keeps two separate embedding spaces — FashionCLIP for garment crops,
  vanilla CLIP for the whole-image scene vector — rather than one "better" encoder
  for everything.

**When it IS the right call.** E-commerce/catalog search where each image contains one
dominant garment and queries are single-attribute ("red midi dress"). As the *crop
encoder* in a region pipeline — its actual role here.

**Cost/complexity.** Same as (a).

---

## c) Region-based embedding (detector + per-garment crops)

**How it works.** A garment detector (here: YOLOS fine-tuned on Fashionpedia,
`valentinafeve/yolos-fashionpedia`) proposes up to 8 garment regions per image; each
padded crop is embedded separately with the fashion encoder; an image becomes N garment
vectors. At query time each parsed garment term ("red tie", "white shirt") is embedded
and scored against region vectors, taking the per-image max per term.

**What it fixes.**
- **Localization / small-object dilution.** The tie is now 100% of its own crop.
  Its color and type dominate its own vector instead of contributing 3% of a global one.
- **Attribute binding at the vector level.** "Red tie" is scored against the *tie
  crop*. A red shirt elsewhere in the image cannot satisfy the red-tie term, because it
  lives in a different vector. Binding becomes spatial rather than linguistic.
- **Explainability for free.** Every match is (query term → region → bbox →
  similarity), which is exactly what the explain panel renders (`MatchExplanation`
  in `core/schemas.py`).
- **Accessory recall.** Ties, belts, hats, watches become first-class retrievable units.

**What it still can't do alone.**
- **Cross-garment conjunction is still soft.** Scores fuse numerically: an image with
  a superb red-tie match and no white shirt can outrank one with decent matches for
  both. Nothing *requires* both terms to be satisfied — you need a hard conjunctive
  filter for that (approach e).
- **Within-crop composition is still bag-of-words.** "Red tie with white polka dots"
  hits the same encoder limitation inside the crop.
- **Detector recall is a hard ceiling.** A missed detection is silent, unrecoverable
  recall loss; occluded or unusual garments simply don't exist in the index.
- **Scene context disappears entirely** unless a separate whole-image channel is kept.
- Still no negation mechanism.

**When it IS the right call.** Whenever queries name specific garments or accessories,
or images contain multiple people/garments. It is the single highest-leverage upgrade
over (a)/(b) for fashion, which is why it is rung (c) of the ablation ladder.

**Cost/complexity.** Detector adds index-time cost (one forward pass per image) and
multiplies vector count ~4–8×. Query cost scales with the number of parsed terms.
Moderate; still CPU-serveable.

---

## d) VLM captioning / attribute extraction + structured or text search (no vectors)

**How it works.** A vision-language model processes each image once, emitting either a
caption or (better) strict structured attributes — here the schema is
`{garments: [{type, color, color_hex, formality, material}], scene, scene_type, lighting}`,
pydantic-validated with a repair-retry loop. Payloads go into a document store or
inverted index. Queries are parsed into the same schema and retrieval is pure boolean /
BM25 matching. No embeddings anywhere.

**Strengths.**
- **Exact conjunctive filtering — compositionality is trivially correct.** "Red tie AND
  white shirt" is two predicates over garment records; a white-tie/red-shirt image
  cannot match. This is a *guarantee*, not a statistical tendency.
- **Negation is trivial** (`material != denim`), auditable, and free.
- **Serving is nearly free**: no vector index, no model at query time; DynamoDB or
  SQLite scale for pennies. Metadata is human-inspectable.

**Weaknesses — brittle recall and vocabulary lock-in.**
- **The schema is the ceiling.** Anything not extracted at index time is invisible
  forever. If the extractor labels the raincoat "coat, yellow" and the schema has no
  "raincoat" alias, the query fails with a hard zero — there is no similarity gradient
  to degrade gracefully onto. A query for "houndstooth" fails if the material
  vocabulary has seven entries.
- **Per-image extraction errors are unrecoverable.** VLM hallucination, JSON drift, a
  mislabeled color — each becomes a permanent false negative. Empirically extraction
  is imperfect enough that this codebase ships a detector-derived fallback
  (`indexer/extraction/fallback.py`) and a soft-fallback in the search path.
- **Free-text nuance collapses.** "Modern office" vs "office", "bright yellow" vs
  "yellow" — canonicalization discards the modifier that made the query specific.
- **Zero-shot capability is exactly zero outside the vocabulary**, which directly
  fails the assignment's stated zero-shot evaluation criterion.

**When it IS the right call.** Faceted-filter products with a fixed facet UI, catalog
systems with clean editorial metadata, compliance search where explainable exact-match
semantics matter more than recall.

**Cost/complexity.** Index-time dominated by the VLM pass (GPU-preferred; ~2–4 s/image
for Qwen2-VL-2B 4-bit locally). Serving trivially cheap.

---

## e) Hybrid: multi-vector + metadata prefilter + rerank (the chosen approach)

**How it works.** The union of (b), (c), and (d), plus a cross-encoder from (g),
composed so each stage covers another stage's failure mode:

1. **Index** — YOLOS proposes garment regions → FashionCLIP embeds each crop
   (marqo-fashionSigLIP config-swappable, benchmarked in `eval/ablations.py
   benchmark_embedders`) → vanilla CLIP ViT-B/32 embeds the whole image as a distinct
   *scene vector* → Qwen2-VL-2B (4-bit) extracts the strict-JSON attribute payload,
   canonicalized into a shared lexicon (`core/lexicon.py`: canonical colors with a
   large alias table, ~25 garment types, 7 scene groups, 5 styles). One image =
   N garment vectors + 1 scene vector + attributes, joined on `image_id`
   (`indexer/storage/base.py` VectorStore ABC; FAISS / OpenSearch / Atlas impls).
2. **Query** — a deterministic lexicon parser (`retriever/parsing/rule_parser.py`;
   Bedrock Haiku swappable) decomposes the query into
   `{garments:[{type,color}], scene, style, negations}` with nearest-following-garment
   color binding.
3. **Prefilter** — each parsed garment becomes an `AttributePredicate`; an image
   qualifies iff *every* predicate is satisfied by *a single garment record* (per-record
   type ∧ color ∧ material). This is the structural compositionality gate. If the hard
   filter leaves < 12 candidates (extraction is imperfect), requirements soften into a
   scoring feature while negations stay hard.
4. **Multi-vector ANN** — each garment term vs garment vectors (per-image max), scene
   text vs scene vectors.
5. **Fusion** — min-max normalized weighted sum (garment 0.55 / scene 0.25 /
   attribute 0.20, YAML-tunable), then **BLIP-ITM cross-encoder rerank** of the top-50,
   blended at 0.35.

**Why the combination covers each individual failure mode.**

| Failure mode | Introduced by | Covered by |
|---|---|---|
| Single-vector bottleneck | (a)/(b) | Per-garment region vectors (c) |
| Bag-of-words attribute binding | (a)/(b)/(c) | Conjunctive per-record predicates (d) — structural, guaranteed |
| Small-object dilution | (a)/(b) | Crops make each garment its own vector |
| Fine-grained fashion vocabulary | (a) | FashionCLIP crop encoder (b) |
| Scene context lost by fashion tuning | (b)/(c) | Separate vanilla-CLIP scene vector + scene channel weight |
| Vocabulary lock-in / brittle extraction | (d) | Vector channels stay active; attribute channel is only 0.20 of fusion; soft fallback when the hard filter starves |
| Soft cross-garment conjunction | (c) | The hard prefilter, plus predicate-satisfaction dominating the attribute score |
| Word-order-blind final ranking | (a)–(c) | BLIP-ITM cross-attention rerank over top-50 |
| No negation | (a)–(c) | Hard exclusion predicates (never softened) |
| Cross-encoder serving cost | (g) | Only ever runs on 50 candidates |

**Honest weaknesses.**
- Pipeline complexity: four models at index time, three to four at query time; more
  failure surfaces, more model downloads, more RAM.
- Errors compound: a detector miss plus an extraction miss means the prefilter can
  actively *remove* a relevant image; the soft fallback bounds but does not eliminate
  this.
- The canonical lexicon is itself a mild vocabulary lock (mitigated because the vector
  channels bypass it entirely).
- The fusion weights are hand-tuned against a small labeled set; they are exposed in
  config precisely because they are not learned (see FUTURE_WORK.md on feedback loops).
- Latency is the highest of all non-(g) options, though still comfortably interactive
  on CPU at 1K scale.

**When it IS the right call.** Compositional multi-attribute queries over a modest
corpus with an explainability requirement — i.e., this assignment.

**Cost/complexity.** Index-time GPU-heavy (VLM dominates); serving moderate on CPU.
High engineering complexity, contained by ABC boundaries (Detector, EmbeddingModel,
AttributeExtractor, VectorStore, QueryParser, Reranker are all swappable interfaces).

---

## f) Fine-tuning routes

Three distinct routes, in increasing order of specificity:

**f1 — Contrastive fine-tune of the dual encoder on fashion triplets.** Continue
CLIP-style training on (image, caption) fashion pairs, or a triplet loss over
(anchor crop, positive, negative). Sharpens vocabulary beyond FashionCLIP for *your*
image distribution. Needs 10K–100K+ curated pairs; risks catastrophic forgetting of
scene/context knowledge; and — critically — does **not** fix compositionality, because
the training objective is unchanged: in-batch negatives still never penalize
word-order confusions.

**f2 — Compositional hard negatives (NegCLIP / replace-attribute style).** Construct
negatives by swapping attributes within the caption ("red tie, white shirt" → "white
tie, red shirt") or compositing swapped images, and train the encoder to separate
them. This attacks the bag-of-words failure directly and shows real gains on
ARO/SugarCrepe-style benchmarks. But it still produces a single-vector model, so the
improvement is *statistical*, not guaranteed per query — whereas the structural
prefilter in (e) gives a hard guarantee for the same failure mode at zero training
cost. The built system's stored attribute payloads are, incidentally, a free source of
swap-negative captions if this route is ever taken (see FUTURE_WORK.md).

**f3 — Detector fine-tune.** The region pipeline's recall ceiling is the detector.
`valentinafeve/yolos-fashionpedia` *is* already a Fashionpedia fine-tune; further
tuning on observed failure cases (occlusion, unusual poses, in-the-wild lighting)
raises the ceiling of everything downstream.

**When fine-tuning is justified vs zero-shot stacking.** When (i) labeled in-domain
data exists at real scale, (ii) the query distribution is stable enough to not regress
silently, and (iii) ablations show the zero-shot stack has plateaued below the target.
None of these hold for a 1K-image assessment with five acceptance queries: the eval
set cannot even measure a fine-tune's improvement reliably (five queries ≈ guaranteed
overfitting), and the GPU budget is contractually one-shot. Zero-shot stacking (e)
reaches the acceptance criteria without any of these risks.

**Cost/complexity.** Highest of all options: data curation + GPU training + regression
eval infrastructure + model versioning.

---

## g) Cross-attention retrieval (BLIP-ITM / ColPali-style late interaction)

**How it works.** Instead of comparing two precomputed pooled vectors, the query and
image interact token-by-token. Two families: **cross-encoders** (BLIP's ITM head —
full cross-attention between text tokens and image patches, trained to answer "does
this caption match this image?") and **late interaction** (ColBERT/ColPali — store
per-patch embeddings, score by MaxSim between every query token and every patch).

**Strengths.** The quality ceiling for compositional matching: attribute–object
binding is resolved in attention patterns rather than a pooled vector, so "red tie
and white shirt" genuinely differs from its swap. ITM outputs are calibrated
match probabilities, useful for blending.

**Weaknesses — serving cost.**
- A cross-encoder cannot be indexed: every (query, image) pair requires a full model
  forward, so full-corpus scoring is O(N) forwards per query. At 1M images this is
  ~days of GPU per query. It can only ever be a **reranker** over a cheap candidate
  generator.
- Late interaction *can* be indexed, but stores hundreds of vectors per image
  (per-patch), inflating index size 100–1000× and requiring specialized MaxSim
  infrastructure.

**When it IS the right call.** As the last stage over a small candidate set — which is
exactly where the built system uses it: BLIP-ITM over the top-50 fused candidates,
blended at weight 0.35 (`rerank.*` in `config/default.yaml`). The expensive model sees
50 images, not 1,000 (or 1,000,000).

**Cost/complexity.** Prohibitive as the primary retriever; cheap and high-value as a
bounded rerank stage (CPU-tolerable at top-50).

---

## Decision matrix

| Approach | Compositionality | Fine-grained attributes | Scene / context | Zero-shot robustness | Query latency | Infra cost |
|---|---|---|---|---|---|---|
| (a) Vanilla CLIP whole-image | none (bag-of-words + single vector) | weak | good | good | ~ms | minimal |
| (b) FashionCLIP whole-image | none (same structure) | moderate | degraded vs (a) | good | ~ms | minimal |
| (c) Region embeddings | partial (per-garment binding; conjunction still soft) | strong | none alone | good | low | moderate |
| (d) VLM attributes + structured search | exact within vocabulary | strong within vocabulary | schema-limited | none outside vocabulary | ~ms | index-time GPU; serving trivial |
| (e) Hybrid multi-vector + prefilter + rerank | **structural guarantee + soft fallback** | **strong** | **dedicated channel** | **strong (vector channels bypass the lexicon)** | moderate (CPU-OK) | moderate–high |
| (f) Fine-tuning routes | statistical improvement (f2) | strong (f1) | risk of forgetting | can regress off-distribution | as base model | highest (training + data + eval) |
| (g) Cross-attention / late interaction | strongest per-pair | strong | strong | good | O(N) forwards, or 100–1000× index | prohibitive as primary |

## Why (e) wins under this assignment's constraints

The corpus is ~1K images, serving runs on a CPU-budget EC2 t3.small, and the grade
hinges on five acceptance queries that all require binding garment + color + scene —
one of them ("red tie and a white shirt in a formal setting") is unanswerable by any
single-vector method *in principle*, not just in practice. Approach (e) is the only
option that makes that binding a structural guarantee (two predicates, each satisfied
by one garment record on the same `image_id`) while keeping open-vocabulary vector
recall as a safety net when extraction is wrong, keeping scene understanding in a
dedicated channel that fashion fine-tuning would otherwise erode, and confining the
expensive cross-attention model to a top-50 rerank the CPU can afford. Fine-tuning is
strictly dominated here: it costs GPU budget the assignment restricts to one-shot
indexing, and a five-query eval set cannot even measure its improvement without
overfitting.
