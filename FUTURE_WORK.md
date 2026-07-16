# Future Work

Every suggestion below is tied to an extension point that already exists in this
codebase — an ABC, a config key, or a schema field — so each is an incremental change,
not a rewrite. The relevant seams: `core/schemas.py` (data contracts),
`core/lexicon.py` (canonical vocabulary), `indexer/extraction/base.py`
(AttributeExtractor ABC + strict-JSON prompt), `indexer/storage/base.py`
(VectorStore ABC with `find_images` and `update_attributes`),
`retriever/search/engine.py` (fusion + `_attribute_score`), and `config/default.yaml`
(every weight and backend choice).

---

## 1. Extending to locations & weather

### 1.1 Geo: cities, landmarks, GPS

**GPS-EXIF ingestion.** `IndexingPipeline._index_one` (`indexer/pipeline.py`) already
opens each image with PIL; reading `GPSInfo` EXIF tags there is a ~15-line addition.
Store the result on the attribute payload: extend `ImageAttributes`
(`core/schemas.py`) with

```python
geo: Optional[GeoInfo] = None   # {lat, lon, geohash, city, landmark}
```

with a pydantic validator computing the geohash from lat/lon. Because one image is
already "N garment records + 1 scene vector + attribute payload joined on image_id",
geo is just another field on that payload — no storage redesign.

**City/landmark recognition for images without EXIF.** Two config-swappable options
behind the existing `AttributeExtractor` ABC, mirroring how `qwen2vl | moondream |
bedrock` are swapped today (`extraction.backend`): (i) extend the VLM strict-JSON
prompt in `indexer/extraction/qwen2vl.py` with `"city_guess"` / `"landmark"` keys —
same pydantic validate + repair-retry path; (ii) a dedicated head — CLIP zero-shot
against a city/landmark prompt set, or a places classifier — as a second extractor
composed in the pipeline. Canonicalize outputs through a new `GEO_LEXICON` in
`core/lexicon.py`, exactly as scenes canonicalize today.

**Geo-hash prefilters.** Extend `AttributePredicate` (`core/schemas.py`) with
`geohash_prefixes: list[str]` and teach `matches_image` prefix matching; the query
parser gains a gazetteer stage (`rule_parser.py` already resolves scene aliases — a
city gazetteer is the same regex-alternation pattern). `VectorStore.find_images`
signatures don't change: the predicate carries the geo constraint. For the
OpenSearch/Atlas backends, map the predicate to native `geo_distance` /
`$geoWithin` filters inside filtered k-NN instead of prefix matching — both impls
already exist behind the ABC.

**Retrieval semantics.** "Blue shirt in Paris" = the existing garment predicate ∧ a
geo predicate, both resolved in the same conjunctive prefilter. The soft-fallback
logic (`search.prefilter.min_candidates`) applies unchanged when geo coverage is
sparse.

### 1.2 Weather-aware attribute extraction

**Schema extension.** Add to `ImageAttributes`:

```python
weather: str = "unknown"   # sunny | rainy | snowy | overcast | foggy | unknown
season: str = "unknown"    # spring | summer | autumn | winter | unknown
```

with the same controlled-vocabulary validators as `scene_type`. Extend the VLM
strict-JSON prompt with both keys; the pydantic repair-retry loop needs no changes.

**No re-embedding required.** `VectorStore.update_attributes` exists precisely for
this: "replace ONLY the attribute payload of an already-indexed image". One re-extraction
pass with the new prompt upgrades the whole corpus while every garment and scene
vector stays untouched — this is the cheapest possible migration.

**Weather-appropriate outfit scoring.** A small compatibility table in
`core/lexicon.py` (weather × garment type/material: raincoat/umbrella ↔ rainy,
wool/coat ↔ winter, shorts/sandals ↔ summer) consumed as a new term inside
`SearchEngine._attribute_score`, which already averages predicate satisfaction +
formality fit + scene fit — weather fit is a fourth signal of the same shape. Its
influence is automatically governed by the existing `search.weights.attribute` key;
a dedicated `search.weights.weather` is a two-line change since the fusion loop
iterates named channels.

### 1.3 Temporal indexing

- **Capture:** EXIF `DateTimeOriginal` at index time → `time_of_day` and month-derived
  `season` fields on `ImageAttributes` (the free-text `lighting` field already acts as
  a weak time-of-day proxy; a categorical field makes it filterable).
- **Partitioning:** the FAISS store already persists to a directory
  (`store.persist(index_dir)`); season/time partitions become sibling index
  directories, loaded and searched per query with score-level merge. On
  OpenSearch this is index-per-partition behind an alias.
- **Freshness-decayed scoring:** multiply the fused score by `exp(-λ · age_days)`
  as a post-fusion modifier at the end of `SearchEngine._retrieve` (after step 4,
  before rerank), with λ exposed under `search.weights` like every other tunable.

---

## 2. Improving precision

### 2.1 Hard negative mining from the eval harness

`eval/ablations.py` already writes per-query rankings to
`eval/results/ablations.json`. The top-ranked *non-relevant* images per labeled query
are, by definition, the system's near-misses — the highest-value hard negatives.
Persisting them is a small extension to `eval/harness.run_eval` (log
`(query, image_id, rank, channel_scores)` for false positives above the last true
positive). Uses: (i) training pairs for 2.2, (ii) targeted lexicon/alias fixes when
the miss is an extraction error, (iii) calibration data for 2.5. The
`ComponentScores` already attached to every `SearchResult` tell you *which channel*
caused each near-miss for free.

### 2.2 Fashion-domain contrastive fine-tuning with attribute-swap negatives

The indexed corpus already contains structured attributes per image — a free caption
source: render each `GarmentAttribute` as text ("a red tie", "a white cotton shirt"),
then generate **replace-attribute negatives** by swapping color or type across
garments of the same image ("a white tie", "a red shirt"). Fine-tune the garment
encoder (LoRA on FashionCLIP) with these NegCLIP-style hard negatives to directly
attack bag-of-words binding *inside* crops. Integration is already zero-friction:
add the tuned model as a new entry under `embedding.models` in `config/default.yaml`,
point `embedding.garment_backend` at it, and compare against the incumbents with
`AblationRunner.benchmark_embedders`, which evaluates any embedder against its own
prebuilt store. The structural prefilter stays as the guarantee; fine-tuning only
sharpens the soft ranking beneath it.

### 2.3 LLM listwise reranking

The current BLIP-ITM reranker scores each (query, image) pair independently — it
cannot reason *across* candidates. A listwise upgrade: feed the top-20 candidates'
garment crops (the bboxes are in each `SearchResult.matches`) plus their attribute
payloads (`SearchResult.attributes`) to a VLM judge that returns a permutation.
Implement it as a new `Reranker` behind `retriever/rerank/base.py` — the engine
already treats the reranker as injected and optional, and `rerank.backend` in config
selects it. The Bedrock client pattern already exists (`indexer/extraction/bedrock.py`,
`retriever/parsing/bedrock_parser.py`); cost is contained because it only ever sees
the final candidate list, and results can be cached on
`(query, hash(candidate_ids))` using the same DynamoDB cache the parser uses.

### 2.4 User feedback loops for the fusion weights

The fusion weights (`search.weights`: garment 0.55 / scene 0.25 / attribute 0.20) and
the rerank blend (`rerank.weight: 0.35`) are hand-tuned and deliberately
config-exposed — which makes them directly learnable without code changes to the
engine:

- **Signal:** log `(parsed_query, shown_results, clicked_result)` in the API layer
  (DynamoDB, same always-free table pattern as the parse cache).
- **Objective:** click-through-derived nDCG proxy per weight vector.
- **Optimizer:** the weight simplex is 3–4 dimensional — Bayesian optimization or a
  simple bandit over a discretized grid converges with hundreds, not millions, of
  interactions. Each arm is just a different YAML weight vector.
- **Refinement:** learn *per-query-type* weights. `ParsedQuery` already labels which
  channels a query activates (`has_garment_terms()`, `scene`, `style`), and the engine
  already renormalizes over active channels — garment-heavy vs scene-heavy queries
  plausibly want different simplex points.

### 2.5 Calibrating the ITM rerank blend

`final = (1-w)·fusion + w·itm` currently blends a min-max-normalized fusion score
against a raw ITM probability — two quantities on different scales, joined by a
hand-set `w = 0.35`. Two cheap fixes: (i) Platt/isotonic-calibrate the ITM
probabilities on the labeled eval set (the near-miss log from 2.1 provides the
negatives); (ii) sweep `w` per ablation variant with `eval/ablations.py` and commit
the winning value with its nDCG delta. Both are pure config/eval work against
`rerank.weight`.

---

## 3. Scaling to 1M images

### 3.1 ANN index migration: the math

At 1M images with ~5 kept regions each (config caps at 8, observed mean is lower):

| Quantity | Value |
|---|---|
| Garment vectors | ~5M × 512-d fp32 ≈ **10.2 GB** |
| Scene vectors | 1M × 512-d fp32 ≈ **2.0 GB** |
| Flat exact IP per garment term | ~5M × 512 MACs ≈ 2.6 GFLOP/query/term — seconds on CPU, per term |

Flat exact search (`storage.faiss.index_type: flat_ip`, correct at 1K) is the first
casualty. Two migration targets, both anticipated by that config key:

- **HNSW** (`IndexHNSWFlat`, M=16–32): ~1.2–1.5× RAM over flat (link overhead),
  sub-millisecond queries at recall ≈ 0.95–0.99. Right when RAM is affordable
  (~15 GB total → an r-class instance).
- **IVF-PQ** (e.g. IVF16384 + PQ64): 64 B/vector → 5M vectors ≈ **320 MB** of codes
  (~32× compression), millisecond queries at nprobe 16–64, small recall hit
  (recoverable by re-scoring the top-200 with exact vectors, which the current
  two-pass design in `SearchEngine._retrieve` already resembles). Right for the
  t3.small-class budget.

### 3.2 Push the prefilter into native filtered ANN

The FAISS path materializes `find_images(...)` as a Python `set[str]` and passes it
as `allowed_ids` — fine at 1K, fatal at 1M (a permissive predicate yields a
hundreds-of-thousands-entry set per query, and post-filtered ANN wastes candidates).
The fix is architectural but already staged: the OpenSearch k-NN and Atlas Vector
Search implementations exist behind the same `VectorStore` ABC, and both engines
support **native filtered k-NN** (Lucene HNSW filter-aware traversal;
`$vectorSearch.filter` in Atlas). The work is mapping `AttributePredicate` to each
engine's filter DSL inside those impls' `search_garments`/`find_images` — the engine
code, predicates, and `storage.backend` config flip don't change. Garment-level
records (one document per region, joined on `image_id`) map naturally to both
engines' document models; the per-record type∧color conjunction becomes a `bool`
filter on the region document, preserving the structural binding guarantee at scale.

### 3.3 Extraction throughput

Qwen2-VL-2B 4-bit locally runs ~2–4 s/image — 1M images ≈ 35–70 GPU-days serially:
infeasible. Two staged options, both consistent with the project's cost rules:

- **One-shot GPU batch (the g4dn path):** the deployment already scripts a
  self-terminating g4dn.xlarge (trap/finally teardown) for indexing. Swap the
  per-image HF generate loop for vLLM continuous batching → ~10–20 img/s ≈ 14–28
  GPU-hours ≈ **$8–15 total** at $0.526/hr. Shard the image list across N sequential
  spot runs; `IndexingPipeline.index_directory(resume=True)` already makes every run
  resumable, and `persist_every` bounds loss on interruption.
- **Bedrock extractor at concurrency:** `indexer/extraction/bedrock.py` already
  implements the same `AttributeExtractor` contract; pay-per-token Haiku with high
  request concurrency needs no GPU at all — the tradeoff is per-image token cost vs
  GPU-hour cost, and it wins for incremental top-ups after the initial batch.

Detection + crop embedding are lighter (~10× faster than the VLM) and batch on the
same GPU pass; `embedding.batch_size` is already a config key.

### 3.4 Index sharding

- **Build:** shard by image-id hash into ~16 FAISS index directories.
  `VectorStore.persist(path)/load(path)` already treats an index as a directory, so a
  `ShardedStore` composite implementing the same ABC (fan-out `search_garments`,
  merge by score) requires no engine changes. Raw cosine similarities are comparable
  across shards; the min-max normalization in `SearchEngine._retrieve` already
  happens *after* candidate collection, so it naturally becomes the post-merge step.
- **Serve:** shards are memory-mapped FAISS files synced from the existing
  `vectors/` S3 prefix; scene index (2 GB flat, or ~130 MB as PQ) can stay resident.
- **Attributes:** the attribute payload store moves from the in-process dict to
  DynamoDB keyed on `image_id` (already the API-layer pattern), or lives inside
  OpenSearch/Atlas documents in the 3.2 path — in which case sharding, filtering,
  and replication all come from the engine and the composite store is unnecessary.

The load-bearing property throughout: ML logic never touches storage internals — every
scaling change above lands behind `VectorStore`, a config key, or a schema field.
