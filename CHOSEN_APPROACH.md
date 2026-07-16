# Chosen Approach

**One sentence**: garment-region FashionCLIP vectors + a separate whole-image scene
vector + VLM-extracted structured attributes, joined on `image_id`, queried through
a parse → conjunctive-prefilter → multi-vector-ANN → weighted-fusion → rerank
pipeline — so compositional binding is enforced by *data structure*, not learned
by an embedding.

## What was built

### Indexing (one-shot, GPU)

| Stage | Choice | Why this one |
|---|---|---|
| Region proposal | YOLOS fine-tuned on Fashionpedia (`valentinafeve/yolos-fashionpedia`) | Emits *garment-vocabulary* classes ("shirt, blouse", "tie") rather than COCO's person/handbag; garment-part labels (sleeve, collar) are filtered out through the shared lexicon |
| Garment embedding | FashionCLIP (`patrickjohncyh/fashion-clip`) on padded crops | CLIP fine-tuned on 800K fashion product pairs — much sharper on fine-grained garment attributes than base CLIP; `Marqo/marqo-fashionSigLIP` implemented behind the same interface and benchmarked (see `eval/results/`) |
| Scene embedding | OpenAI CLIP ViT-B/32 on the **whole image**, kept as a **separate vector** | Environment/context ("office", "park") is global signal; keeping it out of garment vectors means garment similarity is never diluted by background, and the fusion weight between the two is an explicit, tunable number |
| Attribute extraction | Qwen2-VL-2B-Instruct, 4-bit NF4, strict-JSON prompt | Small enough for a 4 GB laptop GPU; emits `{garments:[{type,color,color_hex,formality,material}], scene, scene_type, lighting}`; pydantic-validated with a repair-retry loop **plus a truncation-salvage parser**; every value canonicalized into a shared lexicon (24 colors with hex anchors, 26 garment types, 7 scene types, 5 formality levels). Moondream2 and Bedrock Claude Haiku are drop-in config swaps |
| Storage | FAISS (exact IP) multi-vector store | One image = N garment vectors + 1 scene vector + attribute payload. At 1,158 images / ~4.5K vectors a managed vector DB is pure overhead; OpenSearch k-NN and Atlas Vector Search implementations exist behind the same `VectorStore` ABC for the 1M-image regime |

### Retrieval (per query, CPU-friendly)

1. **Parse** — deterministic lexicon parser (Bedrock Claude Haiku swappable; parsed
   queries cached in DynamoDB): `"a red tie and a white shirt, no denim"` →
   `{garments:[{tie,red},{shirt,white}], negations:[{material:denim}], style:formal}`.
   The binding rule (color attaches to the nearest following garment within its
   noun phrase) is what keeps *red shirt / blue pants* ≠ *blue shirt / red pants*.
2. **Prefilter** — each parsed garment becomes a conjunctive predicate that must be
   satisfied by a **single garment record** of an image. This is where
   compositionality is enforced: an image with a *white* tie and a *red* shirt has
   no record satisfying `{tie, red}`, so it is gone before any vector math.
   If the hard filter leaves `< min_candidates` (extraction is imperfect),
   requirements soften into a scoring feature — but negations stay hard.
3. **Multi-vector ANN** — each garment term is embedded ("a photo of a person
   wearing red tie") and scored against garment vectors (per-image max);
   the scene text against scene vectors.
4. **Fusion** — min-max normalized channels combined with configurable weights
   (`garment 0.55 / scene 0.25 / attribute 0.20`); the attribute channel scores
   predicate satisfaction + formality compatibility + scene-type agreement.
   Weights are YAML config — tuned on the eval set, and defensible: garment
   identity is the dominant signal in all five acceptance queries, scene decides
   between garment-equivalent candidates, attributes reward exact structured hits.
5. **Rerank** — BLIP-ITM cross-encoder over the top-50, blended at weight 0.35.
   A quality stage that needs GPU-tier compute: it runs in local/eval mode and is
   deliberately **off** on the t3.small deployment (15–40 s/query on 2 vCPUs is
   not a latency/quality tradeoff, it is a mistake). The API still accepts
   `use_rerank` per request for capable deployments.

### Explainability (the proof the compositional logic is real)

Every result carries `matches: [{region_id, bbox, label, query_term, similarity}]` —
the frontend draws each query term's best-matching detected region as a colored
box over the image. You can *see* that "red tie" matched the tie region and
"white shirt" matched the shirt region, with scores.

## Measured results

Ablation ladder on the 5 acceptance queries, hand-labeled relevance sets
(pooled-candidate contact sheets, human-judged; corpus = 1,158 Fashionpedia val
+ 45 CC-licensed supplements for thin categories):

| variant | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| vanilla CLIP ViT-B/32, whole image | 0.062 | 0.261 | 0.451 | 0.549 | 0.413 |
| FashionCLIP, whole image | 0.087 | 0.241 | 0.606 | 0.633 | 0.494 |
| FashionCLIP, garment regions | 0.158 | 0.326 | 0.510 | 0.707 | 0.504 |
| + attribute prefilter & fusion | 0.102 | 0.422 | **0.683** | 0.840 | 0.671 |
| **+ BLIP-ITM rerank (full)** | **0.202** | **0.489** | 0.659 | **1.000** | **0.760** |

Reading the ladder:
- **Fashion-domain fine-tuning helps** (+0.08 nDCG over vanilla CLIP) but is
  nowhere near sufficient — whole-image vectors still can't bind attributes.
- **Regions raise precision** (MRR .63→.71): garment similarity is now measured
  on garment pixels.
- **The structural stage is the big jump** (nDCG .50→.67, MRR→.84): conjunctive
  attribute predicates + scene fusion is what actually answers compositional
  and contextual queries.
- **Rerank buys top-rank quality** (MRR→1.00: the #1 result is relevant for
  *every* acceptance query; R@1 doubles) at the cost of a little tail recall
  (R@10 .68→.66) — exactly the precision/recall trade a top-heavy cross-encoder
  should make.

Companion numbers: FashionCLIP vs marqo-fashionSigLIP in
[eval/results/embedder_benchmark.md](eval/results/embedder_benchmark.md),
[latency + 1M-image scaling math](eval/results/latency.md)
(live deployment: p50 811 ms / p95 988 ms end-to-end).

The compositionality guarantee is also a unit test
(`tests/test_compositionality.py`): swapped color-garment bindings return
different rankings, and the structural decoy (white tie + red shirt) is excluded
outright by the prefilter.

## Shortcomings — and the honest fixes

| Shortcoming | Consequence | Fix (see FUTURE_WORK.md) |
|---|---|---|
| Attribute quality is bounded by a 2B VLM at 4-bit | occasional wrong colors/types → prefilter can drop a relevant image (soft-fallback bounds the damage but costs precision) | bigger extractor (Claude Haiku via Bedrock is one config line + a payment method), or ensemble detector-labels × VLM-labels |
| Detector recall ceiling (~0.30 confidence floor) | a garment YOLOS misses never gets a region vector; retrieval then leans on scene vector + attributes | lower-threshold second pass, or Grounding-DINO open-vocabulary proposals |
| Lexicon coverage | out-of-vocabulary phrasings ("cerulean windcheater") degrade the rule parser to partial parses | the Bedrock parser handles long-tail phrasing; lexicon is data, so extending it is a PR without code changes |
| Rerank disabled on the demo box | loses a few points of nDCG vs the eval configuration | int8-ONNX rerank of top-12, or a small GPU tier |
| Scene channel is whole-image CLIP | struggles with subtle scene types ("home" vs "studio") | Places365-style scene head, or VLM scene_type as a filterable attribute (already stored) |
| Single-node FAISS | fine to ~100K images; not beyond | HNSW/IVF-PQ or the OpenSearch/Atlas impls behind the existing ABC — migration is config, not refactor |

## Design principles kept

- **ML logic imports no boto3** — AWS adapters live in `api/`, `scripts/`, and two
  clearly-marked Bedrock modules; the retrieval engine runs identically on a
  laptop and in the cloud.
- **Everything is a swappable backend** — detector, embedders, extractor, store,
  parser, reranker are all ABC + YAML config; the eval harness exploits this to
  run its ablations.
- **Local-first proof** — the pipeline, eval harness and compositionality test ran
  on a 50-image subset before a single AWS resource was provisioned.
