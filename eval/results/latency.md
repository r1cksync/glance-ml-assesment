# Latency benchmark

## Measured — deployed system @ 1,158 images / 4,520 garment vectors

End-to-end `POST /search` through CloudFront → EC2 t3.small (2 vCPU, CPU-only
inference, rerank off), the 5 acceptance queries × 5 runs each, measured
2026-07-16 from a residential connection:

| metric | value |
|---|---|
| p50 | **811 ms** |
| p95 | **988 ms** |
| mean | 823 ms |
| min / max | 700 / 1166 ms |

Server-side composition: query text encoding (2 CLIP text passes, ~250–400 ms on
2 vCPUs) dominates; FAISS exact search over 4.5K vectors is <1 ms; prefilter,
fusion and attribute scoring are microseconds; the remainder is network + TLS.
The CloudWatch dashboard tracks the server-side `LatencyMs` EMF metric (p50/p95)
continuously.

## Scaling projection — 1,000,000 images (computed, not measured)

Assumptions: 4 garment regions/image → 4,000,000 garment vectors (+ 1,000,000 scene vectors), dim=512, float32; ~20 GFLOP/s effective single-core CPU.

### Index size (garment channel — the big one)

| structure | size | math |
| --- | --- | --- |
| FAISS flat (exact IP) | 8.19 GB | 4,000,000 × 512 × 4 B |
| + scene flat index | 2.05 GB | 1,000,000 × 512 × 4 B |
| HNSW (M=32) | 13.31 GB | 1.5× vector storage + 32×8 B links/vec (1.02 GB) |
| IVF-PQ (m=64) | 0.30 GB | 64 B codes + 8 B ids per vec + 4,096×512×4 B centroids (~28× smaller than flat) |

### Query cost (per probe vector, garment channel)

| method | distance evals | FLOPs | est. CPU time |
| --- | --- | --- | --- |
| brute-force flat | 4,000,000 | 4.10 GFLOP | ~205 ms |
| HNSW (ef=128) | ~2,807 (ef × log2 n) | 2.9 MFLOP | ~0.14 ms math + graph overhead ≈ 1–3 ms |

A typical parsed query makes ~3 ANN probes (garment terms + scene). Projected p95 for the vector stage alone: flat ≈ 922 ms (unacceptable at 1M), HNSW ≈ 2–10 ms. End-to-end p95 is then dominated by query text encoding (~10–30 ms CPU) and the optional BLIP-ITM rerank (~100+ ms/image on CPU — cap `rerank.top_n` or serve int8-ONNX/GPU).

### Recommendation

| regime | index | why |
| --- | --- | --- |
| ≤ ~0.5M vectors (≈100K images) | FAISS `flat_ip` (current config) | ≤ ~1.0 GB, exact recall, brute force stays low-ms with BLAS batching — zero ops |
| 0.5M–50M vectors, one node | FAISS HNSW (M=32, ef≈128) | ~13.31 GB RAM at 4M vecs, ms-level p95, recall ≥ 0.95 typical |
| RAM-bound single node | FAISS IVF-PQ (m=64) | 0.30 GB vs 8.19 GB (~28×); recall loss is mitigated by our rerank stage |
| multi-node / HA / filtered ANN at scale | OpenSearch k-NN or Atlas Vector Search | sharding + replication + metadata-prefilter pushdown next to the vectors; managed ops (both already behind our VectorStore ABC) |

At 1,000,000 images the attribute payloads (~1 KB each ≈ 1.02 GB) should also move from in-process dicts to DynamoDB/OpenSearch documents so the prefilter becomes an indexed query.
