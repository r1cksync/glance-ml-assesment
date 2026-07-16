# Garment-embedder benchmark (region channel only)

Same detected regions, same store mechanics, same queries/labels —
only the garment-crop embedder differs. Region-only engine
(no scene channel, no prefilter, no rerank) isolates the embedder.

| embedder | R@1 | R@5 | R@10 | MRR | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| patrickjohncyh/fashion-clip (512d) | 0.1333 | 0.2857 | 0.4143 | 0.4667 | 0.3952 |
| Marqo/marqo-fashionSigLIP (768d) | 0.1286 | 0.2571 | 0.4143 | 0.5000 | 0.3868 |

**Winner on nDCG@10: `fashion-clip`** — configured as the default `embedding.garment_backend`; swapping is one line in config/default.yaml.
