# Ablation ladder

Mean metrics over the labeled acceptance queries (see `ablations.json` for per-query detail).

| variant | R@1 | R@5 | R@10 | MRR | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| vanilla-clip-whole | 0.0619 | 0.2607 | 0.4511 | 0.5486 | 0.4132 |
| fashionclip-whole | 0.0869 | 0.2405 | 0.6060 | 0.6333 | 0.4940 |
| fashionclip-regions | 0.1583 | 0.3261 | 0.5104 | 0.7067 | 0.5035 |
| regions+attr-filter | 0.1023 | 0.4223 | 0.6826 | 0.8400 | 0.6706 |
| full+rerank | 0.2023 | 0.4889 | 0.6588 | 1.0000 | 0.7600 |

Variant definitions:
- **vanilla-clip-whole** — whole-image CLIP ViT-B/32, cosine vs raw query text (baseline)
- **fashionclip-whole** — whole-image FashionCLIP, cosine vs raw query text
- **fashionclip-regions** — engine, garment-region channel only (no scene, no attributes, no prefilter, no rerank)
- **regions+attr-filter** — engine with configured fusion weights + conjunctive attribute prefilter (no rerank)
- **full+rerank** — full engine incl. BLIP-ITM cross-modal rerank
