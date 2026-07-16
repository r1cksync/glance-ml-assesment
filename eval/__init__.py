"""Evaluation package — first-class module of the retrieval system.

Contents:
- eval.metrics     pure ranking metrics (Recall@k, MRR, nDCG@k)
- eval.harness     engine-agnostic runner over queries + labels
- eval.ablations   the ablation ladder (vanilla CLIP -> full engine)
- eval.latency     p50/p95 benchmark + 1M-image scaling projection
- eval.label_pool  labeling-pool builder + weak auto-labels

Results are committed under eval/results/.
"""
