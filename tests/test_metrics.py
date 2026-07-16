"""Exact-value unit tests for eval.metrics (recall@k, MRR, nDCG@k, means)."""
from __future__ import annotations

import math

from eval.metrics import mean_metrics, mrr, ndcg_at_k, recall_at_k

# ── recall@k ─────────────────────────────────────────────────────────────────


def test_recall_exact_values() -> None:
    relevant = {"a", "b", "c", "d"}
    ranked = ["a", "x", "b", "y"]
    assert recall_at_k(relevant, ranked, 1) == 0.25
    assert recall_at_k(relevant, ranked, 2) == 0.25
    assert recall_at_k(relevant, ranked, 3) == 0.5
    assert recall_at_k(relevant, ranked, 4) == 0.5


def test_recall_empty_relevant_is_zero() -> None:
    assert recall_at_k(set(), ["a", "b"], 5) == 0.0


def test_recall_k_exceeds_ranked_length() -> None:
    assert recall_at_k({"a"}, ["x", "a"], 10) == 1.0
    assert recall_at_k({"a", "b"}, ["a"], 10) == 0.5


def test_recall_nonpositive_k_is_zero() -> None:
    assert recall_at_k({"a"}, ["a"], 0) == 0.0
    assert recall_at_k({"a"}, ["a"], -3) == 0.0


def test_recall_duplicates_not_double_counted() -> None:
    # ["a", "a", "b"] dedupes to ["a", "b"], so both relevant ids fit in top-2.
    assert recall_at_k({"a", "b"}, ["a", "a", "b"], 2) == 1.0


def test_recall_empty_ranked_is_zero() -> None:
    assert recall_at_k({"a"}, [], 5) == 0.0


# ── MRR ──────────────────────────────────────────────────────────────────────


def test_mrr_first_hit_positions() -> None:
    assert mrr({"a"}, ["a", "b", "c"]) == 1.0
    assert mrr({"b"}, ["a", "b", "c"]) == 0.5
    assert math.isclose(mrr({"c"}, ["a", "b", "c"]), 1.0 / 3.0, rel_tol=1e-12)


def test_mrr_no_hit_is_zero() -> None:
    assert mrr({"z"}, ["a", "b", "c"]) == 0.0


def test_mrr_empty_inputs() -> None:
    assert mrr(set(), ["a"]) == 0.0
    assert mrr({"a"}, []) == 0.0


def test_mrr_uses_first_relevant_only() -> None:
    assert mrr({"b", "c"}, ["a", "b", "c"]) == 0.5


# ── nDCG@k ───────────────────────────────────────────────────────────────────


def test_ndcg_perfect_ranking_is_exactly_one() -> None:
    assert ndcg_at_k({"a", "b"}, ["a", "b", "x"], 10) == 1.0
    assert ndcg_at_k({"a"}, ["a"], 1) == 1.0


def test_ndcg_perfect_prefix_with_more_relevant_than_k() -> None:
    # ideal DCG truncates at k, so a perfect length-k prefix scores 1.0.
    assert ndcg_at_k({"a", "b", "c"}, ["a", "b"], 2) == 1.0


def test_ndcg_exact_value() -> None:
    # hits at ranks 2 and 4 of the top-4; ideal packs them at ranks 1 and 2.
    got = ndcg_at_k({"a", "b"}, ["x", "a", "y", "b"], 4)
    dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert math.isclose(got, dcg / idcg, rel_tol=1e-12)
    assert math.isclose(got, 0.65096, abs_tol=1e-4)  # human-readable anchor


def test_ndcg_no_relevant_is_zero() -> None:
    assert ndcg_at_k(set(), ["a", "b"], 5) == 0.0


def test_ndcg_nonpositive_k_is_zero() -> None:
    assert ndcg_at_k({"a"}, ["a"], 0) == 0.0


def test_ndcg_relevant_outside_top_k_is_zero() -> None:
    assert ndcg_at_k({"z"}, ["a", "b", "z"], 2) == 0.0


def test_ndcg_k_exceeds_ranked_length() -> None:
    # single relevant id at rank 2; ideal puts it at rank 1 (idcg = 1.0).
    got = ndcg_at_k({"a"}, ["x", "a"], 10)
    assert math.isclose(got, 1.0 / math.log2(3), rel_tol=1e-12)


# ── mean_metrics ─────────────────────────────────────────────────────────────


def test_mean_metrics_exact() -> None:
    rows = [{"recall@1": 1.0, "mrr": 0.5}, {"recall@1": 0.0, "mrr": 1.0}]
    assert mean_metrics(rows) == {"recall@1": 0.5, "mrr": 0.75}


def test_mean_metrics_empty_is_empty() -> None:
    assert mean_metrics([]) == {}


def test_mean_metrics_heterogeneous_keys_average_defining_rows_only() -> None:
    rows = [{"a": 1.0}, {"a": 0.0, "b": 1.0}]
    assert mean_metrics(rows) == {"a": 0.5, "b": 1.0}
