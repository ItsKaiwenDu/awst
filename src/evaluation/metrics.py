"""Evaluation diagnostics: repetition-loop detection, completion, truncation, and pass@1."""

from __future__ import annotations

import statistics
from typing import Any, Iterable


def detect_repetition_loop(
    text: str,
    min_line_repeats: int = 4,
    ngram_words: int = 8,
    min_ngram_repeats: int = 4,
) -> bool:
    """Detect degenerate repetitive loops by repeated lines or repeated word n-grams."""
    if not text:
        return False

    # 1. Consecutive or frequent identical non-trivial lines
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 16]
    if lines:
        line_counts: dict[str, int] = {}
        for ln in lines:
            line_counts[ln] = line_counts.get(ln, 0) + 1
            if line_counts[ln] >= min_line_repeats:
                return True

    # 2. Repeated word n-grams
    words = text.split()
    if len(words) >= ngram_words * min_ngram_repeats:
        counts: dict[tuple[str, ...], int] = {}
        for i in range(0, len(words) - ngram_words + 1):
            ng = tuple(words[i : i + ngram_words])
            counts[ng] = counts.get(ng, 0) + 1
            if counts[ng] >= min_ngram_repeats:
                return True

    # 3. Trailing character block repeat (e.g., same >=24-char chunk repeated >=4 times)
    tail = text[-600:]
    for chunk_len in (24, 48, 80):
        if len(tail) >= chunk_len * 4:
            suffix = tail[-chunk_len:]
            if tail.count(suffix) >= 4:
                return True

    return False


def compute_eval_metrics(
    records: list[dict[str, Any]],
    token_budgets: Iterable[int] = (256, 512, 1024),
) -> dict[str, Any]:
    """Compute primary pass@1 and secondary completion/loop/truncation/budget diagnostics."""
    n = len(records)
    if n == 0:
        raise ValueError("Cannot compute metrics on empty records list")

    correct_all = sum(1 for r in records if r["correct"])
    completed = [r for r in records if not r["truncated"]]
    truncated_n = sum(1 for r in records if r["truncated"])
    loop_n = sum(1 for r in records if r["repetition_loop"])
    invalid_n = sum(1 for r in records if not r["parse_valid"])
    gen_tokens = [int(r["generated_tokens"]) for r in records]

    budget_acc: dict[str, float] = {}
    for b in token_budgets:
        # Matched-token-budget diagnostic: counts as correct only if solved within <= b tokens
        ok_within = sum(
            1 for r in records if r["correct"] and int(r["generated_tokens"]) <= int(b)
        )
        budget_acc[f"pass_at_1_le_{b}_tok"] = ok_within / n

    # Per-subskill breakdown if present
    subskills: dict[str, dict[str, Any]] = {}
    for r in records:
        sk = r.get("subskill")
        if sk:
            subskills.setdefault(sk, []).append(r)  # type: ignore[arg-type]
    subskill_summary: dict[str, Any] = {}
    for sk, items in sorted(subskills.items()):
        m = len(items)
        subskill_summary[sk] = {
            "n": m,
            "pass_at_1": sum(1 for x in items if x["correct"]) / m,
            "completion_rate": sum(1 for x in items if not x["truncated"]) / m,
            "repetition_loop_rate": sum(1 for x in items if x["repetition_loop"]) / m,
            "avg_output_tokens": sum(int(x["generated_tokens"]) for x in items) / m,
        }

    return {
        "n": n,
        "pass_at_1": correct_all / n,
        "correct_count": correct_all,
        "completion_rate": len(completed) / n,
        "truncation_rate": truncated_n / n,
        "repetition_loop_rate": loop_n / n,
        "invalid_parse_rate": invalid_n / n,
        "completed_only_accuracy": (
            sum(1 for r in completed if r["correct"]) / len(completed) if completed else None
        ),
        "avg_output_tokens": sum(gen_tokens) / n,
        "median_output_tokens": float(statistics.median(gen_tokens)),
        "matched_token_budget": budget_acc,
        "by_subskill": subskill_summary,
    }
