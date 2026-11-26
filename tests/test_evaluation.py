"""Unit tests for answer extraction, loop detection, and evaluation metrics."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.metrics import compute_eval_metrics, detect_repetition_loop
from src.evaluation.parser import answers_match, extract_numeric_answer
from src.evaluation.runner import load_dataset_rows


def test_extract_numeric_answer_formats() -> None:
    assert extract_numeric_answer("Thus \\boxed{1,234} is the answer.") == ("1234", "boxed", True)
    assert extract_numeric_answer("Step 1...\n#### 42.00") == ("42", "gsm8k_hash", True)
    assert extract_numeric_answer("<think>wait 99</think>\nFinal answer: $150") == (
        "150",
        "post_think_final_answer_phrase",
        True,
    )
    assert answers_match("The total is \\boxed{305}.", "305")[0] is True
    assert answers_match("No digits here.", "10") == (False, None, "unparsed")


def test_repetition_loop_and_metrics() -> None:
    loop_text = "\n".join(["Let me recheck the carry step carefully."] * 5)
    assert detect_repetition_loop(loop_text) is True
    assert detect_repetition_loop("345 + 678 = 1023. \\boxed{1023}") is False

    records = [
        {
            "correct": True,
            "truncated": False,
            "repetition_loop": False,
            "parse_valid": True,
            "generated_tokens": 120,
            "subskill": "carry_add_sub",
        },
        {
            "correct": False,
            "truncated": True,
            "repetition_loop": True,
            "parse_valid": False,
            "generated_tokens": 512,
            "subskill": "carry_add_sub",
        },
    ]
    m = compute_eval_metrics(records, token_budgets=(256, 512))
    assert m["pass_at_1"] == 0.5
    assert m["completion_rate"] == 0.5
    assert m["truncation_rate"] == 0.5
    assert m["repetition_loop_rate"] == 0.5
    assert m["invalid_parse_rate"] == 0.5
    assert m["completed_only_accuracy"] == 1.0
    assert m["matched_token_budget"]["pass_at_1_le_256_tok"] == 0.5


def test_load_dataset_rows_unicode_separator_and_stratification(tmp_path: Path) -> None:
    p = tmp_path / "sample.jsonl"
    rows = [
        {"id": "1", "subskill": "a", "prompt": "Line1\u2028Line2", "answer": "1"},
        {"id": "2", "subskill": "a", "prompt": "A2", "answer": "2"},
        {"id": "3", "subskill": "b", "prompt": "B1", "answer": "3"},
        {"id": "4", "subskill": "b", "prompt": "B2", "answer": "4"},
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    loaded_all = load_dataset_rows(p)
    assert len(loaded_all) == 4
    assert "\u2028" in loaded_all[0]["prompt"]

    strat = load_dataset_rows(p, per_subskill_limit=1)
    assert [r["id"] for r in strat] == ["1", "3"]
