"""Unit tests for deterministic synthetic arithmetic splits and OOD template isolation."""

from __future__ import annotations

from src.data.synthetic_arithmetic import (
    ID_WORD_TEMPLATES,
    OOD_WORD_TEMPLATES,
    count_borrows,
    count_carries,
    generate_split,
)


def test_carry_and_borrow_counters() -> None:
    assert count_carries(199, 1) == 2
    assert count_carries(123, 456) == 0
    assert count_borrows(1000, 1) == 3
    assert count_borrows(543, 121) == 0


def test_synthetic_splits_are_deterministic_and_disjoint() -> None:
    id_texts = {t for t, _ in ID_WORD_TEMPLATES}
    ood_texts = {t for t, _ in OOD_WORD_TEMPLATES}
    assert id_texts.isdisjoint(ood_texts)

    dev_a = generate_split("dev", seed=1001, per_subskill=25)
    dev_b = generate_split("dev", seed=1001, per_subskill=25)
    assert dev_a == dev_b
    assert len(dev_a) == 75

    seen = {r["prompt"] for r in dev_a}
    val = generate_split("val", seed=2002, per_subskill=25, seen_prompts=seen)
    seen.update(r["prompt"] for r in val)
    test = generate_split("test", seed=3003, per_subskill=25, seen_prompts=seen)
    seen.update(r["prompt"] for r in test)
    ood = generate_split("ood_test", seed=4004, per_subskill=25, ood_word_only=True, seen_prompts=seen)

    assert len({r["prompt"] for r in dev_a}.intersection({r["prompt"] for r in val})) == 0
    assert len({r["prompt"] for r in dev_a}.intersection({r["prompt"] for r in test})) == 0
    assert len({r["prompt"] for r in val}.intersection({r["prompt"] for r in test})) == 0
    assert all(r["meta"]["ood"] is True for r in ood)
