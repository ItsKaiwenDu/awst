"""Seeded deterministic synthetic arithmetic generator with disjoint OOD templates."""

from __future__ import annotations

import hashlib
import random
from typing import Any


NAMES = [
    "Maya", "Liam", "Noah", "Sofia", "Ethan", "Zoe", "Lucas", "Aria",
    "Mateo", "Chloe", "Owen", "Nora", "Caleb", "Elena", "Isaac", "Leah",
]

ITEMS = [
    ("stickers", "notebooks"),
    ("apples", "oranges"),
    ("marbles", "cards"),
    ("pencils", "erasers"),
    ("books", "bookmarks"),
    ("tokens", "coins"),
    ("shells", "pebbles"),
    ("pages", "chapters"),
]

# 10 in-distribution templates (used for dev, val, and ID test)
ID_WORD_TEMPLATES: list[tuple[str, str]] = [
    (
        "{name} starts with {a} {item1}. They collect {b} more {item1}, then give away {c} {item1}. How many {item1} does {name} have now?",
        "add_sub",
    ),
    (
        "A school library has {a} {item1} on Monday. On Tuesday, {b} new {item1} arrive, and {c} {item1} are checked out. How many {item1} remain?",
        "add_sub",
    ),
    (
        "{name} packs {a} boxes with {b} {item1} in each box, and then adds {c} loose {item1}. How many {item1} are there in total?",
        "mul_add",
    ),
    (
        "Each of {a} classrooms receives {b} {item1}. After distributing them, {c} {item1} are damaged and removed. How many good {item1} are left?",
        "mul_sub",
    ),
    (
        "{name} saved ${a} in January and ${b} in February, then spent ${c} on {item1}. How much money does {name} have left?",
        "add_sub",
    ),
    (
        "There are {a} trays with {b} {item1} per tray. {name} puts {c} extra {item1} onto a spare tray. What is the total number of {item1}?",
        "mul_add",
    ),
    (
        "A warehouse holds {a} {item1}. A shipment removes {c} {item1}, and later {b} {item1} are restocked. How many {item1} are in the warehouse?",
        "sub_add",
    ),
    (
        "{name} buys {a} packs of {item1}, each containing {b} {item1}. If {name} shares {c} {item1} with friends, how many {item1} does {name} keep?",
        "mul_sub",
    ),
    (
        "A bus carries {a} passengers. At the first stop, {c} passengers get off and {b} passengers board. How many passengers are on the bus now?",
        "sub_add",
    ),
    (
        "{name} earned {a} points in round one and {b} points in round two, then lost a {c}-point penalty. What is {name}'s final score?",
        "add_sub",
    ),
]

# 6 strictly held-out OOD templates (used ONLY for isolated ood_test)
OOD_WORD_TEMPLATES: list[tuple[str, str]] = [
    (
        "During a science fair, {a} booths each displayed {b} {item1}, while the welcome table held {c} more {item1}. Altogther, how many {item1} were displayed?",
        "mul_add",
    ),
    (
        "An orchard harvested {a} crates in the morning and {b} crates in the afternoon, then sold {c} crates before closing. How many crates remained unsold?",
        "add_sub",
    ),
    (
        "{name} printed {a} booklets with {b} pages per booklet, then trimmed out {c} blank pages across the batch. How many printed pages remained?",
        "mul_sub",
    ),
    (
        "A train departed with {a} parcels, unloaded {c} parcels at Midway Station, and loaded {b} parcels at Oakridge. How many parcels reached the final stop?",
        "sub_add",
    ),
    (
        "To prepare gift bags, {name} combined {a} {item1} from one drawer with {b} {item1} from another, then set aside {c} {item1} for next week. How many {item1} went into the bags?",
        "add_sub",
    ),
    (
        "A volunteer stacked {a} rows of {b} chairs in the auditorium and brought out {c} folding chairs along the wall. How many chairs were set up in total?",
        "mul_add",
    ),
]


def count_carries(a: int, b: int) -> int:
    """Count base-10 addition carries for a + b."""
    carries = 0
    carry = 0
    while a > 0 or b > 0:
        s = (a % 10) + (b % 10) + carry
        if s >= 10:
            carries += 1
            carry = 1
        else:
            carry = 0
        a //= 10
        b //= 10
    return carries


def count_borrows(a: int, b: int) -> int:
    """Count base-10 subtraction borrows for a - b (assuming a >= b >= 0)."""
    assert a >= b >= 0
    borrows = 0
    borrow = 0
    while a > 0 or b > 0:
        da = (a % 10) - borrow
        db = b % 10
        if da < db:
            borrows += 1
            borrow = 1
        else:
            borrow = 0
        a //= 10
        b //= 10
    return borrows


def _sample_carry_add_sub(rng: random.Random) -> dict[str, Any]:
    op = rng.choice(["+", "-"])
    digits = rng.choice([3, 4, 5])
    lo = 10 ** (digits - 1)
    hi = (10**digits) - 1
    while True:
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        if op == "+":
            if count_carries(a, b) >= 2:
                return {
                    "subskill": "carry_add_sub",
                    "prompt": f"Compute the exact integer value: {a} + {b}",
                    "answer": str(a + b),
                    "meta": {"op": "+", "digits": digits, "carries": count_carries(a, b)},
                }
        else:
            if a < b:
                a, b = b, a
            if a != b and count_borrows(a, b) >= 2:
                return {
                    "subskill": "carry_add_sub",
                    "prompt": f"Compute the exact integer value: {a} - {b}",
                    "answer": str(a - b),
                    "meta": {"op": "-", "digits": digits, "borrows": count_borrows(a, b)},
                }


def _sample_mult(rng: random.Random) -> dict[str, Any]:
    variant = rng.choice(["2x2", "3x1"])
    if variant == "2x2":
        a = rng.randint(12, 99)
        b = rng.randint(12, 99)
    else:
        a = rng.randint(101, 999)
        b = rng.randint(3, 9)
    return {
        "subskill": "mult_2x2_3x1",
        "prompt": f"Compute the exact product: {a} * {b}",
        "answer": str(a * b),
        "meta": {"variant": variant},
    }


def _sample_word_2step(rng: random.Random, use_ood: bool) -> dict[str, Any]:
    pool = OOD_WORD_TEMPLATES if use_ood else ID_WORD_TEMPLATES
    tmpl_idx = rng.randrange(len(pool))
    tmpl, kind = pool[tmpl_idx]
    name = rng.choice(NAMES)
    item1, item2 = rng.choice(ITEMS)

    if kind == "add_sub":
        a = rng.randint(25, 450)
        b = rng.randint(18, 350)
        c = rng.randint(10, a + b - 5)
        ans = a + b - c
    elif kind == "sub_add":
        a = rng.randint(60, 500)
        c = rng.randint(15, a - 10)
        b = rng.randint(12, 300)
        ans = a - c + b
    elif kind == "mul_add":
        a = rng.randint(4, 24)
        b = rng.randint(6, 35)
        c = rng.randint(5, 90)
        ans = a * b + c
    elif kind == "mul_sub":
        a = rng.randint(4, 24)
        b = rng.randint(6, 35)
        c = rng.randint(5, a * b - 4)
        ans = a * b - c
    else:
        raise ValueError(f"Unknown word template kind: {kind}")

    prompt = tmpl.format(name=name, item1=item1, item2=item2, a=a, b=b, c=c)
    return {
        "subskill": "word_2step",
        "prompt": prompt,
        "answer": str(ans),
        "meta": {"template_kind": kind, "template_idx": tmpl_idx, "ood": use_ood},
    }


def generate_split(
    split_name: str,
    seed: int,
    per_subskill: int,
    ood_word_only: bool = False,
    seen_prompts: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate a deterministic, deduplicated synthetic arithmetic split."""
    rng = random.Random(seed)
    seen = set(seen_prompts) if seen_prompts is not None else set()
    records: list[dict[str, Any]] = []

    subskills = ["word_2step"] if ood_word_only else ["carry_add_sub", "mult_2x2_3x1", "word_2step"]
    for subskill in subskills:
        count = 0
        while count < per_subskill:
            if subskill == "carry_add_sub":
                item = _sample_carry_add_sub(rng)
            elif subskill == "mult_2x2_3x1":
                item = _sample_mult(rng)
            else:
                item = _sample_word_2step(rng, use_ood=ood_word_only)
            prompt = item["prompt"]
            if prompt in seen:
                continue
            seen.add(prompt)
            item_id = f"{split_name}_{subskill}_{count:04d}"
            item_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
            records.append(
                {
                    "id": item_id,
                    "split": split_name,
                    "subskill": item["subskill"],
                    "prompt": prompt,
                    "answer": item["answer"],
                    "prompt_hash": item_hash,
                    "meta": item["meta"],
                }
            )
            count += 1
    return records
