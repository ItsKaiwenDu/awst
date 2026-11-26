"""Deterministic numeric answer extraction and normalization for math evaluation."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


_HASH_RE = re.compile(r"####\s*([^\n]+)")
_FINAL_RE = re.compile(r"(?:final\s+answer|answer)\s*(?:is|:)\s*([^\n]+)", re.IGNORECASE)
_NUM_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def normalize_numeric_string(raw: str) -> str | None:
    """Normalize an extracted numeric string (strip $, commas, trailing .0) or return None."""
    s = raw.strip()
    if not s:
        return None
    # Grab the first clean number inside the candidate span if extra punctuation trails it
    m = _NUM_RE.search(s.replace("$", "").replace(" ", ""))
    if not m:
        return None
    token = m.group(0).replace(",", "")
    try:
        d = Decimal(token)
    except InvalidOperation:
        return None
    if d == d.to_integral_value():
        return str(int(d))
    # Normalize non-integer decimal without scientific notation or trailing zeros
    norm = format(d.normalize(), "f")
    if "." in norm:
        norm = norm.rstrip("0").rstrip(".")
    return norm


def _extract_boxed(text: str) -> str | None:
    """Extract the last balanced \\boxed{...} span if present."""
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    pos = idx + len(marker)
    depth = 1
    out: list[str] = []
    while pos < len(text) and depth > 0:
        ch = text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        pos += 1
    if depth != 0:
        return None
    return "".join(out)


def extract_numeric_answer(text: str) -> tuple[str | None, str, bool]:
    """Extract and normalize a predicted numeric answer from model generation.

    Returns (normalized_value, parse_method, is_valid).
    """
    if not text or not text.strip():
        return None, "empty", False

    # If thinking tags exist, prioritize text after </think> when available
    if "</think>" in text:
        after_think = text.split("</think>")[-1].strip()
        if after_think:
            val, method, ok = extract_numeric_answer(after_think)
            if ok:
                return val, f"post_think_{method}", True

    boxed = _extract_boxed(text)
    if boxed is not None:
        val = normalize_numeric_string(boxed)
        if val is not None:
            return val, "boxed", True

    hash_matches = _HASH_RE.findall(text)
    if hash_matches:
        val = normalize_numeric_string(hash_matches[-1])
        if val is not None:
            return val, "gsm8k_hash", True

    final_matches = _FINAL_RE.findall(text)
    if final_matches:
        val = normalize_numeric_string(final_matches[-1])
        if val is not None:
            return val, "final_answer_phrase", True

    nums = _NUM_RE.findall(text)
    if nums:
        val = normalize_numeric_string(nums[-1])
        if val is not None:
            return val, "last_number", True

    return None, "unparsed", False


def answers_match(pred_text: str, gold_answer: str) -> tuple[bool, str | None, str]:
    """Return (is_correct, normalized_pred, parse_method) against gold numeric answer."""
    pred_val, method, ok = extract_numeric_answer(pred_text)
    gold_val = normalize_numeric_string(gold_answer)
    if not ok or pred_val is None or gold_val is None:
        return False, pred_val, method
    return pred_val == gold_val, pred_val, method
