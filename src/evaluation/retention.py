"""Deterministic capability-retention (IFEval-style), prose perplexity, and Holm-Bonferroni helpers."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


ConstraintFn = Callable[[str], tuple[bool, str]]


@dataclass(frozen=True)
class RetentionTask:
    task_id: str
    category: str
    prompt: str
    check: ConstraintFn


def _check_bullet_count(expected_n: int) -> ConstraintFn:
    def _fn(text: str) -> tuple[bool, str]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        bullets = [ln for ln in lines if ln.startswith("- ") and len(ln) > 3]
        ok = len(lines) == expected_n and len(bullets) == expected_n
        return ok, f"bullets={len(bullets)}/{len(lines)} lines (expected {expected_n})"

    return _fn


def _check_json_keys(required_keys: tuple[str, ...]) -> ConstraintFn:
    req = set(required_keys)

    def _fn(text: str) -> tuple[bool, str]:
        raw = text.strip()
        if raw.startswith("```"):
            parts = raw.splitlines()
            if len(parts) >= 3 and parts[-1].strip() == "```":
                raw = "\n".join(parts[1:-1]).strip()
        try:
            obj = json.loads(raw)
        except Exception as exc:
            return False, f"invalid_json: {exc}"
        if not isinstance(obj, dict):
            return False, f"expected JSON object, got {type(obj).__name__}"
        keys = set(obj.keys())
        ok = keys == req and all(str(obj[k]).strip() != "" for k in req)
        return ok, f"keys={sorted(keys)} expected={sorted(req)}"

    return _fn


def _check_forbidden_char(forbidden: str, min_alpha: int = 20) -> ConstraintFn:
    bad = forbidden.lower()

    def _fn(text: str) -> tuple[bool, str]:
        alpha = sum(1 for c in text if c.isalpha())
        hits = text.lower().count(bad)
        ok = alpha >= min_alpha and hits == 0
        return ok, f"alpha={alpha} forbidden_{bad}_count={hits}"

    return _fn


def _check_exact_word_count(expected_words: int) -> ConstraintFn:
    def _fn(text: str) -> tuple[bool, str]:
        words = text.strip().split()
        ok = len(words) == expected_words
        return ok, f"word_count={len(words)} expected={expected_words}"

    return _fn


def _check_affix(required: str, *, mode: str) -> ConstraintFn:
    def _fn(text: str) -> tuple[bool, str]:
        stripped = text.strip()
        if mode == "prefix":
            ok = stripped.startswith(required) and len(stripped) >= len(required) + 15
        elif mode == "suffix":
            ok = stripped.endswith(required) and len(stripped) >= len(required) + 15
        else:
            raise ValueError(f"Unknown affix mode: {mode}")
        return ok, f"mode={mode} len={len(stripped)} matched={ok}"

    return _fn


def _check_case(expected_case: str, min_alpha: int = 20) -> ConstraintFn:
    def _fn(text: str) -> tuple[bool, str]:
        stripped = text.strip()
        alpha = sum(1 for c in stripped if c.isalpha())
        if expected_case == "upper":
            ok = alpha >= min_alpha and stripped == stripped.upper() and stripped != stripped.lower()
        elif expected_case == "lower":
            ok = alpha >= min_alpha and stripped == stripped.lower() and stripped != stripped.upper()
        else:
            raise ValueError(f"Unknown case mode: {expected_case}")
        return ok, f"case={expected_case} alpha={alpha} ok={ok}"

    return _fn


RETENTION_PROMPTS: tuple[RetentionTask, ...] = (
    # 1. Exact bullet counts (5 items)
    RetentionTask(
        "ret_bullet_01",
        "bullet_count",
        "List exactly 3 outdoor hobbies. Output only markdown bullet lines starting with '- ' and nothing else.",
        _check_bullet_count(3),
    ),
    RetentionTask(
        "ret_bullet_02",
        "bullet_count",
        "List exactly 4 kitchen utensils. Output only markdown bullet lines starting with '- ' and no intro or outro.",
        _check_bullet_count(4),
    ),
    RetentionTask(
        "ret_bullet_03",
        "bullet_count",
        "List exactly 5 musical instruments. Output only markdown bullet lines starting with '- ' and nothing else.",
        _check_bullet_count(5),
    ),
    RetentionTask(
        "ret_bullet_04",
        "bullet_count",
        "List exactly 3 winter weather precautions. Output only markdown bullet lines starting with '- ' and nothing else.",
        _check_bullet_count(3),
    ),
    RetentionTask(
        "ret_bullet_05",
        "bullet_count",
        "List exactly 4 mountain safety tips. Output only markdown bullet lines starting with '- ' and no extra text.",
        _check_bullet_count(4),
    ),
    # 2. Strict JSON object format (5 items)
    RetentionTask(
        "ret_json_01",
        "json_format",
        'Return only a raw JSON object describing Kyoto with exact keys "city", "country", and "continent". No markdown or prose.',
        _check_json_keys(("city", "country", "continent")),
    ),
    RetentionTask(
        "ret_json_02",
        "json_format",
        'Return only a raw JSON object describing a bicycle with exact keys "item", "wheels", and "material". No extra text.',
        _check_json_keys(("item", "wheels", "material")),
    ),
    RetentionTask(
        "ret_json_03",
        "json_format",
        'Return only a raw JSON object for a library book with exact keys "title", "author", and "genre". No prose outside JSON.',
        _check_json_keys(("title", "author", "genre")),
    ),
    RetentionTask(
        "ret_json_04",
        "json_format",
        'Return only a raw JSON object about Mars with exact keys "planet", "color", and "region". Output JSON only.',
        _check_json_keys(("planet", "color", "region")),
    ),
    RetentionTask(
        "ret_json_05",
        "json_format",
        'Return only a raw JSON object for green tea with exact keys "drink", "origin", and "temperature". No extra text.',
        _check_json_keys(("drink", "origin", "temperature")),
    ),
    # 3. Forbidden letter constraint (5 items)
    RetentionTask(
        "ret_forbid_01",
        "forbidden_char",
        "Write one short paragraph (at least 6 words) praising morning rain without using the letter 'e' or 'E'.",
        _check_forbidden_char("e"),
    ),
    RetentionTask(
        "ret_forbid_02",
        "forbidden_char",
        "Describe a quiet snowy forest in a sentence (at least 6 words) without using the letter 'a' or 'A'.",
        _check_forbidden_char("a"),
    ),
    RetentionTask(
        "ret_forbid_03",
        "forbidden_char",
        "Write one sentence (at least 6 words) about coastal sailing without using the letter 'o' or 'O'.",
        _check_forbidden_char("o"),
    ),
    RetentionTask(
        "ret_forbid_04",
        "forbidden_char",
        "Write one sentence (at least 6 words) about baking warm bread without using the letter 'i' or 'I'.",
        _check_forbidden_char("i"),
    ),
    RetentionTask(
        "ret_forbid_05",
        "forbidden_char",
        "Describe a calm evening sky in one sentence (at least 6 words) without using the letter 'u' or 'U'.",
        _check_forbidden_char("u"),
    ),
    # 4. Exact word count (5 items)
    RetentionTask(
        "ret_words_01",
        "exact_word_count",
        "Write a sentence describing an autumn park that contains exactly 10 words. Output only the sentence.",
        _check_exact_word_count(10),
    ),
    RetentionTask(
        "ret_words_02",
        "exact_word_count",
        "Write a sentence about reading at home that contains exactly 12 words. Output only the sentence.",
        _check_exact_word_count(12),
    ),
    RetentionTask(
        "ret_words_03",
        "exact_word_count",
        "Write a sentence about drinking coffee on a train using exactly 8 words. Output only the sentence.",
        _check_exact_word_count(8),
    ),
    RetentionTask(
        "ret_words_04",
        "exact_word_count",
        "Explain why sleep matters in a single response containing exactly 14 words. Output only those words.",
        _check_exact_word_count(14),
    ),
    RetentionTask(
        "ret_words_05",
        "exact_word_count",
        "Describe a calm harbor at sunrise using exactly 9 words. Output only the sentence.",
        _check_exact_word_count(9),
    ),
    # 5. Exact prefix / suffix (5 items)
    RetentionTask(
        "ret_affix_01",
        "exact_ending_or_prefix",
        "Write two sentences explaining how compost helps garden soil. Start your response with the exact prefix 'NOTE:' (including the colon).",
        _check_affix("NOTE:", mode="prefix"),
    ),
    RetentionTask(
        "ret_affix_02",
        "exact_ending_or_prefix",
        "Write two sentences describing why public libraries matter. End your entire response with the exact sentence 'That completes the note.'",
        _check_affix("That completes the note.", mode="suffix"),
    ),
    RetentionTask(
        "ret_affix_03",
        "exact_ending_or_prefix",
        "Briefly explain why hydration helps focus. Start your response with the exact prefix 'SUMMARY:' (including the colon).",
        _check_affix("SUMMARY:", mode="prefix"),
    ),
    RetentionTask(
        "ret_affix_04",
        "exact_ending_or_prefix",
        "Write a brief tip for backing up important files. End your response with the exact phrase 'End of reminder.'",
        _check_affix("End of reminder.", mode="suffix"),
    ),
    RetentionTask(
        "ret_affix_05",
        "exact_ending_or_prefix",
        "Briefly describe how tides work along a coastline. Start your response with the exact prefix 'BRIEFING:' (including the colon).",
        _check_affix("BRIEFING:", mode="prefix"),
    ),
    # 6. Strict case format (5 items)
    RetentionTask(
        "ret_case_01",
        "case_format",
        "Write two sentences about desert stars using ALL UPPERCASE LETTERS ONLY. Include zero lowercase letters.",
        _check_case("upper"),
    ),
    RetentionTask(
        "ret_case_02",
        "case_format",
        "Write two sentences about mountain rivers using all lowercase letters only. Include zero uppercase letters.",
        _check_case("lower"),
    ),
    RetentionTask(
        "ret_case_03",
        "case_format",
        "Give one piece of advice about patience using ALL UPPERCASE LETTERS ONLY. Include zero lowercase letters.",
        _check_case("upper"),
    ),
    RetentionTask(
        "ret_case_04",
        "case_format",
        "Describe a thunderstorm over open fields using all lowercase letters only. Include zero uppercase letters.",
        _check_case("lower"),
    ),
    RetentionTask(
        "ret_case_05",
        "case_format",
        "Summarize why maps are useful in history using ALL UPPERCASE LETTERS ONLY. Do not use any lowercase letters.",
        _check_case("upper"),
    ),
)


PERPLEXITY_PASSAGES: tuple[str, ...] = (
    (
        "Deep ocean currents circulate cold polar water slowly along the abyssal plains toward the equator, "
        "carrying dissolved oxygen and nutrients that sustain benthic ecosystems far below sunlit surface waters. "
        "As warmer tropical water flows poleward near the ocean surface to replace sinking cold layers, "
        "coastal climates across distant continents experience moderated seasonal temperature swings."
    ),
    (
        "Movable metal type transformed European bookmaking during the fifteenth century by allowing workshops "
        "to assemble pages from durable reusable letterforms rather than carving whole wooden blocks for every leaf. "
        "Faster printing lowered the cost of civic records, grammar manuals, and agricultural treatises, "
        "turning small urban print shops into busy centers of technical correspondence."
    ),
    (
        "Deciduous trees prepare for winter frost long before leaves drop from their branches. "
        "Triggered by shortening autumn days, green chlorophyll breaks down and allows hidden yellow and orange "
        "carotenoid pigments to show through while sugars migrate into protective root tissues beneath the soil."
    ),
    (
        "Masonry arches span wide openings by channeling vertical stone weight outward into angled compression "
        "along curved voussoirs down into heavy supporting piers. Before mortar was widely relied upon, "
        "skilled builders shaped interlocking wedge stones so friction and gravity held bridges rigid under heavy wagons."
    ),
    (
        "Acoustic string instruments rely on a hollow wooden soundboard rather than thick solid timber to project warm tones. "
        "Vibration from a plucked or bowed string travels through a small bridge into thin spruce plates, "
        "setting internal air columns in motion and emphasizing specific harmonic overtones."
    ),
    (
        "High altitude cirrus clouds form in thin wispy streaks where rising moisture freezes directly into tiny ice crystals. "
        "Because upper-level jet stream winds race faster than breezes near the ground, "
        "the trailing ice filaments curl into curved hooks that often signal an approaching frontal weather system."
    ),
    (
        "Volcanic island arcs emerge above subduction trenches where dense oceanic crust bends downward into hotter mantle rock. "
        "Trapped seawater lowers the melting point of surrounding minerals, generating buoyant magma plumes "
        "that slowly construct steep basalt and andesite cones above the surrounding sea."
    ),
    (
        "Early coastal navigators charted unfamiliar harbors using lead sounding lines coated with tallow to inspect "
        "both water depth and seabed sediment composition. Measuring mud, shell fragments, or coarse sand "
        "alongside compass bearings helped pilots recognize submerged shoals even in thick evening fog."
    ),
    (
        "Orchard growers depend on wild solitary bees as well as managed honeybee hives during brief spring blossom windows. "
        "Because solitary mason bees fly willingly in cool drizzly mornings when honeybees remain clustered indoors, "
        "preserved hedgerows and hollow reed nesting sites noticeably stabilize early fruit set."
    ),
    (
        "Glass Fresnel lenses allowed nineteenth-century coastal lighthouses to concentrate an oil lamp's glow "
        "into a sharp horizontal beam visible miles offshore. Concentric stepped rings bent stray upward light "
        "straight toward the horizon while using far less heavy solid glass than a conventional thick lens."
    ),
)


def evaluate_retention(
    model: Any,
    tokenizer: Any,
    device: str,
    max_new_tokens: int = 512,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic 30-item instruction-following retention suite (nothink, greedy)."""
    records: list[dict[str, Any]] = []
    by_cat: dict[str, list[bool]] = {}
    total_tokens = 0
    t_start = time.perf_counter()

    lf = None
    if log_path is not None:
        lp = Path(log_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        lf = lp.open("w", encoding="utf-8")

    try:
        with torch.inference_mode():
            for task in RETENTION_PROMPTS:
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": task.prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
                prompt_len = int(inputs["input_ids"].shape[1])
                t0 = time.perf_counter()
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                elapsed = time.perf_counter() - t0
                gen_ids = out_ids[0][prompt_len:]
                gen_len = int(gen_ids.shape[0])
                total_tokens += gen_len
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                ok, detail = task.check(text)

                rec = {
                    "id": task.task_id,
                    "category": task.category,
                    "correct": bool(ok),
                    "detail": detail,
                    "generated_tokens": gen_len,
                    "latency_sec": round(elapsed, 4),
                    "response_text": text,
                }
                records.append(rec)
                by_cat.setdefault(task.category, []).append(bool(ok))
                if lf is not None:
                    lf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        if lf is not None:
            lf.close()

    eval_sec = time.perf_counter() - t_start
    passed = sum(1 for r in records if r["correct"])
    n = len(records)
    return {
        "n": n,
        "passed_count": passed,
        "pass_rate": passed / max(n, 1),
        "by_category": {
            cat: {
                "n": len(vals),
                "passed_count": sum(vals),
                "pass_rate": sum(vals) / len(vals),
            }
            for cat, vals in sorted(by_cat.items())
        },
        "avg_output_tokens": total_tokens / max(n, 1),
        "tokens_per_sec": round(total_tokens / max(eval_sec, 1e-6), 2),
        "eval_sec": round(eval_sec, 3),
        "records": records,
    }


def evaluate_perplexity(model: Any, tokenizer: Any, device: str) -> dict[str, Any]:
    """Evaluate teacher-forced perplexity across 10 general non-math English passages."""
    total_nll = 0.0
    total_tokens = 0
    t0 = time.perf_counter()

    with torch.inference_mode():
        for idx, passage in enumerate(PERPLEXITY_PASSAGES):
            enc = tokenizer(passage, return_tensors="pt", add_special_tokens=False)
            input_ids = enc["input_ids"].to(device)
            seq_len = int(input_ids.shape[1])
            if seq_len < 2:
                raise ValueError(f"Passage {idx} tokenized to < 2 tokens")
            out = model(input_ids=input_ids, labels=input_ids)
            loss_val = float(out.loss.item())
            if not math.isfinite(loss_val):
                raise RuntimeError(f"Non-finite perplexity loss on passage {idx}: {loss_val}")
            pred_tokens = seq_len - 1
            total_nll += loss_val * pred_tokens
            total_tokens += pred_tokens

    eval_sec = time.perf_counter() - t0
    mean_nll = total_nll / max(total_tokens, 1)
    return {
        "n_passages": len(PERPLEXITY_PASSAGES),
        "total_predicted_tokens": total_tokens,
        "mean_nll": round(mean_nll, 6),
        "perplexity": round(math.exp(mean_nll), 4),
        "eval_sec": round(eval_sec, 3),
    }


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Apply step-down Holm-Bonferroni multiple-comparison adjustment, capped at 1.0."""
    if not p_values:
        return {}
    m = len(p_values)
    ordered = sorted(p_values.items(), key=lambda kv: float(kv[1]))
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, (name, raw_p) in enumerate(ordered):
        candidate = min(1.0, (m - rank) * float(raw_p))
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return {k: adjusted[k] for k in p_values}
