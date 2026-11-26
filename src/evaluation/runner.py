"""Reproducible local evaluation runner for Qwen3.5 checkpoints on Apple M4 Max / CPU."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.evaluation.metrics import compute_eval_metrics, detect_repetition_loop
from src.evaluation.parser import answers_match


PROMPT_INSTRUCTION = (
    "Solve the math problem step by step. Put your final numeric answer inside \\boxed{}."
)


def select_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def select_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def load_model_and_tokenizer(
    model_path: str | Path,
    device: str = "auto",
    dtype_name: str = "bfloat16",
) -> tuple[Any, Any, str, float]:
    """Load tokenizer and model onto target device, returning (model, tokenizer, device, load_sec)."""
    dev = select_device(device)
    dtype = select_dtype(dtype_name)
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    cfg = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    if getattr(cfg, "model_type", "") == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            local_files_only=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            local_files_only=True,
        )
    model = model.to(dev)
    model.eval()
    load_sec = time.perf_counter() - t0
    return model, tokenizer, dev, load_sec


def format_prompt(tokenizer: Any, problem: str, enable_thinking: bool) -> str:
    messages = [
        {"role": "user", "content": f"{PROMPT_INSTRUCTION}\n\nProblem: {problem}"}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def load_dataset_rows(
    dataset_path: str | Path,
    limit: int | None = None,
    per_subskill_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load JSONL rows safely (preserving Unicode line separators inside JSON strings)."""
    with Path(dataset_path).open("r", encoding="utf-8") as df:
        rows = [json.loads(line) for line in df if line.strip()]

    if per_subskill_limit is not None and any("subskill" in r and r["subskill"] for r in rows):
        counts: dict[str, int] = {}
        selected: list[dict[str, Any]] = []
        for r in rows:
            sk = r.get("subskill") or "_default"
            if counts.get(sk, 0) < per_subskill_limit:
                selected.append(r)
                counts[sk] = counts.get(sk, 0) + 1
        rows = selected
    elif limit is not None:
        rows = rows[:limit]
    return rows


def evaluate_jsonl(
    model_path: str | Path,
    dataset_path: str | Path,
    run_id: str,
    output_manifest_path: str | Path,
    output_log_path: str | Path,
    enable_thinking: bool = False,
    max_new_tokens: int = 512,
    limit: int | None = None,
    per_subskill_limit: int | None = None,
    device: str = "auto",
    dtype_name: str = "bfloat16",
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate a local checkpoint deterministically (or with seeded sampling) on a JSONL dataset."""
    rows = load_dataset_rows(
        dataset_path=dataset_path,
        limit=limit,
        per_subskill_limit=per_subskill_limit,
    )

    if do_sample:
        torch.manual_seed(seed)

    model, tokenizer, dev, load_sec = load_model_and_tokenizer(
        model_path=model_path, device=device, dtype_name=dtype_name
    )

    if dev == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    records: list[dict[str, Any]] = []
    total_gen_tokens = 0
    t_eval_start = time.perf_counter()

    log_file = Path(output_log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k

    with log_file.open("w", encoding="utf-8") as lf, torch.inference_mode():
        for item in rows:
            prompt_text = format_prompt(tokenizer, item["prompt"], enable_thinking=enable_thinking)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(dev)
            input_len = int(inputs["input_ids"].shape[1])

            t0 = time.perf_counter()
            out_ids = model.generate(**inputs, **gen_kwargs)
            elapsed = time.perf_counter() - t0

            gen_ids = out_ids[0][input_len:]
            gen_len = int(gen_ids.shape[0])
            total_gen_tokens += gen_len

            # Check truncation: hit max_new_tokens without EOS at the end
            eos_id = tokenizer.eos_token_id
            last_tok = int(gen_ids[-1].item()) if gen_len > 0 else None
            truncated = (gen_len >= max_new_tokens) and (eos_id is None or last_tok != eos_id)

            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            correct, pred_norm, parse_method = answers_match(gen_text, str(item["answer"]))
            loop_flag = detect_repetition_loop(gen_text)

            rec = {
                "id": item["id"],
                "subskill": item.get("subskill"),
                "gold_answer": str(item["answer"]),
                "pred_answer": pred_norm,
                "parse_method": parse_method,
                "parse_valid": pred_norm is not None,
                "correct": bool(correct),
                "truncated": bool(truncated),
                "repetition_loop": bool(loop_flag),
                "input_tokens": input_len,
                "generated_tokens": gen_len,
                "latency_sec": round(elapsed, 4),
                "response_text": gen_text,
            }
            records.append(rec)
            lf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    eval_sec = time.perf_counter() - t_eval_start
    metrics = compute_eval_metrics(records)

    peak_mps_mb = None
    if dev == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
        peak_mps_mb = round(torch.mps.driver_allocated_memory() / (1024 * 1024), 2)
    # macOS ru_maxrss is in bytes
    peak_rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 2)

    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "device": dev,
        "dtype": dtype_name,
        "enable_thinking": enable_thinking,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "top_k": top_k if do_sample else None,
        "seed": seed if do_sample else None,
        "load_sec": round(load_sec, 3),
        "eval_sec": round(eval_sec, 3),
        "total_generated_tokens": total_gen_tokens,
        "tokens_per_sec": round(total_gen_tokens / max(eval_sec, 1e-6), 2),
        "peak_mps_allocated_mb": peak_mps_mb,
        "peak_rss_mb": peak_rss_mb,
        "metrics": metrics,
        "log_path": str(log_file),
    }

    mp = Path(output_manifest_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Release model memory cleanly between sequential runs
    del model
    if dev == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--log-out", required=True)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    res = evaluate_jsonl(
        model_path=args.model_path,
        dataset_path=args.dataset,
        run_id=args.run_id,
        output_manifest_path=args.manifest_out,
        output_log_path=args.log_out,
        enable_thinking=args.enable_thinking,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        device=args.device,
        dtype_name=args.dtype,
    )
    print(json.dumps({k: v for k, v in res.items() if k != "log_path"}, indent=2))


if __name__ == "__main__":
    main()
