"""Run M4 Max micro-benchmark and Stage 0/1 baselines across C0_raw, C0_loopfixed, and Qwen3.5-2B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluation.runner import evaluate_jsonl


def run_suite(
    models: list[tuple[str, str]],
    datasets: list[tuple[str, str]],
    thinking_modes: list[tuple[bool, int]],
    output_summary_path: str | Path,
    limit: int | None = None,
    device: str = "auto",
    dtype_name: str = "bfloat16",
    tag: str = "stage0",
) -> dict[str, Any]:
    """Run all (model, dataset, thinking_mode) evaluations sequentially and save summary JSON."""
    results: list[dict[str, Any]] = []
    for model_label, model_path in models:
        for ds_label, ds_path in datasets:
            for enable_thinking, max_new_tokens in thinking_modes:
                think_tag = "think" if enable_thinking else "nothink"
                run_id = f"{tag}_{model_label}_{ds_label}_{think_tag}_{dtype_name}"
                manifest_out = Path("runs/manifests") / f"{run_id}.json"
                log_out = Path("runs/logs") / f"{run_id}.jsonl"
                manifest = evaluate_jsonl(
                    model_path=model_path,
                    dataset_path=ds_path,
                    run_id=run_id,
                    output_manifest_path=manifest_out,
                    output_log_path=log_out,
                    enable_thinking=enable_thinking,
                    max_new_tokens=max_new_tokens,
                    limit=limit,
                    device=device,
                    dtype_name=dtype_name,
                )
                results.append(
                    {
                        "run_id": run_id,
                        "model_label": model_label,
                        "dataset_label": ds_label,
                        "enable_thinking": enable_thinking,
                        "max_new_tokens": max_new_tokens,
                        "device": manifest["device"],
                        "dtype": manifest["dtype"],
                        "load_sec": manifest["load_sec"],
                        "eval_sec": manifest["eval_sec"],
                        "tokens_per_sec": manifest["tokens_per_sec"],
                        "peak_mps_allocated_mb": manifest["peak_mps_allocated_mb"],
                        "peak_rss_mb": manifest["peak_rss_mb"],
                        "metrics": manifest["metrics"],
                        "manifest_path": str(manifest_out),
                    }
                )

    summary = {"tag": tag, "limit": limit, "runs": results}
    out = Path(output_summary_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="micro")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-08b-raw", default="hf_cache/models/qwen3.5_0.8b")
    parser.add_argument("--model-08b-loopfixed", default=None)
    parser.add_argument("--model-2b", default="hf_cache/models/qwen3.5_2b")
    parser.add_argument("--synth-dataset", default="data/validation/synthetic_val.jsonl")
    parser.add_argument("--gsm8k-dataset", default="data/validation/gsm8k_val.jsonl")
    parser.add_argument("--think-max-tokens", type=int, default=1024)
    parser.add_argument("--nothink-max-tokens", type=int, default=512)
    parser.add_argument("--only-mode", choices=["both", "think", "nothink"], default="both")
    args = parser.parse_args()

    models: list[tuple[str, str]] = [("qwen3.5_0.8b_raw", args.model_08b_raw)]
    if args.model_08b_loopfixed:
        models.append(("qwen3.5_0.8b_loopfixed", args.model_08b_loopfixed))
    models.append(("qwen3.5_2b", args.model_2b))

    datasets = [
        ("synthetic", args.synth_dataset),
        ("gsm8k", args.gsm8k_dataset),
    ]
    if args.only_mode == "both":
        modes = [(False, args.nothink_max_tokens), (True, args.think_max_tokens)]
    elif args.only_mode == "think":
        modes = [(True, args.think_max_tokens)]
    else:
        modes = [(False, args.nothink_max_tokens)]

    summary = run_suite(
        models=models,
        datasets=datasets,
        thinking_modes=modes,
        output_summary_path=args.output,
        limit=args.limit,
        device=args.device,
        dtype_name=args.dtype,
        tag=args.tag,
    )
    compact = [
        {
            "run_id": r["run_id"],
            "pass_at_1": round(r["metrics"]["pass_at_1"], 4),
            "loop_rate": round(r["metrics"]["repetition_loop_rate"], 4),
            "trunc_rate": round(r["metrics"]["truncation_rate"], 4),
            "tok_s": r["tokens_per_sec"],
            "eval_sec": r["eval_sec"],
        }
        for r in summary["runs"]
    ]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
