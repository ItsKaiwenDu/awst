"""Run compact Stage 0 think-mode replication check (C0_raw vs C0_loopfixed) and consolidate M1 baselines."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.runner import evaluate_jsonl


def main() -> None:
    runs = []
    models = [
        ("qwen3.5_0.8b_raw", "hf_cache/models/qwen3.5_0.8b"),
        ("qwen3.5_0.8b_loopfixed", "hf_cache/edited/qwen3.5_0.8b_loopfixed"),
    ]
    for label, path in models:
        run_id = f"stage0_thinkcheck_{label}_gsm8k_sample20"
        manifest = evaluate_jsonl(
            model_path=path,
            dataset_path="data/validation/gsm8k_val.jsonl",
            run_id=run_id,
            output_manifest_path=f"runs/manifests/{run_id}.json",
            output_log_path=f"runs/logs/{run_id}.jsonl",
            enable_thinking=True,
            max_new_tokens=2048,
            limit=20,
            device="mps",
            dtype_name="bfloat16",
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            seed=42,
        )
        runs.append(manifest)

    think_summary = {
        "note": (
            "Seeded sampling (temp=0.6, top_p=0.95, top_k=20, seed=42, max_new_tokens=2048); "
            "Hugging Face generate omits vLLM presence_penalty=1.5 used in Takehi et al."
        ),
        "runs": [
            {
                "run_id": r["run_id"],
                "model_path": r["model_path"],
                "eval_sec": r["eval_sec"],
                "tokens_per_sec": r["tokens_per_sec"],
                "metrics": r["metrics"],
            }
            for r in runs
        ],
    }
    Path("results/m1_stage0_think_replication.json").write_text(
        json.dumps(think_summary, indent=2), encoding="utf-8"
    )

    nothink = json.loads(
        Path("results/m1_stage0_baselines_nothink.json").read_text(encoding="utf-8")
    )
    remote = json.loads(
        Path("results/m1_remote_reader_validation.json").read_text(encoding="utf-8")
    )
    patch = json.loads(
        Path("runs/manifests/stage0_c0_loopfixed_patch.json").read_text(encoding="utf-8")
    )

    consolidated = {
        "milestone": "M1",
        "nothink_baselines": nothink,
        "think_replication_check": think_summary,
        "loopfix_patch_summary": {
            "changed_params": patch["invariants"]["changed_params"],
            "changed_fraction": patch["invariants"]["changed_fraction"],
            "overall_relative_frobenius_norm": patch["invariants"][
                "overall_relative_frobenius_norm"
            ],
            "frozen_tensors_verified": patch["invariants"]["frozen_tensors_verified"],
            "reversion_verified": patch["reversion_verified"],
        },
        "remote_reader_summary": {
            "bytes_read": remote["bytes_read"],
            "total_file_bytes": remote["total_file_bytes"],
            "byte_reduction_ratio": remote["byte_reduction_ratio"],
            "all_targets_bit_exact": all(t["bit_exact_match"] for t in remote["targets_verified"]),
        },
    }
    Path("results/m1_stage0_baselines.json").write_text(
        json.dumps(consolidated, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            [
                {
                    "run_id": r["run_id"],
                    "pass_at_1": r["metrics"]["pass_at_1"],
                    "loop_rate": r["metrics"]["repetition_loop_rate"],
                    "trunc_rate": r["metrics"]["truncation_rate"],
                    "avg_tok": r["metrics"]["avg_output_tokens"],
                    "eval_sec": r["eval_sec"],
                }
                for r in runs
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
