"""Orchestrate Stage 1 dev screening and matched validation for C2, C3, C4_weights, C4_act, and C5."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from src.baselines.activation_transfer import (
    build_c4_activation_guided,
    collect_mlp_activations_stats,
    load_dev_calibration_prompts,
)
from src.baselines.random_controls import (
    build_c2_random_perturbation,
    build_c3_random_donor_slice,
)
from src.baselines.shuffled_donor import build_shuffled_donor_override
from src.baselines.weights_only_transfer import build_c4_weights_only
from src.evaluation.runner import evaluate_jsonl


TARGET_LAYERS = list(range(8, 24))  # Upper 16 of 24 text layers


def _prepare_dev_screen_jsonl(out_path: Path, n_carry: int = 20, n_gsm8k: int = 20) -> Path:
    rows: list[dict[str, Any]] = []
    with Path("data/development/synthetic_dev.jsonl").open("r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("subskill") == "carry_add_sub":
                rows.append(r)
                if len(rows) >= n_carry:
                    break
    gsm_cnt = 0
    with Path("data/development/gsm8k_dev.jsonl").open("r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            rows.append(json.loads(ln))
            gsm_cnt += 1
            if gsm_cnt >= n_gsm8k:
                break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path


def _eval_validation_pair(
    ckpt_dir: Path,
    arm_tag: str,
    device: str = "mps",
    dtype_name: str = "bfloat16",
) -> dict[str, Any]:
    syn_man = evaluate_jsonl(
        model_path=ckpt_dir,
        dataset_path="data/validation/synthetic_val.jsonl",
        run_id=f"m2_{arm_tag}_synthetic_val",
        output_manifest_path=f"runs/manifests/m2_{arm_tag}_synthetic_val.json",
        output_log_path=f"runs/logs/m2_{arm_tag}_synthetic_val.jsonl",
        enable_thinking=False,
        max_new_tokens=1024,
        per_subskill_limit=50,
        device=device,
        dtype_name=dtype_name,
    )
    gsm_man = evaluate_jsonl(
        model_path=ckpt_dir,
        dataset_path="data/validation/gsm8k_val.jsonl",
        run_id=f"m2_{arm_tag}_gsm8k_val",
        output_manifest_path=f"runs/manifests/m2_{arm_tag}_gsm8k_val.json",
        output_log_path=f"runs/logs/m2_{arm_tag}_gsm8k_val.jsonl",
        enable_thinking=False,
        max_new_tokens=1024,
        limit=100,
        device=device,
        dtype_name=dtype_name,
    )
    return {
        "synthetic_val": syn_man["metrics"],
        "gsm8k_val": gsm_man["metrics"],
        "synth_tok_s": syn_man["tokens_per_sec"],
        "gsm8k_tok_s": gsm_man["tokens_per_sec"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor-dir", default="hf_cache/models/qwen3.5_2b")
    parser.add_argument("--recipient-dir", default="hf_cache/models/qwen3.5_0.8b")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", default="results/m2_conventional_baselines.json")
    args = parser.parse_args()

    dev_screen_path = _prepare_dev_screen_jsonl(Path("hf_cache/tmp_m2_dev_screen.jsonl"))
    edited_root = Path("hf_cache/edited")
    edited_root.mkdir(parents=True, exist_ok=True)

    # 1. Screen C4_weights across alpha in {0.02, 0.05, 0.10} (with 0.005 fallback if all <0.25)
    c4w_screen: list[dict[str, Any]] = []
    for alpha in [0.02, 0.05, 0.10]:
        tag = f"c4w_a{str(alpha).replace('.', '_')}"
        ckpt = edited_root / f"tmp_{tag}"
        patch_rec = build_c4_weights_only(
            donor_dir=args.donor_dir,
            recipient_dir=args.recipient_dir,
            out_dir=ckpt,
            target_layers=TARGET_LAYERS,
            alpha=alpha,
            manifest_path=f"runs/manifests/m2_{tag}_patch.json",
        )
        man = evaluate_jsonl(
            model_path=ckpt,
            dataset_path=dev_screen_path,
            run_id=f"m2_devscreen_{tag}",
            output_manifest_path=f"runs/manifests/m2_devscreen_{tag}.json",
            output_log_path=f"runs/logs/m2_devscreen_{tag}.jsonl",
            enable_thinking=False,
            max_new_tokens=1024,
            device=args.device,
            dtype_name=args.dtype,
        )
        c4w_screen.append(
            {
                "alpha": alpha,
                "dev_pass_at_1": man["metrics"]["pass_at_1"],
                "dev_carry_acc": man["metrics"]["by_subskill"].get("carry_add_sub", {}).get("pass_at_1"),
                "rel_frob": patch_rec["invariants"]["overall_relative_frobenius_norm"],
                "changed_params": patch_rec["invariants"]["changed_params"],
            }
        )
        shutil.rmtree(ckpt)

    if max(x["dev_pass_at_1"] for x in c4w_screen) < 0.25:
        alpha = 0.005
        tag = "c4w_a0_005"
        ckpt = edited_root / f"tmp_{tag}"
        patch_rec = build_c4_weights_only(
            donor_dir=args.donor_dir,
            recipient_dir=args.recipient_dir,
            out_dir=ckpt,
            target_layers=TARGET_LAYERS,
            alpha=alpha,
            manifest_path=f"runs/manifests/m2_{tag}_patch.json",
        )
        man = evaluate_jsonl(
            model_path=ckpt,
            dataset_path=dev_screen_path,
            run_id=f"m2_devscreen_{tag}",
            output_manifest_path=f"runs/manifests/m2_devscreen_{tag}.json",
            output_log_path=f"runs/logs/m2_devscreen_{tag}.jsonl",
            enable_thinking=False,
            max_new_tokens=1024,
            device=args.device,
            dtype_name=args.dtype,
        )
        c4w_screen.append(
            {
                "alpha": alpha,
                "dev_pass_at_1": man["metrics"]["pass_at_1"],
                "dev_carry_acc": man["metrics"]["by_subskill"].get("carry_add_sub", {}).get("pass_at_1"),
                "rel_frob": patch_rec["invariants"]["overall_relative_frobenius_norm"],
                "changed_params": patch_rec["invariants"]["changed_params"],
            }
        )
        shutil.rmtree(ckpt)

    best_c4w = max(c4w_screen, key=lambda x: x["dev_pass_at_1"])
    best_alpha_w = float(best_c4w["alpha"])

    # 2. Collect calibration stats once and screen C4_act
    cal_prompts = load_dev_calibration_prompts(n_synth=16, n_gsm8k=16)
    act_stats = collect_mlp_activations_stats(
        donor_dir=args.donor_dir,
        recipient_dir=args.recipient_dir,
        prompts=cal_prompts,
        target_layers=TARGET_LAYERS,
        device=args.device,
        dtype_name=args.dtype,
    )

    c4a_screen: list[dict[str, Any]] = []
    for alpha in [0.02, 0.05, 0.10]:
        tag = f"c4a_a{str(alpha).replace('.', '_')}"
        ckpt = edited_root / f"tmp_{tag}"
        patch_rec = build_c4_activation_guided(
            donor_dir=args.donor_dir,
            recipient_dir=args.recipient_dir,
            out_dir=ckpt,
            target_layers=TARGET_LAYERS,
            act_stats=act_stats,
            alpha=alpha,
            n_calibration_prompts=len(cal_prompts),
            manifest_path=f"runs/manifests/m2_{tag}_patch.json",
        )
        man = evaluate_jsonl(
            model_path=ckpt,
            dataset_path=dev_screen_path,
            run_id=f"m2_devscreen_{tag}",
            output_manifest_path=f"runs/manifests/m2_devscreen_{tag}.json",
            output_log_path=f"runs/logs/m2_devscreen_{tag}.jsonl",
            enable_thinking=False,
            max_new_tokens=1024,
            device=args.device,
            dtype_name=args.dtype,
        )
        c4a_screen.append(
            {
                "alpha": alpha,
                "dev_pass_at_1": man["metrics"]["pass_at_1"],
                "dev_carry_acc": man["metrics"]["by_subskill"].get("carry_add_sub", {}).get("pass_at_1"),
                "rel_frob": patch_rec["invariants"]["overall_relative_frobenius_norm"],
                "changed_params": patch_rec["invariants"]["changed_params"],
            }
        )
        shutil.rmtree(ckpt)

    if max(x["dev_pass_at_1"] for x in c4a_screen) < 0.25:
        alpha = 0.005
        tag = "c4a_a0_005"
        ckpt = edited_root / f"tmp_{tag}"
        patch_rec = build_c4_activation_guided(
            donor_dir=args.donor_dir,
            recipient_dir=args.recipient_dir,
            out_dir=ckpt,
            target_layers=TARGET_LAYERS,
            act_stats=act_stats,
            alpha=alpha,
            n_calibration_prompts=len(cal_prompts),
            manifest_path=f"runs/manifests/m2_{tag}_patch.json",
        )
        man = evaluate_jsonl(
            model_path=ckpt,
            dataset_path=dev_screen_path,
            run_id=f"m2_devscreen_{tag}",
            output_manifest_path=f"runs/manifests/m2_devscreen_{tag}.json",
            output_log_path=f"runs/logs/m2_devscreen_{tag}.jsonl",
            enable_thinking=False,
            max_new_tokens=1024,
            device=args.device,
            dtype_name=args.dtype,
        )
        c4a_screen.append(
            {
                "alpha": alpha,
                "dev_pass_at_1": man["metrics"]["pass_at_1"],
                "dev_carry_acc": man["metrics"]["by_subskill"].get("carry_add_sub", {}).get("pass_at_1"),
                "rel_frob": patch_rec["invariants"]["overall_relative_frobenius_norm"],
                "changed_params": patch_rec["invariants"]["changed_params"],
            }
        )
        shutil.rmtree(ckpt)

    best_c4a = max(c4a_screen, key=lambda x: x["dev_pass_at_1"])
    best_alpha_a = float(best_c4a["alpha"])

    # 3. Build final C4_weights, C4_act, matched C2, and matched C3 checkpoints & run validation
    validation_results: dict[str, Any] = {}

    ckpt_c4w = edited_root / "m2_c4_weights"
    rec_c4w = build_c4_weights_only(
        donor_dir=args.donor_dir,
        recipient_dir=args.recipient_dir,
        out_dir=ckpt_c4w,
        target_layers=TARGET_LAYERS,
        alpha=best_alpha_w,
        manifest_path="runs/manifests/m2_c4_weights_patch.json",
    )
    # Match C2 per-tensor Frobenius norm to the mean per-tensor rel_frob of C4_weights
    c4w_tensor_frob = [t["relative_frobenius_norm"] for t in rec_c4w["invariants"]["changed_tensors"]]
    mean_tensor_rel_frob = sum(c4w_tensor_frob) / max(len(c4w_tensor_frob), 1)
    validation_results["C4_weights"] = {
        "patch_invariants": rec_c4w["invariants"],
        "alpha": best_alpha_w,
        **_eval_validation_pair(ckpt_c4w, "c4_weights", device=args.device, dtype_name=args.dtype),
    }

    ckpt_c4a = edited_root / "m2_c4_act"
    rec_c4a = build_c4_activation_guided(
        donor_dir=args.donor_dir,
        recipient_dir=args.recipient_dir,
        out_dir=ckpt_c4a,
        target_layers=TARGET_LAYERS,
        act_stats=act_stats,
        alpha=best_alpha_a,
        n_calibration_prompts=len(cal_prompts),
        manifest_path="runs/manifests/m2_c4_act_patch.json",
    )
    validation_results["C4_act"] = {
        "patch_invariants": rec_c4a["invariants"],
        "alpha": best_alpha_a,
        **_eval_validation_pair(ckpt_c4a, "c4_act", device=args.device, dtype_name=args.dtype),
    }

    ckpt_c2 = edited_root / "m2_c2_random_perturb"
    rec_c2 = build_c2_random_perturbation(
        recipient_dir=args.recipient_dir,
        out_dir=ckpt_c2,
        target_layers=TARGET_LAYERS,
        target_rel_frob=mean_tensor_rel_frob,
        seed=42,
        manifest_path="runs/manifests/m2_c2_patch.json",
    )
    validation_results["C2"] = {
        "patch_invariants": rec_c2["invariants"],
        "target_rel_frob": mean_tensor_rel_frob,
        **_eval_validation_pair(ckpt_c2, "c2", device=args.device, dtype_name=args.dtype),
    }
    shutil.rmtree(ckpt_c2)

    ckpt_c3 = edited_root / "m2_c3_random_slice"
    rec_c3 = build_c3_random_donor_slice(
        donor_dir=args.donor_dir,
        recipient_dir=args.recipient_dir,
        out_dir=ckpt_c3,
        target_layers=TARGET_LAYERS,
        alpha=best_alpha_w,
        seed=42,
        manifest_path="runs/manifests/m2_c3_patch.json",
    )
    validation_results["C3"] = {
        "patch_invariants": rec_c3["invariants"],
        "alpha": best_alpha_w,
        **_eval_validation_pair(ckpt_c3, "c3", device=args.device, dtype_name=args.dtype),
    }
    shutil.rmtree(ckpt_c3)

    # 4. Run C5 shuffled-donor control if either C4_weights or C4_act beats C0_raw (0.62 GSM8K or 0.40 carry) by >= +0.02
    c4w_gsm = validation_results["C4_weights"]["gsm8k_val"]["pass_at_1"]
    c4w_car = validation_results["C4_weights"]["synthetic_val"]["by_subskill"]["carry_add_sub"]["pass_at_1"]
    c4a_gsm = validation_results["C4_act"]["gsm8k_val"]["pass_at_1"]
    c4a_car = validation_results["C4_act"]["synthetic_val"]["by_subskill"]["carry_add_sub"]["pass_at_1"]

    if (c4w_gsm >= 0.64 or c4w_car >= 0.42):
        shuf_w = build_shuffled_donor_override(args.donor_dir, TARGET_LAYERS, include_embed=True, seed=42)
        ckpt_c5w = edited_root / "m2_c5_shuffled_c4w"
        rec_c5w = build_c4_weights_only(
            donor_dir=args.donor_dir,
            recipient_dir=args.recipient_dir,
            out_dir=ckpt_c5w,
            target_layers=TARGET_LAYERS,
            alpha=best_alpha_w,
            manifest_path="runs/manifests/m2_c5_c4w_patch.json",
            donor_override=shuf_w,
        )
        validation_results["C5_shuffled_C4_weights"] = {
            "patch_invariants": rec_c5w["invariants"],
            **_eval_validation_pair(ckpt_c5w, "c5_c4w", device=args.device, dtype_name=args.dtype),
        }
        shutil.rmtree(ckpt_c5w)

    if (c4a_gsm >= 0.64 or c4a_car >= 0.42):
        shuf_a = build_shuffled_donor_override(args.donor_dir, TARGET_LAYERS, include_embed=False, seed=43)
        ckpt_c5a = edited_root / "m2_c5_shuffled_c4a"
        rec_c5a = build_c4_activation_guided(
            donor_dir=args.donor_dir,
            recipient_dir=args.recipient_dir,
            out_dir=ckpt_c5a,
            target_layers=TARGET_LAYERS,
            act_stats=act_stats,
            alpha=best_alpha_a,
            n_calibration_prompts=len(cal_prompts),
            manifest_path="runs/manifests/m2_c5_c4a_patch.json",
            donor_override=shuf_a,
        )
        validation_results["C5_shuffled_C4_act"] = {
            "patch_invariants": rec_c5a["invariants"],
            **_eval_validation_pair(ckpt_c5a, "c5_c4a", device=args.device, dtype_name=args.dtype),
        }
        shutil.rmtree(ckpt_c5a)

    out_data = {
        "target_layers": TARGET_LAYERS,
        "dev_screen": {"C4_weights": c4w_screen, "C4_act": c4a_screen},
        "validation": validation_results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out_data, indent=2), encoding="utf-8")

    compact = {
        k: {
            "gsm8k_val": v["gsm8k_val"]["pass_at_1"],
            "synth_val": v["synthetic_val"]["pass_at_1"],
            "carry_add_sub": v["synthetic_val"]["by_subskill"]["carry_add_sub"]["pass_at_1"],
            "mult_2x2_3x1": v["synthetic_val"]["by_subskill"]["mult_2x2_3x1"]["pass_at_1"],
            "word_2step": v["synthetic_val"]["by_subskill"]["word_2step"]["pass_at_1"],
            "rel_frob": round(v["patch_invariants"]["overall_relative_frobenius_norm"], 5),
        }
        for k, v in validation_results.items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
