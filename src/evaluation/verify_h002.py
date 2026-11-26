"""Independent Antigravity verification of H002 (C1, T1 replay, float64 reversion, C5 controls, and blinded validation)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runs.codex_h002.apply_c1 import apply_c1
from runs.codex_h002.apply_t1 import apply_t1
from src.baselines.shuffled_donor import build_shuffled_donor_override
from src.evaluation.runner import evaluate_jsonl
from src.interventions.patch_applier import verify_exact_reversion
from src.weight_io.safetensors_reader import _resolve_tensor_file, sha256_file

EMBED_TENSOR = "model.language_model.embed_tokens.weight"
EXPECTED_BASE_SHA256 = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
EXPECTED_C1_SHA256 = "b11f16df6de7bfb5f060789477a4e67764946f111feccd3a871d2440227ad11a"
EXPECTED_T1_SHA256 = "da9ad05f9c4931477197b46581ed83653485f16763eeb7197bb9aaead2f81414"

RECIPIENT_DIR = Path("hf_cache/models/qwen3.5_0.8b")
DONOR_DIR = Path("hf_cache/models/qwen3.5_2b")
EDITED_ROOT = Path("hf_cache/edited")
INVARIANTS_PATH = Path("runs/manifests/m2_verify_h002_invariants.json")

ARMS = [
    ("C1", "verify_h002_c1", "h002_c1"),
    ("T1", "verify_h002_t1", "h002_t1"),
    ("C5_elem", "verify_h002_c5_elem", "h002_c5_elem"),
    ("C5_rowperm", "verify_h002_c5_rowperm", "h002_c5_rowperm"),
]


def _load_digit_rows(ckpt_dir: Path) -> torch.Tensor:
    shard = _resolve_tensor_file(ckpt_dir, EMBED_TENSOR)
    with safe_open(shard, framework="pt", device="cpu") as f:
        return f.get_slice(EMBED_TENSOR)[15:25].to(torch.float64)


def _assert_arm_invariants(label: str, record: dict[str, Any]) -> None:
    inv = record["invariants"]
    prov = record["provenance"]
    assert inv["total_params"] == 873_438_784, f"{label}: unexpected total_params {inv['total_params']}"
    assert inv["frozen_tensors_verified"] == 153, (
        f"{label}: expected 153 frozen vision tensors, got {inv['frozen_tensors_verified']}"
    )
    assert inv["changed_tensor_count"] == 1, (
        f"{label}: expected 1 changed tensor, got {inv['changed_tensor_count']}"
    )
    assert inv["changed_tensors"][0]["name"] == EMBED_TENSOR, (
        f"{label}: unexpected changed tensor {inv['changed_tensors'][0]['name']}"
    )
    assert prov["donor_forward_passes"] == 0, f"{label}: non-zero donor_forward_passes"
    assert prov["gradient_steps"] == 0, f"{label}: non-zero gradient_steps"


def _build_rowperm_donor_override(donor_dir: Path, seed: int = 42) -> dict[str, torch.Tensor]:
    shard = _resolve_tensor_file(donor_dir, EMBED_TENSOR)
    with safe_open(shard, framework="pt", device="cpu") as f:
        t = f.get_tensor(EMBED_TENSOR)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(t.shape[0], generator=gen)
    return {EMBED_TENSOR: t.index_select(0, perm)}


def run_invariants() -> dict[str, Any]:
    """Replay C1/T1, run float64 reversion, build C5 controls, and verify all weight invariants."""
    EDITED_ROOT.mkdir(parents=True, exist_ok=True)
    Path("runs/manifests").mkdir(parents=True, exist_ok=True)

    base_shard = _resolve_tensor_file(RECIPIENT_DIR, EMBED_TENSOR)
    base_sha = sha256_file(base_shard)
    if base_sha != EXPECTED_BASE_SHA256:
        raise AssertionError(f"Pristine recipient SHA-256 mismatch: {base_sha} != {EXPECTED_BASE_SHA256}")

    # Clean any stale verification directories and manifests (leave Codex's h002_c1_low / h002_t1_low untouched)
    for _, dir_name, tag in ARMS:
        target = EDITED_ROOT / dir_name
        if target.exists():
            shutil.rmtree(target)
        manifest = Path(f"runs/manifests/m2_{tag}_patch.json")
        if manifest.exists():
            manifest.unlink()
    for rev_name in ("tmp_revert_c1", "tmp_revert_t1"):
        rev_dir = EDITED_ROOT / rev_name
        if rev_dir.exists():
            shutil.rmtree(rev_dir)

    c1_dir = EDITED_ROOT / "verify_h002_c1"
    c1_manifest = Path("runs/manifests/m2_h002_c1_patch.json")
    c1_rec = apply_c1(RECIPIENT_DIR, c1_dir, c1_manifest)
    c1_sha = sha256_file(_resolve_tensor_file(c1_dir, EMBED_TENSOR))
    if c1_sha != EXPECTED_C1_SHA256:
        raise AssertionError(f"C1 replay SHA-256 mismatch: {c1_sha} != {EXPECTED_C1_SHA256}")
    codex_c1_dir = EDITED_ROOT / "h002_c1_low"
    c1_matches_codex_dir = (
        sha256_file(_resolve_tensor_file(codex_c1_dir, EMBED_TENSOR)) == c1_sha
        if codex_c1_dir.exists()
        else None
    )
    if c1_matches_codex_dir is False:
        raise AssertionError("C1 replay differs from hf_cache/edited/h002_c1_low shard")

    t1_dir = EDITED_ROOT / "verify_h002_t1"
    t1_manifest = Path("runs/manifests/m2_h002_t1_patch.json")
    t1_rec = apply_t1(DONOR_DIR, RECIPIENT_DIR, t1_dir, t1_manifest)
    t1_sha = sha256_file(_resolve_tensor_file(t1_dir, EMBED_TENSOR))
    if t1_sha != EXPECTED_T1_SHA256:
        raise AssertionError(f"T1 replay SHA-256 mismatch: {t1_sha} != {EXPECTED_T1_SHA256}")
    codex_t1_dir = EDITED_ROOT / "h002_t1_low"
    t1_matches_codex_dir = (
        sha256_file(_resolve_tensor_file(codex_t1_dir, EMBED_TENSOR)) == t1_sha
        if codex_t1_dir.exists()
        else None
    )
    if t1_matches_codex_dir is False:
        raise AssertionError("T1 replay differs from hf_cache/edited/h002_t1_low shard")

    # float64 bit-exact delta reversion checks
    rev_c1_dir = EDITED_ROOT / "tmp_revert_c1"
    rev_c1_inv = verify_exact_reversion(RECIPIENT_DIR, c1_dir, rev_c1_dir)
    assert rev_c1_inv["changed_params"] == 0 and rev_c1_inv["frozen_tensors_verified"] == 153
    shutil.rmtree(rev_c1_dir)

    rev_t1_dir = EDITED_ROOT / "tmp_revert_t1"
    rev_t1_inv = verify_exact_reversion(RECIPIENT_DIR, t1_dir, rev_t1_dir)
    assert rev_t1_inv["changed_params"] == 0 and rev_t1_inv["frozen_tensors_verified"] == 153
    shutil.rmtree(rev_t1_dir)

    # C5 shuffled-donor controls
    c5_elem_dir = EDITED_ROOT / "verify_h002_c5_elem"
    c5_elem_manifest = Path("runs/manifests/m2_h002_c5_elem_patch.json")
    shuf_elem = build_shuffled_donor_override(DONOR_DIR, target_layers=[], include_embed=True, seed=42)
    c5_elem_rec = apply_t1(
        DONOR_DIR, RECIPIENT_DIR, c5_elem_dir, c5_elem_manifest, donor_override=shuf_elem
    )
    del shuf_elem

    c5_rowperm_dir = EDITED_ROOT / "verify_h002_c5_rowperm"
    c5_rowperm_manifest = Path("runs/manifests/m2_h002_c5_rowperm_patch.json")
    shuf_rowperm = _build_rowperm_donor_override(DONOR_DIR, seed=42)
    c5_rowperm_rec = apply_t1(
        DONOR_DIR, RECIPIENT_DIR, c5_rowperm_dir, c5_rowperm_manifest, donor_override=shuf_rowperm
    )
    del shuf_rowperm

    arm_records = {
        "C1": (c1_rec, c1_dir),
        "T1": (t1_rec, t1_dir),
        "C5_elem": (c5_elem_rec, c5_elem_dir),
        "C5_rowperm": (c5_rowperm_rec, c5_rowperm_dir),
    }
    for label, (rec, _) in arm_records.items():
        _assert_arm_invariants(label, rec)

    # Pairwise cosine similarities and Frobenius norms of the 10-row digit deltas
    base_digits = _load_digit_rows(RECIPIENT_DIR)
    deltas = {label: (_load_digit_rows(ckpt) - base_digits) for label, (_, ckpt) in arm_records.items()}

    delta_frob = {k: float(torch.linalg.vector_norm(v).item()) for k, v in deltas.items()}
    cosine_matrix: dict[str, dict[str, float]] = {}
    flat = {k: v.reshape(-1) for k, v in deltas.items()}
    for a in deltas:
        cosine_matrix[a] = {}
        for b in deltas:
            cos = float(
                (flat[a] @ flat[b]) / (torch.linalg.vector_norm(flat[a]) * torch.linalg.vector_norm(flat[b]))
            )
            cosine_matrix[a][b] = round(cos, 6)

    # Verify C5 controls produce genuinely distinct deltas from both T1 and C1
    for c5_name in ("C5_elem", "C5_rowperm"):
        assert not torch.equal(deltas[c5_name], deltas["T1"]), f"{c5_name} collapsed to T1"
        assert not torch.equal(deltas[c5_name], deltas["C1"]), f"{c5_name} collapsed to C1"

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pristine_recipient_sha256": base_sha,
        "replay_verification": {
            "C1_sha256": c1_sha,
            "C1_matches_codex_expected": c1_sha == EXPECTED_C1_SHA256,
            "C1_matches_codex_dir": c1_matches_codex_dir,
            "T1_sha256": t1_sha,
            "T1_matches_codex_expected": t1_sha == EXPECTED_T1_SHA256,
            "T1_matches_codex_dir": t1_matches_codex_dir,
        },
        "float64_reversion": {
            "C1": {
                "changed_params": rev_c1_inv["changed_params"],
                "frozen_tensors_verified": rev_c1_inv["frozen_tensors_verified"],
            },
            "T1": {
                "changed_params": rev_t1_inv["changed_params"],
                "frozen_tensors_verified": rev_t1_inv["frozen_tensors_verified"],
            },
        },
        "arms": {
            label: {
                "provenance": rec["provenance"],
                "invariants": rec["invariants"],
            }
            for label, (rec, _) in arm_records.items()
        },
        "digit_delta_geometry": {
            "delta_frobenius_norms_f64": delta_frob,
            "pairwise_cosine_similarity": cosine_matrix,
        },
    }

    INVARIANTS_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    compact = {
        "replay_C1_bit_exact": True,
        "replay_T1_bit_exact": True,
        "reversion_C1_changed_params": rev_c1_inv["changed_params"],
        "reversion_T1_changed_params": rev_t1_inv["changed_params"],
        "delta_frobenius_norms_f64": {k: round(v, 6) for k, v in delta_frob.items()},
        "pairwise_cosine_similarity": cosine_matrix,
    }
    print(json.dumps(compact, indent=2), flush=True)
    return summary


def _try_reuse_eval(manifest_path: Path, log_path: Path, expected_n: int) -> dict[str, Any] | None:
    if not manifest_path.exists() or not log_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("metrics", {}).get("n") != expected_n:
            return None
        with log_path.open("r", encoding="utf-8") as f:
            lines = sum(1 for ln in f if ln.strip())
        if lines != expected_n:
            return None
        return manifest
    except Exception:
        return None


def _load_outcomes(log_path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with log_path.open("r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            rows[str(r["id"])] = r
    return rows


def _paired_stats(
    rows_a: dict[str, dict[str, Any]],
    rows_b: dict[str, dict[str, Any]],
    subskill: str | None = None,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    common_ids = sorted(set(rows_a) & set(rows_b))
    if subskill is not None:
        common_ids = [i for i in common_ids if rows_a[i].get("subskill") == subskill]
    n = len(common_ids)
    if n == 0:
        raise ValueError(f"No common IDs for subskill={subskill}")

    a_vec = np.array([1 if rows_a[i]["correct"] else 0 for i in common_ids], dtype=np.int64)
    b_vec = np.array([1 if rows_b[i]["correct"] else 0 for i in common_ids], dtype=np.int64)

    both_correct = int(np.sum((a_vec == 1) & (b_vec == 1)))
    both_incorrect = int(np.sum((a_vec == 0) & (b_vec == 0)))
    improved = int(np.sum((a_vec == 0) & (b_vec == 1)))  # b fixes a's error
    regressed = int(np.sum((a_vec == 1) & (b_vec == 0)))  # b breaks a's success

    discordant = improved + regressed
    mcnemar_p = float(binomtest(min(improved, regressed), discordant, 0.5).pvalue) if discordant > 0 else 1.0

    diff_vec = (b_vec - a_vec).astype(np.float64)
    delta_pp = float(np.mean(diff_vec) * 100.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff_vec[idx].mean(axis=1) * 100.0
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return {
        "n": n,
        "acc_a": float(np.mean(a_vec)),
        "acc_b": float(np.mean(b_vec)),
        "delta_pp": round(delta_pp, 2),
        "bootstrap_95ci_pp": [round(float(ci_low), 2), round(float(ci_high), 2)],
        "improved_b_over_a": improved,
        "regressed_b_vs_a": regressed,
        "both_correct": both_correct,
        "both_incorrect": both_incorrect,
        "mcnemar_exact_p": mcnemar_p,
    }


def run_eval(device: str, dtype_name: str, output_path: Path, keep_temp: bool) -> dict[str, Any]:
    """Evaluate replayed C1, T1, C5_elem, C5_rowperm on data/validation/ and compute paired stats."""
    if not INVARIANTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INVARIANTS_PATH}; run '--stage invariants' first."
        )
    invariants_data = json.loads(INVARIANTS_PATH.read_text(encoding="utf-8"))

    validation_results: dict[str, Any] = {}
    for label, dir_name, tag in ARMS:
        ckpt_dir = EDITED_ROOT / dir_name
        syn_manifest_p = Path(f"runs/manifests/m2_{tag}_synthetic_val.json")
        syn_log_p = Path(f"runs/logs/m2_{tag}_synthetic_val.jsonl")
        gsm_manifest_p = Path(f"runs/manifests/m2_{tag}_gsm8k_val.json")
        gsm_log_p = Path(f"runs/logs/m2_{tag}_gsm8k_val.jsonl")

        syn_man = _try_reuse_eval(syn_manifest_p, syn_log_p, expected_n=150)
        if syn_man is None:
            if not ckpt_dir.exists():
                raise FileNotFoundError(f"Missing checkpoint directory {ckpt_dir}; run invariants first.")
            syn_man = evaluate_jsonl(
                model_path=ckpt_dir,
                dataset_path="data/validation/synthetic_val.jsonl",
                run_id=f"m2_{tag}_synthetic_val",
                output_manifest_path=syn_manifest_p,
                output_log_path=syn_log_p,
                enable_thinking=False,
                max_new_tokens=1024,
                per_subskill_limit=50,
                device=device,
                dtype_name=dtype_name,
            )

        gsm_man = _try_reuse_eval(gsm_manifest_p, gsm_log_p, expected_n=100)
        if gsm_man is None:
            if not ckpt_dir.exists():
                raise FileNotFoundError(f"Missing checkpoint directory {ckpt_dir}; run invariants first.")
            gsm_man = evaluate_jsonl(
                model_path=ckpt_dir,
                dataset_path="data/validation/gsm8k_val.jsonl",
                run_id=f"m2_{tag}_gsm8k_val",
                output_manifest_path=gsm_manifest_p,
                output_log_path=gsm_log_p,
                enable_thinking=False,
                max_new_tokens=1024,
                limit=100,
                device=device,
                dtype_name=dtype_name,
            )

        validation_results[label] = {
            "patch_invariants": invariants_data["arms"][label]["invariants"],
            "provenance": invariants_data["arms"][label]["provenance"],
            "synthetic_val": syn_man["metrics"],
            "gsm8k_val": gsm_man["metrics"],
            "synth_tok_s": syn_man["tokens_per_sec"],
            "gsm8k_tok_s": gsm_man["tokens_per_sec"],
            "synth_eval_sec": syn_man["eval_sec"],
            "gsm8k_eval_sec": gsm_man["eval_sec"],
            "peak_mps_allocated_mb": max(
                float(syn_man.get("peak_mps_allocated_mb") or 0.0),
                float(gsm_man.get("peak_mps_allocated_mb") or 0.0),
            ),
        }

        print(
            f"[{label}] gsm8k_val={gsm_man['metrics']['pass_at_1']:.3f} "
            f"carry={syn_man['metrics']['by_subskill']['carry_add_sub']['pass_at_1']:.3f} "
            f"mult={syn_man['metrics']['by_subskill']['mult_2x2_3x1']['pass_at_1']:.3f} "
            f"word={syn_man['metrics']['by_subskill']['word_2step']['pass_at_1']:.3f} "
            f"tok/s={gsm_man['tokens_per_sec']}",
            flush=True,
        )

        if not keep_temp and ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)

    # Load item-level outcomes for paired tests
    logs = {
        "C0_raw": {
            "synth": _load_outcomes(Path("runs/logs/stage0_qwen3.5_0.8b_raw_synthetic_nothink_bfloat16.jsonl")),
            "gsm8k": _load_outcomes(Path("runs/logs/stage0_qwen3.5_0.8b_raw_gsm8k_nothink_bfloat16.jsonl")),
        },
        "C4_weights": {
            "synth": _load_outcomes(Path("runs/logs/m2_c4_weights_synthetic_val.jsonl")),
            "gsm8k": _load_outcomes(Path("runs/logs/m2_c4_weights_gsm8k_val.jsonl")),
        },
        "C1": {
            "synth": _load_outcomes(Path("runs/logs/m2_h002_c1_synthetic_val.jsonl")),
            "gsm8k": _load_outcomes(Path("runs/logs/m2_h002_c1_gsm8k_val.jsonl")),
        },
        "T1": {
            "synth": _load_outcomes(Path("runs/logs/m2_h002_t1_synthetic_val.jsonl")),
            "gsm8k": _load_outcomes(Path("runs/logs/m2_h002_t1_gsm8k_val.jsonl")),
        },
        "C5_elem": {
            "synth": _load_outcomes(Path("runs/logs/m2_h002_c5_elem_synthetic_val.jsonl")),
            "gsm8k": _load_outcomes(Path("runs/logs/m2_h002_c5_elem_gsm8k_val.jsonl")),
        },
        "C5_rowperm": {
            "synth": _load_outcomes(Path("runs/logs/m2_h002_c5_rowperm_synthetic_val.jsonl")),
            "gsm8k": _load_outcomes(Path("runs/logs/m2_h002_c5_rowperm_gsm8k_val.jsonl")),
        },
    }

    pairs = [
        ("C0_raw", "C1"),
        ("C0_raw", "T1"),
        ("C0_raw", "C5_elem"),
        ("C0_raw", "C5_rowperm"),
        ("C1", "T1"),
        ("C5_elem", "T1"),
        ("C5_rowperm", "T1"),
        ("C4_weights", "T1"),
    ]
    paired_comparisons: dict[str, dict[str, Any]] = {}
    for arm_a, arm_b in pairs:
        key = f"{arm_a}_to_{arm_b}"
        paired_comparisons[key] = {
            "gsm8k_val": _paired_stats(logs[arm_a]["gsm8k"], logs[arm_b]["gsm8k"]),
            "synthetic_val": _paired_stats(logs[arm_a]["synth"], logs[arm_b]["synth"]),
            "carry_add_sub": _paired_stats(logs[arm_a]["synth"], logs[arm_b]["synth"], subskill="carry_add_sub"),
            "mult_2x2_3x1": _paired_stats(logs[arm_a]["synth"], logs[arm_b]["synth"], subskill="mult_2x2_3x1"),
            "word_2step": _paired_stats(logs[arm_a]["synth"], logs[arm_b]["synth"], subskill="word_2step"),
        }

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "milestone": "M2b_H002_verification",
        "device": device,
        "dtype": dtype_name,
        "invariants_and_replay": invariants_data,
        "validation": validation_results,
        "paired_comparisons": paired_comparisons,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    compact = {
        "validation": {
            k: {
                "gsm8k_val": v["gsm8k_val"]["pass_at_1"],
                "synthetic_val": round(v["synthetic_val"]["pass_at_1"], 4),
                "carry_add_sub": v["synthetic_val"]["by_subskill"]["carry_add_sub"]["pass_at_1"],
                "mult_2x2_3x1": v["synthetic_val"]["by_subskill"]["mult_2x2_3x1"]["pass_at_1"],
                "word_2step": v["synthetic_val"]["by_subskill"]["word_2step"]["pass_at_1"],
                "gsm8k_tok_s": v["gsm8k_tok_s"],
            }
            for k, v in validation_results.items()
        },
        "key_paired_gsm8k": {
            k: {
                "delta_pp": paired_comparisons[k]["gsm8k_val"]["delta_pp"],
                "ci95_pp": paired_comparisons[k]["gsm8k_val"]["bootstrap_95ci_pp"],
                "improved": paired_comparisons[k]["gsm8k_val"]["improved_b_over_a"],
                "regressed": paired_comparisons[k]["gsm8k_val"]["regressed_b_vs_a"],
                "p": round(paired_comparisons[k]["gsm8k_val"]["mcnemar_exact_p"], 4),
            }
            for k in ("C0_raw_to_C1", "C0_raw_to_T1", "C1_to_T1", "C5_elem_to_T1", "C5_rowperm_to_T1")
        },
        "key_paired_carry": {
            k: {
                "delta_pp": paired_comparisons[k]["carry_add_sub"]["delta_pp"],
                "ci95_pp": paired_comparisons[k]["carry_add_sub"]["bootstrap_95ci_pp"],
                "improved": paired_comparisons[k]["carry_add_sub"]["improved_b_over_a"],
                "regressed": paired_comparisons[k]["carry_add_sub"]["regressed_b_vs_a"],
                "p": round(paired_comparisons[k]["carry_add_sub"]["mcnemar_exact_p"], 4),
            }
            for k in ("C0_raw_to_C1", "C0_raw_to_T1", "C1_to_T1", "C5_elem_to_T1", "C5_rowperm_to_T1")
        },
    }
    print(json.dumps(compact, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("invariants", "eval", "all"), default="all")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", default="results/m2_h002_verification.json")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    if args.stage in ("invariants", "all"):
        run_invariants()
    if args.stage in ("eval", "all"):
        run_eval(
            device=args.device,
            dtype_name=args.dtype,
            output_path=Path(args.output),
            keep_temp=args.keep_temp,
        )


if __name__ == "__main__":
    main()
