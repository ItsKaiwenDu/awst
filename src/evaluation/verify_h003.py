"""Independent Antigravity verification of H003 (T2 replay, float64 reversion, C5 controls, and blinded validation)."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
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

from runs.codex_h003.apply_t2 import apply_t2
from src.evaluation.runner import evaluate_jsonl
from src.interventions.patch_applier import verify_exact_reversion
from src.weight_io.safetensors_reader import _resolve_tensor_file, sha256_file

KEY_A = "model.language_model.layers.13.linear_attn.A_log"
KEY_D = "model.language_model.layers.13.linear_attn.dt_bias"
HEAD_IDX = 12

EXPECTED_BASE_SHA256 = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
EXPECTED_T2_SHA256 = "ea5c744fe4d6627c2faefee13e2c2ad97c44f3ad012c7adc491d60290f6bb732"
EXPECTED_C5_PERM_SHA256 = "f47af66b18e02bf65920f47b750616cf2301da02c9a1ea31cb9972471926708c"

RECIPIENT_DIR = Path("hf_cache/models/qwen3.5_0.8b")
DONOR_DIR = Path("hf_cache/models/qwen3.5_2b")
EDITED_ROOT = Path("hf_cache/edited")
LOCALIZATION_PATH = Path("runs/codex_h003/localization.json")
INVARIANTS_PATH = Path("runs/manifests/m2_verify_h003_invariants.json")

ARMS = [
    ("T2", "verify_h003_t2", "h003_t2"),
    ("C5_perm", "verify_h003_c5_perm", "h003_c5_perm"),
    ("C5_norm_matched", "verify_h003_c5_norm_matched", "h003_c5_norm_matched"),
]


class _InaccessibleDonor:
    """Sentinel donor path asserting that donor_override never accesses donor_dir."""

    def __fspath__(self) -> str:
        raise AssertionError("donor_override must bypass resolving donor_dir")


def _load_head_vectors(ckpt_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    shard_a = _resolve_tensor_file(ckpt_dir, KEY_A)
    shard_d = _resolve_tensor_file(ckpt_dir, KEY_D)
    with safe_open(shard_a, framework="pt", device="cpu") as fa:
        vec_a = fa.get_tensor(KEY_A).clone()
    if shard_d == shard_a:
        with safe_open(shard_d, framework="pt", device="cpu") as fd:
            vec_d = fd.get_tensor(KEY_D).clone()
    else:
        with safe_open(shard_d, framework="pt", device="cpu") as fd:
            vec_d = fd.get_tensor(KEY_D).clone()
    return vec_a, vec_d


def _assert_arm_invariants(label: str, record: dict[str, Any]) -> None:
    inv = record["invariants"]
    prov = record["provenance"]
    assert inv["total_params"] == 873_438_784, f"{label}: unexpected total_params {inv['total_params']}"
    assert inv["frozen_tensors_verified"] == 153, (
        f"{label}: expected 153 frozen vision tensors, got {inv['frozen_tensors_verified']}"
    )
    assert inv["changed_params"] == 2, f"{label}: expected 2 changed params, got {inv['changed_params']}"
    changed_names = {t["name"] for t in inv["changed_tensors"]}
    assert changed_names.issubset({KEY_A, KEY_D}) and len(changed_names) == 2, (
        f"{label}: unexpected changed tensors {changed_names}"
    )
    assert prov["donor_forward_passes"] == 0, f"{label}: non-zero donor_forward_passes"
    assert prov["gradient_steps"] == 0, f"{label}: non-zero gradient_steps"


def run_invariants() -> dict[str, Any]:
    """Replay T2, test override & float64 reversion, build C5 controls, and verify all invariants."""
    EDITED_ROOT.mkdir(parents=True, exist_ok=True)
    Path("runs/manifests").mkdir(parents=True, exist_ok=True)

    base_shard = _resolve_tensor_file(RECIPIENT_DIR, KEY_A)
    base_sha = sha256_file(base_shard)
    if base_sha != EXPECTED_BASE_SHA256:
        raise AssertionError(f"Pristine recipient SHA-256 mismatch: {base_sha} != {EXPECTED_BASE_SHA256}")

    # Clean stale verification directories and manifests (never touch h003_half or h003_full)
    for _, dir_name, tag in ARMS:
        target = EDITED_ROOT / dir_name
        if target.exists():
            shutil.rmtree(target)
        manifest = Path(f"runs/manifests/m2_{tag}_patch.json")
        if manifest.exists():
            manifest.unlink()
    rev_dir = EDITED_ROOT / "tmp_revert_h003_t2"
    if rev_dir.exists():
        shutil.rmtree(rev_dir)

    # 1. Replay T2
    t2_dir = EDITED_ROOT / "verify_h003_t2"
    t2_manifest = Path("runs/manifests/m2_h003_t2_patch.json")
    t2_rec = apply_t2(DONOR_DIR, RECIPIENT_DIR, t2_dir, t2_manifest)
    t2_sha = sha256_file(_resolve_tensor_file(t2_dir, KEY_A))
    if t2_sha != EXPECTED_T2_SHA256:
        raise AssertionError(f"T2 replay SHA-256 mismatch: {t2_sha} != {EXPECTED_T2_SHA256}")
    codex_t2_dir = EDITED_ROOT / "h003_half"
    t2_matches_codex_dir = (
        sha256_file(_resolve_tensor_file(codex_t2_dir, KEY_A)) == t2_sha
        if codex_t2_dir.exists()
        else None
    )
    if t2_matches_codex_dir is False:
        raise AssertionError("T2 replay differs from hf_cache/edited/h003_half shard")

    # 2. Verify donor_override contract with exact donor vectors + inaccessible sentinel
    dA, dD = _load_head_vectors(DONOR_DIR)
    rA, rD = _load_head_vectors(RECIPIENT_DIR)
    with tempfile.TemporaryDirectory(prefix="awst-h003-ovcheck-", dir=EDITED_ROOT) as tmp_dir:
        tmp_path = Path(tmp_dir)
        ov_rec = apply_t2(
            _InaccessibleDonor(),
            RECIPIENT_DIR,
            tmp_path / "ckpt",
            tmp_path / "patch.json",
            donor_override={KEY_A: dA, KEY_D: dD},
        )
        identical_override_sha = ov_rec["edited_shard_sha256"]
        if identical_override_sha != EXPECTED_T2_SHA256:
            raise AssertionError(
                f"Identical donor_override SHA-256 mismatch: {identical_override_sha} != {EXPECTED_T2_SHA256}"
            )

    # 3. Verify float64 bit-exact delta reversion
    rev_inv = verify_exact_reversion(RECIPIENT_DIR, t2_dir, rev_dir)
    rev_shard_sha = sha256_file(_resolve_tensor_file(rev_dir, KEY_A))
    assert rev_inv["changed_params"] == 0, f"Expected 0 differing params, got {rev_inv['changed_params']}"
    assert rev_inv["frozen_tensors_verified"] == 153
    assert rev_shard_sha == EXPECTED_BASE_SHA256, "Reverted shard SHA-256 differs from pristine base"
    shutil.rmtree(rev_dir)

    # 4. Canonical C5_perm (joint seed=42 permutation of the 16-head donor vectors)
    perm = torch.randperm(16, generator=torch.Generator(device="cpu").manual_seed(42))
    c5_perm_dir = EDITED_ROOT / "verify_h003_c5_perm"
    c5_perm_manifest = Path("runs/manifests/m2_h003_c5_perm_patch.json")
    c5_perm_rec = apply_t2(
        _InaccessibleDonor(),
        RECIPIENT_DIR,
        c5_perm_dir,
        c5_perm_manifest,
        donor_override={KEY_A: dA[perm], KEY_D: dD[perm]},
    )
    c5_perm_sha = sha256_file(_resolve_tensor_file(c5_perm_dir, KEY_A))
    if c5_perm_sha != EXPECTED_C5_PERM_SHA256:
        raise AssertionError(
            f"C5_perm SHA-256 mismatch: {c5_perm_sha} != {EXPECTED_C5_PERM_SHA256}"
        )

    # 5. Norm-matched C5_norm_matched (preserves perm[12] direction scaled to T2's 2D displacement norm)
    loc = json.loads(LOCALIZATION_PATH.read_text(encoding="utf-8"))
    offset = float(loc["selected"]["donor_digit_a_median"] - loc["selected"]["recipient_digit_a_median"])
    v_t2 = torch.tensor(
        [float(dA[HEAD_IDX]) - float(rA[HEAD_IDX]), float(dD[HEAD_IDX]) + offset - float(rD[HEAD_IDX])],
        dtype=torch.float64,
    )
    h_shuf = int(perm[HEAD_IDX].item())
    v_perm = torch.tensor(
        [float(dA[h_shuf]) - float(rA[HEAD_IDX]), float(dD[h_shuf]) + offset - float(rD[HEAD_IDX])],
        dtype=torch.float64,
    )
    v_matched = v_perm * (torch.linalg.vector_norm(v_t2) / torch.linalg.vector_norm(v_perm))
    mA = dA[perm].clone()
    mD = dD[perm].clone()
    mA[HEAD_IDX] = torch.tensor(float(rA[HEAD_IDX]) + float(v_matched[0]), dtype=torch.float32)
    mD[HEAD_IDX] = torch.tensor(float(rD[HEAD_IDX]) + float(v_matched[1]) - offset, dtype=torch.bfloat16)

    c5_nm_dir = EDITED_ROOT / "verify_h003_c5_norm_matched"
    c5_nm_manifest = Path("runs/manifests/m2_h003_c5_norm_matched_patch.json")
    c5_nm_rec = apply_t2(
        _InaccessibleDonor(),
        RECIPIENT_DIR,
        c5_nm_dir,
        c5_nm_manifest,
        donor_override={KEY_A: mA, KEY_D: mD},
    )

    rel_t2 = float(t2_rec["overall_relative_frobenius_norm"])
    rel_c5nm = float(c5_nm_rec["overall_relative_frobenius_norm"])
    rel_diff_frac = abs(rel_c5nm - rel_t2) / max(rel_t2, 1e-12)
    assert rel_diff_frac <= 0.05, f"C5_norm_matched rel_frob {rel_c5nm} drifted >5% from T2 {rel_t2}"

    arm_records = {
        "T2": (t2_rec, t2_dir),
        "C5_perm": (c5_perm_rec, c5_perm_dir),
        "C5_norm_matched": (c5_nm_rec, c5_nm_dir),
    }
    for label, (rec, _) in arm_records.items():
        _assert_arm_invariants(label, rec)

    # 2D delta geometry from stored weights
    deltas: dict[str, np.ndarray] = {}
    for label, (_, ckpt) in arm_records.items():
        eA, eD = _load_head_vectors(ckpt)
        deltas[label] = np.array(
            [float(eA[HEAD_IDX].item()) - float(rA[HEAD_IDX].item()),
             float(eD[HEAD_IDX].item()) - float(rD[HEAD_IDX].item())],
            dtype=np.float64,
        )

    norms = {k: float(np.linalg.norm(v)) for k, v in deltas.items()}
    ratios = {k: float(norms[k] / max(norms["T2"], 1e-12)) for k in deltas}
    cosines: dict[str, dict[str, float]] = {}
    for a, va in deltas.items():
        cosines[a] = {}
        for b, vb in deltas.items():
            cosines[a][b] = round(float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))), 6)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pristine_recipient_sha256": base_sha,
        "replay_verification": {
            "T2_sha256": t2_sha,
            "T2_matches_codex_expected": t2_sha == EXPECTED_T2_SHA256,
            "T2_matches_codex_dir": t2_matches_codex_dir,
            "identical_override_sha256": identical_override_sha,
            "identical_override_matches_T2": identical_override_sha == EXPECTED_T2_SHA256,
            "C5_perm_sha256": c5_perm_sha,
            "C5_perm_matches_codex_shuffled_receipt": c5_perm_sha == EXPECTED_C5_PERM_SHA256,
        },
        "float64_reversion": {
            "changed_params": rev_inv["changed_params"],
            "frozen_tensors_verified": rev_inv["frozen_tensors_verified"],
            "reverted_shard_sha256": rev_shard_sha,
            "matches_pristine_base": rev_shard_sha == EXPECTED_BASE_SHA256,
        },
        "arms": {
            label: {
                "provenance": rec["provenance"],
                "invariants": rec["invariants"],
            }
            for label, (rec, _) in arm_records.items()
        },
        "delta_2d_geometry": {
            "head_index": HEAD_IDX,
            "recipient_base_values": {"A_log": float(rA[HEAD_IDX].item()), "dt_bias": float(rD[HEAD_IDX].item())},
            "activation_offset_dt_bias": offset,
            "c5_perm_head_mapping": perm.tolist(),
            "c5_shuffled_donor_head": h_shuf,
            "c5_norm_matched_rel_diff_frac": round(rel_diff_frac, 6),
            "deltas_a_log_dt_bias": {k: [round(float(x), 8) for x in v.tolist()] for k, v in deltas.items()},
            "delta_l2_norms": {k: round(v, 8) for k, v in norms.items()},
            "norm_ratio_vs_t2": {k: round(v, 4) for k, v in ratios.items()},
            "pairwise_cosine_similarity": cosines,
        },
    }

    INVARIANTS_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    compact = {
        "replay_T2_bit_exact": True,
        "identical_override_bit_exact": True,
        "c5_perm_matches_codex_receipt": True,
        "float64_reversion_changed_params": rev_inv["changed_params"],
        "c5_norm_matched_rel_diff_pct": round(rel_diff_frac * 100.0, 2),
        "deltas_a_log_dt_bias": summary["delta_2d_geometry"]["deltas_a_log_dt_bias"],
        "norm_ratio_vs_t2": summary["delta_2d_geometry"]["norm_ratio_vs_t2"],
        "pairwise_cosine_similarity": cosines,
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
    improved = int(np.sum((a_vec == 0) & (b_vec == 1)))
    regressed = int(np.sum((a_vec == 1) & (b_vec == 0)))

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
    """Evaluate replayed T2, C5_perm, and C5_norm_matched on data/validation/ and compute paired stats."""
    if not INVARIANTS_PATH.exists():
        raise FileNotFoundError(f"Missing {INVARIANTS_PATH}; run '--stage invariants' first.")
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

    logs = {
        "C0_raw": {
            "synth": _load_outcomes(Path("runs/logs/stage0_qwen3.5_0.8b_raw_synthetic_nothink_bfloat16.jsonl")),
            "gsm8k": _load_outcomes(Path("runs/logs/stage0_qwen3.5_0.8b_raw_gsm8k_nothink_bfloat16.jsonl")),
        },
        "C4_act": {
            "synth": _load_outcomes(Path("runs/logs/m2_c4_act_synthetic_val.jsonl")),
            "gsm8k": _load_outcomes(Path("runs/logs/m2_c4_act_gsm8k_val.jsonl")),
        },
        "T2": {
            "synth": _load_outcomes(Path("runs/logs/m2_h003_t2_synthetic_val.jsonl")),
            "gsm8k": _load_outcomes(Path("runs/logs/m2_h003_t2_gsm8k_val.jsonl")),
        },
        "C5_perm": {
            "synth": _load_outcomes(Path("runs/logs/m2_h003_c5_perm_synthetic_val.jsonl")),
            "gsm8k": _load_outcomes(Path("runs/logs/m2_h003_c5_perm_gsm8k_val.jsonl")),
        },
        "C5_norm_matched": {
            "synth": _load_outcomes(Path("runs/logs/m2_h003_c5_norm_matched_synthetic_val.jsonl")),
            "gsm8k": _load_outcomes(Path("runs/logs/m2_h003_c5_norm_matched_gsm8k_val.jsonl")),
        },
    }

    pairs = [
        ("C0_raw", "T2"),
        ("C0_raw", "C5_perm"),
        ("C0_raw", "C5_norm_matched"),
        ("C4_act", "T2"),
        ("C5_perm", "T2"),
        ("C5_norm_matched", "T2"),
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
        "milestone": "M2b_H003_verification",
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
            for k in ("C0_raw_to_T2", "C4_act_to_T2", "C5_perm_to_T2", "C5_norm_matched_to_T2")
        },
        "key_paired_carry": {
            k: {
                "delta_pp": paired_comparisons[k]["carry_add_sub"]["delta_pp"],
                "ci95_pp": paired_comparisons[k]["carry_add_sub"]["bootstrap_95ci_pp"],
                "improved": paired_comparisons[k]["carry_add_sub"]["improved_b_over_a"],
                "regressed": paired_comparisons[k]["carry_add_sub"]["regressed_b_vs_a"],
                "p": round(paired_comparisons[k]["carry_add_sub"]["mcnemar_exact_p"], 4),
            }
            for k in ("C0_raw_to_T2", "C4_act_to_T2", "C5_perm_to_T2", "C5_norm_matched_to_T2")
        },
    }
    print(json.dumps(compact, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("invariants", "eval", "all"), default="all")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", default="results/m2_h003_verification.json")
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
