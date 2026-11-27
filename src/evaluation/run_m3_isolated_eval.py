"""Milestone M3 frozen confirmatory evaluation on external /Users/kaiwendu/.awst_isolated_eval/."""

from __future__ import annotations

import argparse
import gc
import json
import random
import resource
import shutil
import sys
import time
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

from runs.codex_h002.apply_t1 import apply_t1
from runs.codex_h003.apply_t2 import apply_t2
from src.baselines.random_controls import build_c3_random_donor_slice
from src.evaluation.metrics import compute_eval_metrics, detect_repetition_loop
from src.evaluation.parser import answers_match
from src.evaluation.retention import evaluate_perplexity, evaluate_retention, holm_bonferroni
from src.evaluation.runner import format_prompt, load_dataset_rows, load_model_and_tokenizer
from src.interventions.patch_applier import verify_checkpoint_invariants
from src.weight_io.safetensors_reader import _resolve_tensor_file, sha256_file

RECIPIENT_DIR = Path("hf_cache/models/qwen3.5_0.8b")
DONOR_DIR = Path("hf_cache/models/qwen3.5_2b")
EDITED_ROOT = Path("hf_cache/edited")
DEFAULT_ISOLATED_DIR = Path("/Users/kaiwendu/.awst_isolated_eval")

EXPECTED_08B_SHA256 = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
EXPECTED_2B_SHA256 = "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"

DATASET_SHA256 = {
    "gsm8k_test": ("gsm8k_test.jsonl", "5315f256fccc1e33b51a13a820133a08959cb4ba77248917a157bdcabfe7c414"),
    "synthetic_test": ("synthetic_test.jsonl", "c81c74146cf3029586fd5d7602ad7e9f882c9a2b40f07bfa002391d1c3c023d3"),
    "synthetic_ood_test": ("synthetic_ood_test.jsonl", "4904d73b73c91bc3d8022a38c0493d99cb1e2263f4aa8c6257ebbbc33eb8281e"),
}

GSM8K_SAMPLE_SEED = 20261126
GSM8K_SAMPLE_N = 200
SYNTHETIC_PER_SUBSKILL = 50
TARGET_LAYERS_C3 = list(range(8, 24))

FROZEN_ARMS: tuple[tuple[str, str], ...] = (
    ("C0_raw", "persistent"),
    ("Qwen3.5-2B", "donor"),
    ("C1", "persistent"),
    ("C3", "temp"),
    ("C4_weights", "persistent"),
    ("C4_act", "persistent"),
    ("T1", "persistent"),
    ("C5_T1_rowperm", "temp"),
    ("T2", "persistent"),
    ("C5_T2_norm_matched", "temp"),
)

PERSISTENT_DIRS: dict[str, Path] = {
    "C0_raw": RECIPIENT_DIR,
    "Qwen3.5-2B": DONOR_DIR,
    "C1": EDITED_ROOT / "h002_c1_low",
    "C4_weights": EDITED_ROOT / "m2_c4_weights",
    "C4_act": EDITED_ROOT / "m2_c4_act",
    "T1": EDITED_ROOT / "h002_t1_low",
    "T2": EDITED_ROOT / "h003_half",
}

AUDIT_PATH = Path("runs/manifests/m3_frozen_checkpoint_audit.json")


def _primary_shard_sha256(ckpt_dir: Path) -> str:
    shard = _resolve_tensor_file(ckpt_dir, "model.language_model.embed_tokens.weight")
    return sha256_file(shard)


def _assert_recipient_invariants(label: str, inv: dict[str, Any]) -> None:
    if inv["total_params"] != 873_438_784:
        raise AssertionError(f"{label}: expected 873,438,784 params, got {inv['total_params']}")
    if inv["frozen_tensors_verified"] != 153:
        raise AssertionError(f"{label}: expected 153 frozen vision tensors, got {inv['frozen_tensors_verified']}")


def _build_temp_arm(arm: str) -> tuple[Path, dict[str, Any]]:
    """Build deterministic temporary control checkpoint (C3, C5_T1_rowperm, or C5_T2_norm_matched)."""
    if arm == "C3":
        out_dir = EDITED_ROOT / "m3_tmp_c3"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        rec = build_c3_random_donor_slice(
            donor_dir=DONOR_DIR,
            recipient_dir=RECIPIENT_DIR,
            out_dir=out_dir,
            target_layers=TARGET_LAYERS_C3,
            alpha=0.05,
            seed=42,
            manifest_path="runs/manifests/m3_C3_patch.json",
            overwrite=True,
        )
        _assert_recipient_invariants("C3", rec["invariants"])
        return out_dir, rec

    if arm == "C5_T1_rowperm":
        out_dir = EDITED_ROOT / "m3_tmp_c5_t1"
        manifest = Path("runs/manifests/m3_C5_T1_rowperm_patch.json")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        if manifest.exists():
            manifest.unlink()
        embed_key = "model.language_model.embed_tokens.weight"
        shard = _resolve_tensor_file(DONOR_DIR, embed_key)
        with safe_open(shard, framework="pt", device="cpu") as f:
            emb = f.get_tensor(embed_key)
        perm = torch.randperm(emb.shape[0], generator=torch.Generator(device="cpu").manual_seed(42))
        rowperm = emb.index_select(0, perm)
        rec = apply_t1(
            DONOR_DIR,
            RECIPIENT_DIR,
            out_dir,
            manifest,
            donor_override={embed_key: rowperm},
        )
        _assert_recipient_invariants("C5_T1_rowperm", rec["invariants"])
        return out_dir, rec

    if arm == "C5_T2_norm_matched":
        out_dir = EDITED_ROOT / "m3_tmp_c5_t2"
        manifest = Path("runs/manifests/m3_C5_T2_norm_matched_patch.json")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        if manifest.exists():
            manifest.unlink()
        key_a = "model.language_model.layers.13.linear_attn.A_log"
        key_d = "model.language_model.layers.13.linear_attn.dt_bias"
        head_idx = 12
        d_shard = _resolve_tensor_file(DONOR_DIR, key_a)
        r_shard = _resolve_tensor_file(RECIPIENT_DIR, key_a)
        with safe_open(d_shard, framework="pt", device="cpu") as fd, safe_open(
            r_shard, framework="pt", device="cpu"
        ) as fr:
            d_a, d_d = fd.get_tensor(key_a).clone(), fd.get_tensor(key_d).clone()
            r_a, r_d = fr.get_tensor(key_a).clone(), fr.get_tensor(key_d).clone()
        loc = json.loads(Path("runs/codex_h003/localization.json").read_text(encoding="utf-8"))
        offset = float(loc["selected"]["donor_digit_a_median"] - loc["selected"]["recipient_digit_a_median"])
        perm = torch.randperm(16, generator=torch.Generator(device="cpu").manual_seed(42))
        h_shuf = int(perm[head_idx].item())
        v_t2 = torch.tensor(
            [float(d_a[head_idx]) - float(r_a[head_idx]), float(d_d[head_idx]) + offset - float(r_d[head_idx])],
            dtype=torch.float64,
        )
        v_perm = torch.tensor(
            [float(d_a[h_shuf]) - float(r_a[head_idx]), float(d_d[h_shuf]) + offset - float(r_d[head_idx])],
            dtype=torch.float64,
        )
        v_matched = v_perm * (torch.linalg.vector_norm(v_t2) / torch.linalg.vector_norm(v_perm))
        m_a = d_a[perm].clone()
        m_d = d_d[perm].clone()
        m_a[head_idx] = torch.tensor(float(r_a[head_idx]) + float(v_matched[0]), dtype=torch.float32)
        m_d[head_idx] = torch.tensor(float(r_d[head_idx]) + float(v_matched[1]) - offset, dtype=torch.bfloat16)
        rec = apply_t2(
            DONOR_DIR,
            RECIPIENT_DIR,
            out_dir,
            manifest,
            donor_override={key_a: m_a, key_d: m_d},
        )
        _assert_recipient_invariants("C5_T2_norm_matched", rec["invariants"])
        return out_dir, rec

    raise ValueError(f"Unknown temporary arm: {arm}")


def run_audit() -> dict[str, Any]:
    """Verify frozen persistent checkpoints (shard SHA-256, 873.44M params, 153 frozen vision tensors)."""
    Path("runs/manifests").mkdir(parents=True, exist_ok=True)
    sha_08b = _primary_shard_sha256(RECIPIENT_DIR)
    sha_2b = _primary_shard_sha256(DONOR_DIR)
    if sha_08b != EXPECTED_08B_SHA256:
        raise AssertionError(f"0.8B base shard SHA-256 mismatch: {sha_08b}")
    if sha_2b != EXPECTED_2B_SHA256:
        raise AssertionError(f"2B donor shard SHA-256 mismatch: {sha_2b}")

    persistent_audit: dict[str, Any] = {
        "C0_raw": {"path": str(RECIPIENT_DIR), "shard_sha256": sha_08b, "total_params": 873_438_784},
        "Qwen3.5-2B": {"path": str(DONOR_DIR), "shard_sha256": sha_2b, "total_params": 2_274_069_824},
    }
    for arm in ("C1", "C4_weights", "C4_act", "T1", "T2"):
        ckpt = PERSISTENT_DIRS[arm]
        inv = verify_checkpoint_invariants(RECIPIENT_DIR, ckpt)
        _assert_recipient_invariants(arm, inv)
        persistent_audit[arm] = {
            "path": str(ckpt),
            "shard_sha256": _primary_shard_sha256(ckpt),
            "total_params": inv["total_params"],
            "frozen_tensors_verified": inv["frozen_tensors_verified"],
            "changed_tensor_count": inv["changed_tensor_count"],
            "changed_params": inv["changed_params"],
            "overall_relative_frobenius_norm": inv["overall_relative_frobenius_norm"],
        }

    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "milestone": "M3_checkpoint_audit",
        "persistent_arms": persistent_audit,
        "rebuilt_temp_arms": ["C3", "C5_T1_rowperm", "C5_T2_norm_matched"],
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    compact = {
        k: {
            "changed_params": v.get("changed_params", 0),
            "rel_frob": round(v.get("overall_relative_frobenius_norm", 0.0), 6),
            "frozen_vision": v.get("frozen_tensors_verified"),
        }
        for k, v in persistent_audit.items()
    }
    print(json.dumps(compact, indent=2), flush=True)
    return audit


def _load_isolated_datasets(isolated_dir: Path) -> dict[str, tuple[list[dict[str, Any]], str]]:
    datasets: dict[str, tuple[list[dict[str, Any]], str]] = {}
    for name, (fname, expected_sha) in DATASET_SHA256.items():
        path = isolated_dir / fname
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise AssertionError(f"Isolated dataset {path} SHA-256 mismatch: {actual_sha} != {expected_sha}")
        if name == "gsm8k_test":
            all_rows = load_dataset_rows(path)
            if len(all_rows) != 1319:
                raise AssertionError(f"Expected 1319 gsm8k_test rows, got {len(all_rows)}")
            idx = sorted(random.Random(GSM8K_SAMPLE_SEED).sample(range(len(all_rows)), GSM8K_SAMPLE_N))
            rows = [all_rows[i] for i in idx]
        elif name == "synthetic_test":
            rows = load_dataset_rows(path, per_subskill_limit=SYNTHETIC_PER_SUBSKILL)
            if len(rows) != 150:
                raise AssertionError(f"Expected 150 balanced synthetic_test rows, got {len(rows)}")
        else:
            rows = load_dataset_rows(path)
            if len(rows) != 100:
                raise AssertionError(f"Expected 100 synthetic_ood_test rows, got {len(rows)}")
        datasets[name] = (rows, actual_sha)
    return datasets


def _eval_dataset_rows(
    model: Any,
    tokenizer: Any,
    device: str,
    rows: list[dict[str, Any]],
    log_path: Path,
    max_new_tokens: int = 1024,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    total_tokens = 0
    t_eval_start = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as lf, torch.inference_mode():
        for item in rows:
            prompt_text = format_prompt(tokenizer, item["prompt"], enable_thinking=False)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            input_len = int(inputs["input_ids"].shape[1])

            t0 = time.perf_counter()
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            elapsed = time.perf_counter() - t0

            gen_ids = out_ids[0][input_len:]
            gen_len = int(gen_ids.shape[0])
            total_tokens += gen_len
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
    return {
        "metrics": compute_eval_metrics(records),
        "total_generated_tokens": total_tokens,
        "tokens_per_sec": round(total_tokens / max(eval_sec, 1e-6), 2),
        "eval_sec": round(eval_sec, 3),
    }


def _arm_is_complete(arm: str, logs_dir: Path) -> dict[str, Any] | None:
    summary_p = Path(f"runs/manifests/m3_{arm}_summary.json")
    if not summary_p.exists():
        return None
    expected = {"gsm8k_test": 200, "synthetic_test": 150, "synthetic_ood_test": 100}
    try:
        for ds, exp_n in expected.items():
            lp = logs_dir / f"m3_{arm}_{ds}.jsonl"
            if not lp.exists():
                return None
            with lp.open("r", encoding="utf-8") as f:
                if sum(1 for ln in f if ln.strip()) != exp_n:
                    return None
        ret_lp = Path(f"runs/logs/m3_{arm}_retention.jsonl")
        if not ret_lp.exists():
            return None
        with ret_lp.open("r", encoding="utf-8") as f:
            if sum(1 for ln in f if ln.strip()) != 30:
                return None
        return json.loads(summary_p.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_eval(isolated_dir: Path, device: str, dtype_name: str, selected_arms: list[str] | None) -> None:
    """Evaluate all frozen arms once loaded onto M4 Max MPS, keeping raw test logs strictly external."""
    datasets = _load_isolated_datasets(isolated_dir)
    ext_logs_dir = isolated_dir / "m3_logs"
    ext_logs_dir.mkdir(parents=True, exist_ok=True)
    Path("runs/manifests").mkdir(parents=True, exist_ok=True)
    Path("runs/logs").mkdir(parents=True, exist_ok=True)

    arms_to_run = [a for a, _ in FROZEN_ARMS if selected_arms is None or a in selected_arms]
    for arm in arms_to_run:
        existing = _arm_is_complete(arm, ext_logs_dir)
        if existing is not None:
            print(f"[{arm}] reusing completed M3 summary", flush=True)
            continue

        arm_mode = dict(FROZEN_ARMS)[arm]
        temp_dir: Path | None = None
        patch_meta: dict[str, Any] | None = None
        if arm_mode == "temp":
            ckpt_dir, patch_rec = _build_temp_arm(arm)
            temp_dir = ckpt_dir
            patch_meta = patch_rec["invariants"]
        else:
            ckpt_dir = PERSISTENT_DIRS[arm]
            if arm != "Qwen3.5-2B" and arm != "C0_raw":
                patch_meta = verify_checkpoint_invariants(RECIPIENT_DIR, ckpt_dir)
                _assert_recipient_invariants(arm, patch_meta)

        model, tokenizer, dev, load_sec = load_model_and_tokenizer(ckpt_dir, device=device, dtype_name=dtype_name)
        if dev == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        t_arm = time.perf_counter()
        ds_results: dict[str, Any] = {}
        try:
            for ds_name in ("synthetic_test", "synthetic_ood_test", "gsm8k_test"):
                rows, ds_sha = datasets[ds_name]
                res = _eval_dataset_rows(
                    model=model,
                    tokenizer=tokenizer,
                    device=dev,
                    rows=rows,
                    log_path=ext_logs_dir / f"m3_{arm}_{ds_name}.jsonl",
                    max_new_tokens=1024,
                )
                ds_results[ds_name] = {**res, "dataset_sha256": ds_sha}

            ret_full = evaluate_retention(
                model=model,
                tokenizer=tokenizer,
                device=dev,
                max_new_tokens=512,
                log_path=f"runs/logs/m3_{arm}_retention.jsonl",
            )
            ret_summary = {k: v for k, v in ret_full.items() if k != "records"}
            ppl_summary = evaluate_perplexity(model=model, tokenizer=tokenizer, device=dev)

            peak_mps_mb = None
            if dev == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
                peak_mps_mb = round(torch.mps.driver_allocated_memory() / (1024 * 1024), 2)
            peak_rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 2)

            summary = {
                "arm": arm,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint_path": str(ckpt_dir),
                "shard_sha256": _primary_shard_sha256(ckpt_dir),
                "patch_invariants": patch_meta,
                "device": dev,
                "dtype": dtype_name,
                "load_sec": round(load_sec, 3),
                "arm_wall_sec": round(time.perf_counter() - t_arm, 3),
                "peak_mps_allocated_mb": peak_mps_mb,
                "peak_rss_mb": peak_rss_mb,
                "datasets": ds_results,
                "retention_ifeval": ret_summary,
                "prose_perplexity": ppl_summary,
            }
            Path(f"runs/manifests/m3_{arm}_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )

            syn_sub = ds_results["synthetic_test"]["metrics"]["by_subskill"]
            print(
                f"[{arm}] gsm8k={ds_results['gsm8k_test']['metrics']['pass_at_1']:.3f} "
                f"carry={syn_sub['carry_add_sub']['pass_at_1']:.3f} "
                f"mult={syn_sub['mult_2x2_3x1']['pass_at_1']:.3f} "
                f"word={syn_sub['word_2step']['pass_at_1']:.3f} "
                f"ood={ds_results['synthetic_ood_test']['metrics']['pass_at_1']:.3f} "
                f"ret={ret_summary['pass_rate']:.3f} "
                f"ppl={ppl_summary['perplexity']:.2f} "
                f"tok/s={ds_results['gsm8k_test']['tokens_per_sec']} "
                f"mps_mb={peak_mps_mb}",
                flush=True,
            )
        finally:
            del model
            gc.collect()
            if dev == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            if temp_dir is not None and temp_dir.exists():
                shutil.rmtree(temp_dir)


def _load_outcomes_bool(path: Path) -> dict[str, tuple[bool, str | None]]:
    out: dict[str, tuple[bool, str | None]] = {}
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            out[str(r["id"])] = (bool(r["correct"]), r.get("subskill"))
    return out


def _paired_comparison(
    outcomes_a: dict[str, tuple[bool, str | None]],
    outcomes_b: dict[str, tuple[bool, str | None]],
    subskill: str | None = None,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    common = sorted(set(outcomes_a) & set(outcomes_b))
    if subskill is not None:
        common = [i for i in common if outcomes_a[i][1] == subskill]
    n = len(common)
    if n == 0:
        raise ValueError(f"Empty intersection for subskill={subskill}")
    a = np.array([1 if outcomes_a[i][0] else 0 for i in common], dtype=np.int64)
    b = np.array([1 if outcomes_b[i][0] else 0 for i in common], dtype=np.int64)
    improved = int(np.sum((a == 0) & (b == 1)))
    regressed = int(np.sum((a == 1) & (b == 0)))
    both_correct = int(np.sum((a == 1) & (b == 1)))
    both_incorrect = int(np.sum((a == 0) & (b == 0)))
    disc = improved + regressed
    p_val = float(binomtest(min(improved, regressed), disc, 0.5).pvalue) if disc > 0 else 1.0
    diff = (b - a).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1) * 100.0
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": n,
        "acc_a": round(float(a.mean()), 4),
        "acc_b": round(float(b.mean()), 4),
        "delta_pp": round(float(diff.mean() * 100.0), 2),
        "bootstrap_95ci_pp": [round(float(ci_lo), 2), round(float(ci_hi), 2)],
        "improved": improved,
        "regressed": regressed,
        "both_correct": both_correct,
        "both_incorrect": both_incorrect,
        "mcnemar_exact_p": p_val,
    }


def run_analyze(isolated_dir: Path, output_path: Path) -> dict[str, Any]:
    """Aggregate M3 results and compute Holm-Bonferroni corrected paired statistics (zero hidden strings)."""
    ext_logs_dir = isolated_dir / "m3_logs"
    summaries: dict[str, Any] = {}
    outcomes: dict[str, dict[str, dict[str, tuple[bool, str | None]]]] = {}

    for arm, _ in FROZEN_ARMS:
        summary_p = Path(f"runs/manifests/m3_{arm}_summary.json")
        if not summary_p.exists():
            raise FileNotFoundError(f"Missing M3 summary for {arm}: {summary_p}")
        summaries[arm] = json.loads(summary_p.read_text(encoding="utf-8"))
        outcomes[arm] = {
            "gsm8k_test": _load_outcomes_bool(ext_logs_dir / f"m3_{arm}_gsm8k_test.jsonl"),
            "synthetic_test": _load_outcomes_bool(ext_logs_dir / f"m3_{arm}_synthetic_test.jsonl"),
            "synthetic_ood_test": _load_outcomes_bool(ext_logs_dir / f"m3_{arm}_synthetic_ood_test.jsonl"),
            "retention_ifeval": _load_outcomes_bool(Path(f"runs/logs/m3_{arm}_retention.jsonl")),
        }

    pair_list = (
        [("C0_raw", arm) for arm, _ in FROZEN_ARMS if arm != "C0_raw"]
        + [
            ("C1", "T1"),
            ("C4_weights", "T1"),
            ("C5_T1_rowperm", "T1"),
            ("C1", "T2"),
            ("C4_act", "T2"),
            ("C5_T2_norm_matched", "T2"),
            ("C3", "C4_weights"),
            ("C3", "C4_act"),
            ("C3", "T1"),
            ("C3", "T2"),
        ]
    )

    paired: dict[str, dict[str, Any]] = {}
    for arm_a, arm_b in pair_list:
        key = f"{arm_a}_to_{arm_b}"
        paired[key] = {
            "gsm8k_test": _paired_comparison(outcomes[arm_a]["gsm8k_test"], outcomes[arm_b]["gsm8k_test"]),
            "synthetic_test": _paired_comparison(outcomes[arm_a]["synthetic_test"], outcomes[arm_b]["synthetic_test"]),
            "carry_add_sub": _paired_comparison(
                outcomes[arm_a]["synthetic_test"], outcomes[arm_b]["synthetic_test"], subskill="carry_add_sub"
            ),
            "mult_2x2_3x1": _paired_comparison(
                outcomes[arm_a]["synthetic_test"], outcomes[arm_b]["synthetic_test"], subskill="mult_2x2_3x1"
            ),
            "word_2step": _paired_comparison(
                outcomes[arm_a]["synthetic_test"], outcomes[arm_b]["synthetic_test"], subskill="word_2step"
            ),
            "synthetic_ood_test": _paired_comparison(
                outcomes[arm_a]["synthetic_ood_test"], outcomes[arm_b]["synthetic_ood_test"]
            ),
            "retention_ifeval": _paired_comparison(
                outcomes[arm_a]["retention_ifeval"], outcomes[arm_b]["retention_ifeval"]
            ),
        }

    # Step-down Holm-Bonferroni correction across primary contrasts per track
    holm_t1 = {
        ds: holm_bonferroni(
            {
                "T1_vs_C1": paired["C1_to_T1"][ds]["mcnemar_exact_p"],
                "T1_vs_C4_weights": paired["C4_weights_to_T1"][ds]["mcnemar_exact_p"],
            }
        )
        for ds in ("gsm8k_test", "carry_add_sub", "synthetic_ood_test")
    }
    holm_t2 = {
        ds: holm_bonferroni(
            {
                "T2_vs_C1": paired["C1_to_T2"][ds]["mcnemar_exact_p"],
                "T2_vs_C4_act": paired["C4_act_to_T2"][ds]["mcnemar_exact_p"],
            }
        )
        for ds in ("gsm8k_test", "carry_add_sub", "synthetic_ood_test")
    }

    base_tok_s = float(summaries["C0_raw"]["datasets"]["gsm8k_test"]["tokens_per_sec"])
    base_mps_mb = float(summaries["C0_raw"]["peak_mps_allocated_mb"] or 0.0)

    def _criteria_for(track_arm: str, c4_arm: str, c5_arm: str, holm_dict: dict[str, Any]) -> dict[str, Any]:
        gsm_gain_pp = paired[f"C0_raw_to_{track_arm}"]["gsm8k_test"]["delta_pp"]
        carry_gain_pp = paired[f"C0_raw_to_{track_arm}"]["carry_add_sub"]["delta_pp"]
        adv_c1_pp = paired[f"C1_to_{track_arm}"]["gsm8k_test"]["delta_pp"]
        adv_c4_pp = paired[f"{c4_arm}_to_{track_arm}"]["gsm8k_test"]["delta_pp"]
        adv_c5_pp = paired[f"{c5_arm}_to_{track_arm}"]["gsm8k_test"]["delta_pp"]
        ood_gain_pp = paired[f"C0_raw_to_{track_arm}"]["synthetic_ood_test"]["delta_pp"]
        ret_delta_pp = paired[f"C0_raw_to_{track_arm}"]["retention_ifeval"]["delta_pp"]
        arm_tok_s = float(summaries[track_arm]["datasets"]["gsm8k_test"]["tokens_per_sec"])
        arm_mps_mb = float(summaries[track_arm]["peak_mps_allocated_mb"] or 0.0)
        speed_ratio = arm_tok_s / max(base_tok_s, 1e-6)
        mem_ratio = arm_mps_mb / max(base_mps_mb, 1e-6)
        inv = summaries[track_arm]["patch_invariants"]
        return {
            "math_gain_ge_5pp_gsm8k_and_pos_carry": gsm_gain_pp >= 5.0 and carry_gain_pp > 0.0,
            "agent_advantage_over_C1_and_C4_holm_sig": (
                adv_c1_pp > 0.0
                and adv_c4_pp > 0.0
                and max(holm_dict["gsm8k_test"].values()) < 0.05
            ),
            "donor_dependence_pos_over_C5": adv_c5_pp > 0.0 and paired[f"{c5_arm}_to_{track_arm}"]["gsm8k_test"]["mcnemar_exact_p"] < 0.05,
            "ood_generalization_pos": ood_gain_pp > 0.0,
            "retention_drop_le_3pp": ret_delta_pp >= -3.0,
            "structural_invariants_exact": (inv["total_params"] == 873_438_784 and inv["frozen_tensors_verified"] == 153),
            "device_efficiency_within_10pct": (0.90 <= speed_ratio <= 1.10 and mem_ratio <= 1.10),
            "observed_values": {
                "gsm8k_vs_c0_pp": gsm_gain_pp,
                "carry_vs_c0_pp": carry_gain_pp,
                "gsm8k_vs_c1_pp": adv_c1_pp,
                "gsm8k_vs_c4_pp": adv_c4_pp,
                "gsm8k_vs_c5_pp": adv_c5_pp,
                "ood_vs_c0_pp": ood_gain_pp,
                "retention_vs_c0_pp": ret_delta_pp,
                "speed_ratio_vs_c0": round(speed_ratio, 4),
                "mps_mem_ratio_vs_c0": round(mem_ratio, 4),
            },
        }

    criteria_check = {
        "T1": _criteria_for("T1", "C4_weights", "C5_T1_rowperm", holm_t1),
        "T2": _criteria_for("T2", "C4_act", "C5_T2_norm_matched", holm_t2),
    }

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "milestone": "M3_isolated_confirmatory_evaluation",
        "single_session_limitation_note": "N=1 agent session per AWST track (T1 H002, T2 H003) per D010.",
        "isolated_sampling": {
            "gsm8k_test_seed": GSM8K_SAMPLE_SEED,
            "gsm8k_test_n": GSM8K_SAMPLE_N,
            "synthetic_test_n": 150,
            "synthetic_ood_test_n": 100,
            "retention_n": 30,
            "perplexity_passages_n": 10,
        },
        "arms": summaries,
        "paired_comparisons": paired,
        "holm_bonferroni": {"T1": holm_t1, "T2": holm_t2},
        "preregistered_criteria_check": criteria_check,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    table = {
        arm: {
            "gsm8k_test": summaries[arm]["datasets"]["gsm8k_test"]["metrics"]["pass_at_1"],
            "carry_test": summaries[arm]["datasets"]["synthetic_test"]["metrics"]["by_subskill"]["carry_add_sub"]["pass_at_1"],
            "mult_test": summaries[arm]["datasets"]["synthetic_test"]["metrics"]["by_subskill"]["mult_2x2_3x1"]["pass_at_1"],
            "word_test": summaries[arm]["datasets"]["synthetic_test"]["metrics"]["by_subskill"]["word_2step"]["pass_at_1"],
            "ood_test": summaries[arm]["datasets"]["synthetic_ood_test"]["metrics"]["pass_at_1"],
            "ret_ifeval": round(summaries[arm]["retention_ifeval"]["pass_rate"], 4),
            "prose_ppl": summaries[arm]["prose_perplexity"]["perplexity"],
            "gsm8k_tok_s": summaries[arm]["datasets"]["gsm8k_test"]["tokens_per_sec"],
            "peak_mps_mb": summaries[arm]["peak_mps_allocated_mb"],
        }
        for arm, _ in FROZEN_ARMS
    }
    print(json.dumps({"table": table, "holm": report["holm_bonferroni"], "criteria": criteria_check}, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "eval", "analyze", "all"), default="all")
    parser.add_argument("--arms", default=None, help="Comma-separated subset of FROZEN_ARMS")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--isolated-dir", default=str(DEFAULT_ISOLATED_DIR))
    parser.add_argument("--output", default="results/m3_isolated_evaluation.json")
    args = parser.parse_args()

    selected = [x.strip() for x in args.arms.split(",") if x.strip()] if args.arms else None
    iso_dir = Path(args.isolated_dir)

    if args.stage in ("audit", "all"):
        run_audit()
    if args.stage in ("eval", "all"):
        run_eval(iso_dir, device=args.device, dtype_name=args.dtype, selected_arms=selected)
    if args.stage in ("analyze", "all"):
        run_analyze(iso_dir, output_path=Path(args.output))


if __name__ == "__main__":
    main()
