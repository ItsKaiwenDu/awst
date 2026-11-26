"""Unit tests for bounded agent weight toolbox, track isolation, and RQ6 competency suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from src.agent_interface.competency_suite import (
    generate_competency_suite,
    verify_competency_submission,
)
from src.agent_interface.tools import (
    AgentWeightToolbox,
    BudgetExceededError,
    TrackViolationError,
)


def _dummy_ckpt(dirpath: Path, hidden: int, inter: int) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "model.language_model.embed_tokens.weight": torch.randn((32, hidden), dtype=torch.bfloat16),
            "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn((inter, hidden), dtype=torch.bfloat16),
        },
        str(dirpath / "model.safetensors"),
    )
    return dirpath


def test_toolbox_track_and_budget_enforcement(tmp_path: Path) -> None:
    don = _dummy_ckpt(tmp_path / "donor", 16, 32)
    rec = _dummy_ckpt(tmp_path / "rec", 8, 16)

    # C1 blocks all donor access
    tb_c1 = AgentWeightToolbox(donor_dir=None, recipient_dir=rec, track="C1")
    with pytest.raises(TrackViolationError):
        tb_c1.list_tensors("donor")

    # T1 permits donor stats/SVD up to budget, forbids donor forward passes
    tb_t1 = AgentWeightToolbox(donor_dir=don, recipient_dir=rec, track="T1", max_donor_bytes=2000)
    svd = tb_t1.tensor_svd_summary("donor", "model.language_model.layers.0.mlp.gate_proj.weight", top_k=4)
    assert svd["top_k"] == 4
    assert tb_t1.donor_bytes_accessed == 32 * 16 * 2
    with pytest.raises(TrackViolationError):
        tb_t1.record_forward_pass("donor", "data/development/synthetic_dev.jsonl")
    with pytest.raises(BudgetExceededError):
        tb_t1.tensor_stats("donor", "model.language_model.embed_tokens.weight")

    # T2 permits donor forward pass ONLY on data/development/, never validation or isolated test
    tb_t2 = AgentWeightToolbox(donor_dir=don, recipient_dir=rec, track="T2")
    tb_t2.record_forward_pass("donor", "data/development/synthetic_dev.jsonl", count=4)
    assert tb_t2.donor_forward_passes == 4
    with pytest.raises(TrackViolationError):
        tb_t2.record_forward_pass("donor", "data/validation/synthetic_val.jsonl")
    with pytest.raises(TrackViolationError):
        tb_t2.record_forward_pass("donor", "/Users/kaiwendu/.awst_isolated_eval/gsm8k_test.jsonl")


def test_competency_suite_oracle_and_naive_failure(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    generate_competency_suite(out_dir=suite_dir, seed=2026)

    # Solve Task B cleanly via exact rank-8 SVD factorization of M = W2_don @ W1_don (4 x 8)
    w_b_don = load_file(str(suite_dir / "task_b_transplant_rotated/donor.safetensors"))
    m_true = w_b_don["fc2.weight"] @ w_b_don["fc1.weight"]  # [4, 8]
    w1_sol = torch.zeros((16, 8), dtype=torch.float32)
    w2_sol = torch.zeros((4, 16), dtype=torch.float32)
    w1_sol[:4, :] = m_true
    w2_sol[:, :4] = torch.eye(4)
    b_patched = tmp_path / "task_b_patched.safetensors"
    save_file(
        {"fc1.weight": w1_sol, "fc1.bias": torch.zeros((16,)), "fc2.weight": w2_sol, "fc2.bias": torch.zeros((4,))},
        str(b_patched),
    )

    # Solve Task C cleanly by downscaling outlier columns by -1/25
    w_c = load_file(str(suite_dir / "task_c_localize_corruption/corrupted.safetensors"))
    col_norms = torch.linalg.vector_norm(w_c["fc2.weight"], dim=0)
    med = torch.median(col_norms)
    outlier_cols = torch.where(col_norms > 5.0 * med)[0].tolist()
    assert set(outlier_cols) == {5, 11, 19, 27}
    w2_rep = w_c["fc2.weight"].clone()
    w2_rep[:, outlier_cols] = w2_rep[:, outlier_cols] / -25.0
    c_repaired = tmp_path / "task_c_repaired.safetensors"
    save_file({**w_c, "fc2.weight": w2_rep}, str(c_repaired))

    res = verify_competency_submission(
        suite_dir=suite_dir,
        task_a_active_neurons=[0, 3],
        task_b_patched_recipient_path=b_patched,
        task_c_tensor_name="fc2.weight",
        task_c_cols=outlier_cols,
        task_c_repaired_path=c_repaired,
        seed=2026,
    )
    assert res["overall_score"] == 3
    assert res["task_b_agreement"] == 1.0
    assert res["task_c_agreement"] == 1.0

    # Verify naive unaligned copy fails Task B (H3 motivation)
    naive_b = tmp_path / "task_b_naive.safetensors"
    save_file(
        {
            "fc1.weight": w_b_don["fc1.weight"][:16, :].contiguous(),
            "fc1.bias": torch.zeros((16,)),
            "fc2.weight": w_b_don["fc2.weight"][:, :16].contiguous(),
            "fc2.bias": torch.zeros((4,)),
        },
        str(naive_b),
    )
    res_naive = verify_competency_submission(
        suite_dir=suite_dir,
        task_a_active_neurons=[0, 3],
        task_b_patched_recipient_path=naive_b,
        task_c_tensor_name="fc2.weight",
        task_c_cols=outlier_cols,
        task_c_repaired_path=c_repaired,
        seed=2026,
    )
    assert res_naive["task_b_passed"] is False
