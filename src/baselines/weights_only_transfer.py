"""C4_weights: Weights-only conventional donor transfer (Procrustes embedding basis + neuron matching)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors import safe_open
from scipy.optimize import linear_sum_assignment

from src.interventions.patch_applier import apply_tensor_patches
from src.weight_io.safetensors_reader import _resolve_tensor_file


def _load_tensor_f32(ckpt_dir: Path, name: str) -> torch.Tensor:
    shard = _resolve_tensor_file(ckpt_dir, name)
    with safe_open(shard, framework="pt", device="cpu") as f:
        return f.get_tensor(name).to(torch.float32)


def compute_weights_only_residual_projection(
    donor_dir: Path,
    recipient_dir: Path,
    sample_tokens: int = 16384,
    embed_name: str = "model.language_model.embed_tokens.weight",
) -> torch.Tensor:
    """Compute orthonormal Procrustes map P in R^{d_donor x d_rec} from shared token embeddings.

    Zero forward passes are used; relies solely on shared vocabulary token alignment.
    """
    e_don = _load_tensor_f32(donor_dir, embed_name)
    e_rec = _load_tensor_f32(recipient_dir, embed_name)
    if e_don.shape[0] != e_rec.shape[0]:
        raise ValueError(f"Vocab mismatch: donor {e_don.shape[0]} vs recipient {e_rec.shape[0]}")

    vocab_n = e_don.shape[0]
    n = min(sample_tokens, vocab_n)
    idx = torch.linspace(0, vocab_n - 1, steps=n, dtype=torch.long)
    ed_sub = e_don.index_select(0, idx).to(torch.float64)
    er_sub = e_rec.index_select(0, idx).to(torch.float64)

    # Solve argmin_{P^T P = I} || E_d P - E_r ||_F via SVD of E_d^T E_r
    cross = ed_sub.T @ er_sub
    u, _, vt = torch.linalg.svd(cross, full_matrices=False)
    proj = (u @ vt).to(torch.float32)
    return proj


def align_swiglu_mlp_weights_only(
    gate_d: torch.Tensor,
    up_d: torch.Tensor,
    down_d: torch.Tensor,
    gate_r: torch.Tensor,
    up_r: torch.Tensor,
    down_r: torch.Tensor,
    proj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project donor SwiGLU MLP into recipient residual basis, prune & permute-align neurons."""
    # 1. Project residual input/output dimension: d_donor -> d_rec
    g_proj = gate_d @ proj  # [I_d, H_r]
    u_proj = up_d @ proj    # [I_d, H_r]
    d_proj = proj.T @ down_d  # [H_r, I_d]

    inter_r = gate_r.shape[0]
    inter_d = gate_d.shape[0]

    # 2. Weights-only neuron salience pruning if I_d > I_r
    if inter_d > inter_r:
        imp = (
            torch.linalg.vector_norm(g_proj, dim=1)
            * torch.linalg.vector_norm(u_proj, dim=1)
            * torch.linalg.vector_norm(d_proj, dim=0)
        )
        top_idx = torch.topk(imp, k=inter_r, largest=True).indices
        g_sel = g_proj.index_select(0, top_idx)
        u_sel = u_proj.index_select(0, top_idx)
        d_sel = d_proj.index_select(1, top_idx)
    else:
        g_sel, u_sel, d_sel = g_proj, u_proj, d_proj

    # 3. Permutation matching (weights-only Git Re-Basin) via cosine similarity of neuron signatures
    sig_d = torch.cat([g_sel, u_sel, d_sel.T], dim=1)  # [I_r, 3*H_r]
    sig_r = torch.cat([gate_r, up_r, down_r.T], dim=1)  # [I_r, 3*H_r]
    sig_d_n = sig_d / torch.linalg.vector_norm(sig_d, dim=1, keepdim=True).clamp_min(1e-12)
    sig_r_n = sig_r / torch.linalg.vector_norm(sig_r, dim=1, keepdim=True).clamp_min(1e-12)
    sim = (sig_r_n @ sig_d_n.T).numpy()

    _, col_ind = linear_sum_assignment(-sim)
    perm = torch.from_numpy(col_ind.astype(np.int64))

    g_al = g_sel.index_select(0, perm)
    u_al = u_sel.index_select(0, perm)
    d_al = d_sel.index_select(1, perm)

    # 4. Norm-match each aligned donor tensor to the recipient tensor norm
    def _match_norm(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        rn = float(torch.linalg.vector_norm(ref).item())
        sn = max(float(torch.linalg.vector_norm(src).item()), 1e-12)
        return src * (rn / sn)

    return _match_norm(g_al, gate_r), _match_norm(u_al, up_r), _match_norm(d_al, down_r)


def build_c4_weights_only(
    donor_dir: str | Path,
    recipient_dir: str | Path,
    out_dir: str | Path,
    target_layers: Iterable[int],
    alpha: float = 0.05,
    manifest_path: str | Path | None = None,
    overwrite: bool = True,
    donor_override: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """C4_weights: Weights-only Procrustes + neuron-aligned SwiGLU MLP transfer (Track T1 baseline)."""
    d_path = Path(donor_dir).resolve()
    r_path = Path(recipient_dir).resolve()
    layers = sorted(set(target_layers))

    if donor_override and "model.language_model.embed_tokens.weight" in donor_override:
        e_don = donor_override["model.language_model.embed_tokens.weight"].to(torch.float32)
        e_rec = _load_tensor_f32(r_path, "model.language_model.embed_tokens.weight")
        n = min(16384, e_don.shape[0])
        idx = torch.linspace(0, e_don.shape[0] - 1, steps=n, dtype=torch.long)
        cross = e_don.index_select(0, idx).to(torch.float64).T @ e_rec.index_select(0, idx).to(torch.float64)
        u, _, vt = torch.linalg.svd(cross, full_matrices=False)
        proj = (u @ vt).to(torch.float32)
    else:
        proj = compute_weights_only_residual_projection(d_path, r_path)

    # Precompute aligned MLP replacements per target layer
    replacements: dict[str, torch.Tensor] = {}
    for l in layers:
        g_name = f"model.language_model.layers.{l}.mlp.gate_proj.weight"
        u_name = f"model.language_model.layers.{l}.mlp.up_proj.weight"
        d_name = f"model.language_model.layers.{l}.mlp.down_proj.weight"

        def _get_d(nm: str) -> torch.Tensor:
            if donor_override and nm in donor_override:
                return donor_override[nm].to(torch.float32)
            return _load_tensor_f32(d_path, nm)

        g_d, u_d, d_d = _get_d(g_name), _get_d(u_name), _get_d(d_name)
        g_r = _load_tensor_f32(r_path, g_name)
        u_r = _load_tensor_f32(r_path, u_name)
        d_r = _load_tensor_f32(r_path, d_name)

        g_al, u_al, d_al = align_swiglu_mlp_weights_only(g_d, u_d, d_d, g_r, u_r, d_r, proj)
        replacements[g_name] = (1.0 - alpha) * g_r + alpha * g_al
        replacements[u_name] = (1.0 - alpha) * u_r + alpha * u_al
        replacements[d_name] = (1.0 - alpha) * d_r + alpha * d_al

    def patch_fn(name: str, tensor: torch.Tensor) -> torch.Tensor | None:
        if name not in replacements:
            return None
        return replacements[name].to(tensor.dtype)

    return apply_tensor_patches(
        base_dir=r_path,
        out_dir=out_dir,
        patch_fn=patch_fn,
        provenance={
            "condition": "C4_weights",
            "track": "T1_weights_only",
            "description": "Weights-only orthogonal Procrustes embedding basis + SwiGLU neuron importance & permutation alignment",
            "donor_dir": str(d_path),
            "target_layers": layers,
            "alpha": alpha,
            "donor_forward_passes": 0,
            "used_shuffled_donor": donor_override is not None,
        },
        manifest_path=manifest_path,
        overwrite=overwrite,
    )
