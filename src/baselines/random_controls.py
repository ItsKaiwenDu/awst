"""Matched non-informed baseline controls: C2 (random perturbation) and C3 (random donor slices)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch

from src.interventions.patch_applier import apply_tensor_patches
from src.weight_io.safetensors_reader import _resolve_tensor_file
from safetensors import safe_open


MLP_PROJECTIONS = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")


def _is_target_mlp(name: str, target_layers: set[int]) -> bool:
    for l in target_layers:
        prefix = f"model.language_model.layers.{l}.mlp."
        if name.startswith(prefix) and name.endswith(MLP_PROJECTIONS):
            return True
    return False


def build_c2_random_perturbation(
    recipient_dir: str | Path,
    out_dir: str | Path,
    target_layers: Iterable[int],
    target_rel_frob: float = 0.05,
    seed: int = 42,
    manifest_path: str | Path | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    """C2: Add Gaussian noise to target recipient MLP weights at matched relative Frobenius norm."""
    layers_set = set(target_layers)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def patch_fn(name: str, tensor: torch.Tensor) -> torch.Tensor | None:
        if not _is_target_mlp(name, layers_set):
            return None
        w = tensor.to(torch.float32)
        noise = torch.randn(w.shape, generator=gen, dtype=torch.float32)
        w_norm = torch.linalg.vector_norm(w)
        n_norm = torch.linalg.vector_norm(noise)
        scaled = noise * (target_rel_frob * w_norm / max(float(n_norm.item()), 1e-12))
        return (w + scaled).to(tensor.dtype)

    return apply_tensor_patches(
        base_dir=recipient_dir,
        out_dir=out_dir,
        patch_fn=patch_fn,
        provenance={
            "condition": "C2",
            "description": "Random Gaussian recipient perturbation with matched per-tensor relative Frobenius norm",
            "target_layers": sorted(layers_set),
            "target_rel_frob": target_rel_frob,
            "seed": seed,
        },
        manifest_path=manifest_path,
        overwrite=overwrite,
    )


def build_c3_random_donor_slice(
    donor_dir: str | Path,
    recipient_dir: str | Path,
    out_dir: str | Path,
    target_layers: Iterable[int],
    alpha: float = 0.05,
    seed: int = 42,
    manifest_path: str | Path | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    """C3: Copy random unaligned donor MLP sub-slices (norm-matched) and interpolate by alpha."""
    layers_set = set(target_layers)
    donor_path = Path(donor_dir).resolve()
    rng = torch.Generator(device="cpu").manual_seed(seed)

    def patch_fn(name: str, tensor: torch.Tensor) -> torch.Tensor | None:
        if not _is_target_mlp(name, layers_set):
            return None
        shard = _resolve_tensor_file(donor_path, name)
        with safe_open(shard, framework="pt", device="cpu") as f:
            d_w = f.get_tensor(name).to(torch.float32)
        r_w = tensor.to(torch.float32)

        # Sample random unaligned row and column index subsets matching recipient shape
        if d_w.ndim != 2 or r_w.ndim != 2:
            raise ValueError(f"Expected 2D weight for {name}")
        row_idx = torch.randperm(d_w.shape[0], generator=rng)[: r_w.shape[0]]
        col_idx = torch.randperm(d_w.shape[1], generator=rng)[: r_w.shape[1]]
        d_slice = d_w.index_select(0, row_idx).index_select(1, col_idx)

        # Match Frobenius norm to recipient tensor before interpolation
        r_norm = float(torch.linalg.vector_norm(r_w).item())
        d_norm = max(float(torch.linalg.vector_norm(d_slice).item()), 1e-12)
        d_normed = d_slice * (r_norm / d_norm)
        merged = (1.0 - alpha) * r_w + alpha * d_normed
        return merged.to(tensor.dtype)

    return apply_tensor_patches(
        base_dir=recipient_dir,
        out_dir=out_dir,
        patch_fn=patch_fn,
        provenance={
            "condition": "C3",
            "description": "Random unaligned donor MLP slice transfer (norm-matched, unaligned)",
            "donor_dir": str(donor_path),
            "target_layers": sorted(layers_set),
            "alpha": alpha,
            "seed": seed,
            "donor_forward_passes": 0,
        },
        manifest_path=manifest_path,
        overwrite=overwrite,
    )
