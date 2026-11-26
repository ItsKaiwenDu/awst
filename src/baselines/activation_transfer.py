"""C4_act: Activation-assisted APM + ridge residual basis alignment (Track T2 conventional baseline)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors import safe_open
from scipy.optimize import linear_sum_assignment

from src.evaluation.runner import format_prompt, load_model_and_tokenizer
from src.interventions.patch_applier import apply_tensor_patches
from src.weight_io.safetensors_reader import _resolve_tensor_file


def load_dev_calibration_prompts(
    synth_dev_path: str | Path = "data/development/synthetic_dev.jsonl",
    gsm8k_dev_path: str | Path = "data/development/gsm8k_dev.jsonl",
    n_synth: int = 16,
    n_gsm8k: int = 16,
) -> list[dict[str, str]]:
    """Load calibration prompts strictly from data/development/ (never val or isolated test)."""
    for p in (synth_dev_path, gsm8k_dev_path):
        resolved = str(Path(p).resolve())
        if "development" not in resolved:
            raise ValueError(f"Calibration must use data/development/ only, got {resolved}")

    prompts: list[dict[str, str]] = []
    with Path(synth_dev_path).open("r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("subskill") == "carry_add_sub":
                prompts.append({"id": r["id"], "prompt": r["prompt"]})
                if len(prompts) >= n_synth:
                    break

    gsm_added = 0
    with Path(gsm8k_dev_path).open("r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            prompts.append({"id": r["id"], "prompt": r["prompt"]})
            gsm_added += 1
            if gsm_added >= n_gsm8k:
                break
    return prompts


def collect_mlp_activations_stats(
    donor_dir: str | Path,
    recipient_dir: str | Path,
    prompts: list[dict[str, str]],
    target_layers: Iterable[int],
    device: str = "mps",
    dtype_name: str = "bfloat16",
    ridge_lambda: float = 1e-2,
) -> dict[int, dict[str, torch.Tensor]]:
    """Collect token-aligned MLP input/intermediate/output cross-covariances without keeping all tokens in RAM."""
    layers = sorted(set(target_layers))

    def _run_model(ckpt_dir: str | Path) -> tuple[dict[int, list[torch.Tensor]], dict[int, list[torch.Tensor]], dict[int, list[torch.Tensor]]]:
        model, tokenizer, dev, _ = load_model_and_tokenizer(ckpt_dir, device=device, dtype_name=dtype_name)
        xs: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
        hs: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
        ys: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
        handles = []

        lm_layers = model.model.language_model.layers
        for l in layers:
            mlp = lm_layers[l].mlp

            def _make_mlp_hook(layer_id: int):
                def _hook(_mod, inp, out):
                    x_t = inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.float32).cpu()
                    y_t = out.detach().reshape(-1, out.shape[-1]).to(torch.float32).cpu()
                    xs[layer_id].append(x_t)
                    ys[layer_id].append(y_t)
                return _hook

            def _make_down_prehook(layer_id: int):
                def _pre(_mod, inp):
                    h_t = inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.float32).cpu()
                    hs[layer_id].append(h_t)
                return _pre

            handles.append(mlp.register_forward_hook(_make_mlp_hook(l)))
            handles.append(mlp.down_proj.register_forward_pre_hook(_make_down_prehook(l)))

        with torch.inference_mode():
            for item in prompts:
                txt = format_prompt(tokenizer, item["prompt"], enable_thinking=False)
                enc = tokenizer(txt, return_tensors="pt").to(dev)
                _ = model(**enc, use_cache=False)

        for h in handles:
            h.remove()
        del model
        if dev == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        return xs, hs, ys

    xs_d, hs_d, ys_d = _run_model(donor_dir)
    xs_r, hs_r, ys_r = _run_model(recipient_dir)

    stats: dict[int, dict[str, torch.Tensor]] = {}
    for l in layers:
        xd = torch.cat(xs_d[l], dim=0).to(torch.float64)
        xr = torch.cat(xs_r[l], dim=0).to(torch.float64)
        yd = torch.cat(ys_d[l], dim=0).to(torch.float64)
        yr = torch.cat(ys_r[l], dim=0).to(torch.float64)
        hd = torch.cat(hs_d[l], dim=0)  # float32 [N_tok, I_d]
        hr = torch.cat(hs_r[l], dim=0)  # float32 [N_tok, I_r]

        # Ridge regression maps: X_d @ P_in ≈ X_r  and  Y_d @ P_out ≈ Y_r
        d_hidden = xd.shape[1]
        reg = ridge_lambda * torch.eye(d_hidden, dtype=torch.float64)
        p_in = torch.linalg.solve(xd.T @ xd + reg, xd.T @ xr).to(torch.float32)
        p_out = torch.linalg.solve(yd.T @ yd + reg, yd.T @ yr).to(torch.float32)

        # APM intermediate neuron mean magnitude & normalized activation cross-correlation
        hd_mean_abs = hd.abs().mean(dim=0)
        hd_c = hd - hd.mean(dim=0, keepdim=True)
        hr_c = hr - hr.mean(dim=0, keepdim=True)
        hd_n = hd_c / torch.linalg.vector_norm(hd_c, dim=0, keepdim=True).clamp_min(1e-8)
        hr_n = hr_c / torch.linalg.vector_norm(hr_c, dim=0, keepdim=True).clamp_min(1e-8)
        act_corr = (hr_n.T @ hd_n)  # [I_r, I_d]

        stats[l] = {
            "p_in": p_in,
            "p_out": p_out,
            "hd_mean_abs": hd_mean_abs,
            "act_corr": act_corr,
        }
    return stats


def _load_tensor_f32(ckpt_dir: Path, name: str) -> torch.Tensor:
    shard = _resolve_tensor_file(ckpt_dir, name)
    with safe_open(shard, framework="pt", device="cpu") as f:
        return f.get_tensor(name).to(torch.float32)


def build_c4_activation_guided(
    donor_dir: str | Path,
    recipient_dir: str | Path,
    out_dir: str | Path,
    target_layers: Iterable[int],
    act_stats: dict[int, dict[str, torch.Tensor]],
    alpha: float = 0.05,
    n_calibration_prompts: int = 32,
    manifest_path: str | Path | None = None,
    overwrite: bool = True,
    donor_override: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """C4_act: Activation-Prune-Merge + Ridge Activation Alignment (Track T2 baseline)."""
    d_path = Path(donor_dir).resolve()
    r_path = Path(recipient_dir).resolve()
    layers = sorted(set(target_layers))
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

        st = act_stats[l]
        p_in = st["p_in"]    # [2048, 1024]
        p_out = st["p_out"]  # [2048, 1024]
        hd_mean = st["hd_mean_abs"]
        act_corr = st["act_corr"]  # [3584, 6144]

        inter_r = g_r.shape[0]
        inter_d = g_d.shape[0]

        # 1. APM salience pruning: mean(|h_j|) * ||W_down[:, j]||_2
        if inter_d > inter_r:
            sal = hd_mean * torch.linalg.vector_norm(d_d, dim=0)
            top_idx = torch.topk(sal, k=inter_r, largest=True).indices
            g_sel = g_d.index_select(0, top_idx)
            u_sel = u_d.index_select(0, top_idx)
            d_sel = d_d.index_select(1, top_idx)
            corr_sel = act_corr.index_select(1, top_idx)
        else:
            g_sel, u_sel, d_sel, corr_sel = g_d, u_d, d_d, act_corr

        # 2. Activation correlation permutation matching (Transport & Merge / APM)
        _, col_ind = linear_sum_assignment(-corr_sel.numpy())
        perm = torch.from_numpy(col_ind.astype(np.int64))
        g_perm = g_sel.index_select(0, perm)
        u_perm = u_sel.index_select(0, perm)
        d_perm = d_sel.index_select(1, perm)

        # 3. Map donor input/output residual spaces into recipient basis:
        # g_r(x_r) ≈ g_d(x_d) where x_d @ p_in ≈ x_r -> use pseudoinverse / transpose of p_in
        # Specifically: if x_d ≈ x_r @ p_in^+ (with p_in^+ shape [H_r, H_d]), then g_d(x_d) = x_r @ (p_in^+ @ g_perm^T)
        p_in_pinv = torch.linalg.pinv(p_in, rcond=1e-3)  # [1024, 2048]
        g_al = g_perm @ p_in_pinv.T
        u_al = u_perm @ p_in_pinv.T
        d_al = p_out.T @ d_perm  # y_r ≈ p_out^T @ y_d

        def _match_norm(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
            rn = float(torch.linalg.vector_norm(ref).item())
            sn = max(float(torch.linalg.vector_norm(src).item()), 1e-12)
            return src * (rn / sn)

        replacements[g_name] = (1.0 - alpha) * g_r + alpha * _match_norm(g_al, g_r)
        replacements[u_name] = (1.0 - alpha) * u_r + alpha * _match_norm(u_al, u_r)
        replacements[d_name] = (1.0 - alpha) * d_r + alpha * _match_norm(d_al, d_r)

    def patch_fn(name: str, tensor: torch.Tensor) -> torch.Tensor | None:
        if name not in replacements:
            return None
        return replacements[name].to(tensor.dtype)

    return apply_tensor_patches(
        base_dir=r_path,
        out_dir=out_dir,
        patch_fn=patch_fn,
        provenance={
            "condition": "C4_act",
            "track": "T2_activation_assisted",
            "description": "Activation-Prune-Merge (APM) + Ridge activation alignment on data/development/ calibration prompts",
            "donor_dir": str(d_path),
            "target_layers": layers,
            "alpha": alpha,
            "donor_forward_passes": n_calibration_prompts,
            "used_shuffled_donor": donor_override is not None,
        },
        manifest_path=manifest_path,
        overwrite=overwrite,
    )
