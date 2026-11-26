"""Bounded weight-space inspection tools with strict T1/T2/C1 track and budget enforcement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from src.weight_io.safetensors_reader import (
    _resolve_tensor_file,
    classify_tensor_group,
    inspect_safetensors_header,
    read_tensor_slice,
    tensor_summary_stats,
)


class TrackViolationError(RuntimeError):
    """Raised when an agent operation violates C1/T1/T2 track isolation rules."""


class BudgetExceededError(RuntimeError):
    """Raised when an agent operation exceeds donor-byte or dev-trial budgets."""


class AgentWeightToolbox:
    """Controlled inspection API enforcing track separation and bounded context outputs."""

    def __init__(
        self,
        donor_dir: str | Path | None,
        recipient_dir: str | Path,
        track: str = "T1",
        max_donor_bytes: int = 200_000_000,
        max_dev_trials: int = 12,
        log_path: str | Path | None = None,
    ) -> None:
        if track not in ("C1", "T1", "T2"):
            raise ValueError(f"Unsupported track '{track}'; expected C1, T1, or T2")
        self.track = track
        self.recipient_dir = Path(recipient_dir).resolve()
        self.donor_dir = Path(donor_dir).resolve() if donor_dir is not None else None
        if track in ("T1", "T2") and self.donor_dir is None:
            raise ValueError(f"Track {track} requires donor_dir")

        self.max_donor_bytes = int(max_donor_bytes)
        self.max_dev_trials = int(max_dev_trials)
        self.donor_bytes_accessed = 0
        self.donor_forward_passes = 0
        self.recipient_forward_passes = 0
        self.dev_trials_used = 0
        self.events: list[dict[str, Any]] = []
        self.log_path = Path(log_path) if log_path else None

    def _checkpoint_for(self, which: str) -> Path:
        if which == "recipient":
            return self.recipient_dir
        if which == "donor":
            if self.track == "C1":
                raise TrackViolationError("Track C1 (donor-free WSP) forbids all donor weight access")
            assert self.donor_dir is not None
            return self.donor_dir
        raise ValueError(f"which must be 'donor' or 'recipient', got {which!r}")

    def _charge_donor_bytes(self, nbytes: int, op: str, tensor_name: str) -> None:
        if self.donor_bytes_accessed + nbytes > self.max_donor_bytes:
            raise BudgetExceededError(
                f"Operation {op}({tensor_name}) needs {nbytes} bytes, exceeding max_donor_bytes={self.max_donor_bytes}"
            )
        self.donor_bytes_accessed += int(nbytes)

    def _log(self, op: str, detail: dict[str, Any]) -> None:
        self.events.append({"op": op, **detail})

    def list_tensors(self, which: str, group: str | None = None) -> list[dict[str, Any]]:
        """List tensor metadata (never raw weights) for donor or recipient."""
        ckpt = self._checkpoint_for(which)
        index_path = ckpt / "model.safetensors.index.json"
        shards = (
            sorted({ckpt / v for v in json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values()})
            if index_path.exists()
            else sorted(ckpt.glob("*.safetensors"))
        )
        out: list[dict[str, Any]] = []
        for s in shards:
            hdr = inspect_safetensors_header(s)
            for name, info in hdr["tensors"].items():
                if group is not None and info["group"] != group:
                    continue
                out.append(
                    {
                        "name": name,
                        "group": info["group"],
                        "shape": info["shape"],
                        "dtype": info["dtype"],
                        "nbytes": info["nbytes"],
                    }
                )
        self._log("list_tensors", {"which": which, "group": group, "count": len(out)})
        return out

    def tensor_stats(self, which: str, tensor_name: str) -> dict[str, Any]:
        """Compute bounded summary statistics for a tensor."""
        ckpt = self._checkpoint_for(which)
        stats = tensor_summary_stats(ckpt, tensor_name)
        if which == "donor":
            self._charge_donor_bytes(stats["nbytes"], "tensor_stats", tensor_name)
        self._log("tensor_stats", {"which": which, "tensor": tensor_name, "nbytes": stats["nbytes"]})
        return stats

    def tensor_slice(
        self,
        which: str,
        tensor_name: str,
        row_start: int = 0,
        row_end: int | None = None,
        col_start: int | None = None,
        col_end: int | None = None,
        max_bytes: int = 65_536,
    ) -> dict[str, Any]:
        """Read a small bounded slice (default <= 64 KB) so matrices never flood context."""
        ckpt = self._checkpoint_for(which)
        res = read_tensor_slice(
            ckpt,
            tensor_name,
            row_start=row_start,
            row_end=row_end,
            col_start=col_start,
            col_end=col_end,
            max_bytes=max_bytes,
        )
        if which == "donor":
            self._charge_donor_bytes(res["nbytes"], "tensor_slice", tensor_name)
        self._log("tensor_slice", {"which": which, "tensor": tensor_name, "nbytes": res["nbytes"]})
        return res

    def tensor_svd_summary(
        self,
        which: str,
        tensor_name: str,
        top_k: int = 16,
    ) -> dict[str, Any]:
        """Compute top singular values and effective rank of a 2D weight matrix."""
        ckpt = self._checkpoint_for(which)
        shard = _resolve_tensor_file(ckpt, tensor_name)
        hdr = inspect_safetensors_header(shard)
        info = hdr["tensors"][tensor_name]
        if len(info["shape"]) != 2:
            raise ValueError(f"SVD summary requires 2D tensor, got shape={info['shape']}")
        if which == "donor":
            self._charge_donor_bytes(info["nbytes"], "tensor_svd_summary", tensor_name)

        with safe_open(shard, framework="pt", device="cpu") as f:
            w = f.get_tensor(tensor_name).to(torch.float32)

        svals = torch.linalg.svdvals(w)
        s2 = svals * svals
        total_energy = float(s2.sum().item())
        p = s2 / max(total_energy, 1e-12)
        p_pos = p[p > 1e-12]
        eff_rank = float(torch.exp(-torch.sum(p_pos * torch.log(p_pos))).item())
        k = min(top_k, int(svals.numel()))
        cum_energy_k = float(s2[:k].sum().item() / max(total_energy, 1e-12))

        out = {
            "tensor_name": tensor_name,
            "which": which,
            "shape": list(w.shape),
            "frobenius_norm": float(total_energy**0.5),
            "effective_rank": round(eff_rank, 2),
            "top_k": k,
            "top_k_energy_fraction": round(cum_energy_k, 4),
            "top_singular_values": [round(float(v), 5) for v in svals[:k].tolist()],
            "min_singular_value": float(svals[-1].item()),
        }
        self._log("tensor_svd_summary", {"which": which, "tensor": tensor_name, "eff_rank": out["effective_rank"]})
        return out

    def layer_alignment_diagnostics(self, layer_idx: int) -> dict[str, Any]:
        """Compare donor vs recipient tensor shapes, norms, and groups for a text layer."""
        if self.track == "C1":
            raise TrackViolationError("Track C1 forbids donor comparison diagnostics")
        prefix = f"model.language_model.layers.{layer_idx}."
        rec_tensors = [t for t in self.list_tensors("recipient") if t["name"].startswith(prefix)]
        don_map = {
            t["name"]: t for t in self.list_tensors("donor") if t["name"].startswith(prefix)
        }
        rows: list[dict[str, Any]] = []
        for rt in rec_tensors:
            dt = don_map.get(rt["name"])
            r_stats = self.tensor_stats("recipient", rt["name"])
            d_stats = self.tensor_stats("donor", rt["name"]) if dt else None
            rows.append(
                {
                    "name": rt["name"],
                    "group": classify_tensor_group(rt["name"]),
                    "recipient_shape": rt["shape"],
                    "donor_shape": dt["shape"] if dt else None,
                    "recipient_frob": round(r_stats["frobenius_norm"], 4),
                    "donor_frob": round(d_stats["frobenius_norm"], 4) if d_stats else None,
                    "recipient_std": round(r_stats["std"], 6),
                    "donor_std": round(d_stats["std"], 6) if d_stats else None,
                }
            )
        return {"layer_idx": layer_idx, "tensors": rows}

    def record_forward_pass(self, which: str, dataset_path: str | Path, count: int = 1) -> None:
        """Enforce track isolation and development-only data isolation on forward passes."""
        ds = str(Path(dataset_path).resolve())
        if "awst_isolated_eval" in ds or "/data/test" in ds:
            raise TrackViolationError(f"Access to isolated test path forbidden: {ds}")
        if "/data/validation" in ds:
            raise TrackViolationError(
                f"Agent activation/forward-pass probing must use data/development/ only, not {ds}"
            )
        if which == "donor":
            if self.track != "T2":
                raise TrackViolationError(
                    f"Donor forward passes are strictly forbidden in track={self.track}"
                )
            self.donor_forward_passes += int(count)
        elif which == "recipient":
            self.recipient_forward_passes += int(count)
        else:
            raise ValueError(f"Unknown model role: {which}")

    def record_dev_trial(self) -> None:
        """Enforce max candidate dev trials per agent session."""
        if self.dev_trials_used + 1 > self.max_dev_trials:
            raise BudgetExceededError(f"Exceeded max_dev_trials={self.max_dev_trials}")
        self.dev_trials_used += 1

    def save_budget_report(self, path: str | Path | None = None) -> dict[str, Any]:
        report = {
            "track": self.track,
            "max_donor_bytes": self.max_donor_bytes,
            "donor_bytes_accessed": self.donor_bytes_accessed,
            "donor_forward_passes": self.donor_forward_passes,
            "recipient_forward_passes": self.recipient_forward_passes,
            "max_dev_trials": self.max_dev_trials,
            "dev_trials_used": self.dev_trials_used,
            "events": self.events,
        }
        target = Path(path) if path else self.log_path
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
