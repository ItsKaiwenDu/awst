"""Apply the H003 finalist to the pinned local recipient, using frozen T2 localization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runs.codex_h003.common import (DON_REV, DON_SHA, REC_REV, REC_SHA, RUN, SHARD,
                                    schema, sha256_file, write_json)
from src.agent_interface.tools import AgentWeightToolbox
from src.evaluation.runner import load_model_and_tokenizer
from src.interventions.patch_applier import apply_tensor_patches


def apply_candidate(donor_dir, recipient_dir, out_dir, manifest_path, strength, donor_override=None):
    started = time.perf_counter()
    base, out, manifest = (Path(p).resolve() for p in (recipient_dir, out_dir, manifest_path))
    source = Path(donor_dir).resolve() if donor_override is None else None
    protected = [base] if source is None else [base, source]
    if any(out.is_relative_to(p) or manifest.is_relative_to(p) for p in protected):
        raise ValueError("Output paths must be outside the source checkpoints")
    if out.exists() or manifest.exists() or manifest == out or out.is_relative_to(manifest):
        raise FileExistsError("A fresh output directory and manifest file are required")
    if strength not in (0.5, 1.0):
        raise ValueError("Unexpected H003 strength")
    if sha256_file(base / SHARD) != REC_SHA or (source is not None and sha256_file(source / SHARD) != DON_SHA):
        raise ValueError("Checkpoint SHA-256 differs from the pinned H003 inputs")

    evidence_path = RUN / "localization.json"
    evidence = json.loads(evidence_path.read_text())
    if (evidence["track"], evidence["recipient_shard_sha256"], evidence["donor_shard_sha256"]) != (
            "T2", REC_SHA, DON_SHA):
        raise ValueError("Incompatible localization evidence")
    pair = evidence["selected"]
    prefix = f"model.language_model.layers.{pair['layer']}.linear_attn."
    donor_head, recipient_head = pair["donor_head"], pair["recipient_head"]
    offset = pair["donor_digit_a_median"] - pair["recipient_digit_a_median"]
    # With C5, even an inaccessible donor_dir is supported. No real donor fallback.
    box = AgentWeightToolbox(source, base, track="T2") if source is not None else None
    sources = {}
    for suffix, dtype in (("A_log", torch.float32), ("dt_bias", torch.bfloat16)):
        key = prefix + suffix
        if donor_override is not None:
            value = donor_override[key].detach().cpu()
            if value.shape != (16,) or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"Expected a finite donor head vector of shape (16,) for {key}")
            vector = value.to(dtype)
        else:
            block = box.tensor_slice("donor", key, row_start=0, row_end=16, max_bytes=64)
            if block["full_shape"] != [16] or block["dtype"] != str(dtype):
                raise ValueError(f"Unexpected donor schema for {key}")
            vector = torch.tensor(block["values"], dtype=dtype)
        if not torch.isfinite(vector).all():
            raise ValueError(f"Nonfinite donor vector for {key}")
        chosen = float(vector[donor_head])
        sources[key] = {"source_tensor": key, "source_head": donor_head, "recipient_tensor": key,
                        "recipient_head": recipient_head, "source_dtype": str(dtype),
                        "source_vector_sha256": hashlib.sha256(vector.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                        "donor_value": chosen, "activation_offset": offset if suffix == "dt_bias" else 0.0}

    def transplant(name, old):
        if name not in sources:
            return None
        s = sources[name]
        before = float(old[recipient_head])
        target = s["donor_value"] + s["activation_offset"]
        after = old.clone()
        after[recipient_head] = before + strength * (target - before)
        assert math.isfinite(float(after[recipient_head]))
        s.update(recipient_before=before, recipient_after=float(after[recipient_head]))
        return after

    provenance = {"track": "T2", "condition": "T2" if box else "C5_frozen_T2_localization",
                  "recipient_revision": REC_REV, "localization_donor_revision": DON_REV,
                  "transplant_value_source": DON_REV if box else "donor_override",
                  "localization_sha256": sha256_file(evidence_path), "strength": strength,
                  "formula": "cast_to_storage(r + strength * (selected_donor_value + activation_offset - r))",
                  "sources": sources, "gradient_steps": 0, "donor_forward_passes": 0,
                  "recipient_forward_passes": 0, "donor_bytes_sliced": box.donor_bytes_accessed if box else 0,
                  "override_bytes_after_storage_cast": 0 if box else 96}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="h003-build-", dir=out.parent) as tmp:
        staged = Path(tmp) / "checkpoint"
        record = apply_tensor_patches(base, staged, transplant, provenance)
        inv = record["invariants"]
        if (inv["total_params"], inv["frozen_tensors_verified"]) != (873438784, 153):
            raise RuntimeError("Stored parameter or frozen vision count mismatch")
        if {t["name"] for t in inv["changed_tensors"]} - sources.keys() or inv["changed_params"] > 2:
            raise RuntimeError("Out-of-scope checkpoint change")
        if any("recipient_after" not in s for s in sources.values()):
            raise ValueError("Missing destination tensors")
        files = {p.name: sha256_file(p) for p in base.iterdir() if p.is_file() and p.suffix != ".safetensors"}
        if any(sha256_file(staged / name) != digest for name, digest in files.items()):
            raise RuntimeError("Configuration or tokenizer file differs")
        runtime = []
        for p in (base, staged):
            model, _, _, _ = load_model_and_tokenizer(p, "cpu", "bfloat16")
            runtime.append(schema(model))
            del model
        if runtime[0] != runtime[1] or runtime[0]["runtime_parameters"] != 852985920 or not runtime[0]["tied_embeddings"]:
            raise RuntimeError("The loaded architectures differ from the pristine recipient")
        inv["edited_dir"] = str(out)
        record.update(runtime=runtime[1], metadata_sha256=files, edited_shard_sha256=sha256_file(staged / SHARD),
                      changed_params=inv["changed_params"], overall_relative_frobenius_norm=inv["overall_relative_frobenius_norm"],
                      wall_seconds=time.perf_counter() - started)
        if manifest.is_relative_to(out):
            dest = staged / manifest.relative_to(out)
            if dest.exists():
                raise FileExistsError(f"Manifest conflicts with checkpoint contents: {manifest}")
            write_json(dest, record)
        staged.rename(out)
        if not manifest.is_relative_to(out):
            try:
                write_json(manifest, record)
            except BaseException:
                out.rename(staged)
                raise
    return record


def apply_t2(donor_dir, recipient_dir, out_dir, manifest_path,
             donor_override: dict[str, torch.Tensor] | None = None):
    """Rebuild the frozen finalist. Override both named 16-head donor vectors for C5."""
    selection = json.loads((RUN / "selection.json").read_text())
    if selection["localization_sha256"] != sha256_file(RUN / "localization.json"):
        raise ValueError("The H003 localization record has changed since selection")
    return apply_candidate(donor_dir, recipient_dir, out_dir, manifest_path, selection["strength"], donor_override)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("donor-dir", "recipient-dir", "out-dir", "manifest-path"):
        parser.add_argument("--" + arg, required=True)
    args = parser.parse_args()
    result = apply_t2(**vars(args))
    print(result["edited_shard_sha256"], result["changed_params"])
