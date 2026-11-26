"""Shared recipient row access, immutable patch application and H002 accounting."""

import hashlib
import json
import tempfile
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.agent_interface.tools import AgentWeightToolbox
from src.interventions.patch_applier import apply_tensor_patches
from src.weight_io.safetensors_reader import sha256_file

EMBED = "model.language_model.embed_tokens.weight"
DIGITS = tuple(range(15, 25))
RADII = {"low": 1 / 32, "high": 1 / 8}
RECIPIENT_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def verify_token_ids(recipient_dir, token_ids):
    tokenizer = Tokenizer.from_file(str(Path(recipient_dir) / "tokenizer.json"))
    for token, token_id in token_ids:
        if tokenizer.encode(token, add_special_tokens=False).ids != [token_id]:
            raise ValueError(f"Unexpected token ID for {token!r}")


def read_rows(box: AgentWeightToolbox, role, ids, width):
    rows = []
    for index in ids:
        chunk = box.tensor_slice(role, EMBED, row_start=index, row_end=index + 1)
        if chunk["full_shape"] != [248320, width] or chunk["dtype"] != "torch.bfloat16":
            raise ValueError(f"Unexpected {role} embedding shape or dtype")
        rows.append(chunk["values"][0])
    return torch.tensor(rows, dtype=torch.float64)


def center(rows):
    return rows - rows.mean(dim=0, keepdim=True)


def row_digest(rows):
    raw = rows.to(torch.bfloat16).contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()


def write_patch(box, recipient_dir, out_dir, manifest_path, original, delta, provenance, started):
    base, out, manifest = (Path(p).resolve() for p in (recipient_dir, out_dir, manifest_path))
    pristine = [base, Path(__file__).resolve().parents[2] / "hf_cache/models"]
    if box.donor_dir is not None:
        pristine.append(box.donor_dir)
    if any(path == root or root in path.parents for path in (out, manifest) for root in pristine):
        raise ValueError("Output paths overlap a pristine checkpoint")
    if out.exists():
        raise FileExistsError(out)
    if out == manifest or out in manifest.parents or manifest.exists():
        raise ValueError("Use a fresh manifest path outside the output checkpoint")
    updated = (original + delta).to(torch.bfloat16)
    if not torch.isfinite(updated).all() or torch.equal(updated, original.to(torch.bfloat16)):
        raise ValueError("Digit patch is nonfinite or made no bfloat16 change")

    def patch(name, tensor):
        if name != EMBED:
            return None
        if not torch.equal(tensor[DIGITS[0]:DIGITS[-1] + 1], original.to(tensor.dtype)):
            raise AssertionError("Recipient digit rows differ from inspected rows")
        result = tensor.clone()
        result[DIGITS[0]:DIGITS[-1] + 1] = updated
        return result

    provenance.update({
        "recipient_model": "Qwen/Qwen3.5-0.8B", "recipient_revision": RECIPIENT_REVISION,
        "target_tensor": EMBED, "target_row_ids": list(DIGITS), "input_output_tied": True,
        "original_digit_rows_sha256_bf16": row_digest(original),
        "edited_digit_rows_sha256_bf16": row_digest(updated),
        "pre_round_delta_frobenius": torch.linalg.vector_norm(delta).item(),
        "gradient_steps": 0, "donor_forward_passes": 0, "recipient_forward_passes": 0,
        "donor_bytes_accessed": box.donor_bytes_accessed,
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=f".{out.name}.") as scratch:
        staged = Path(scratch) / "checkpoint"
        result = apply_tensor_patches(base, staged, patch, provenance, overwrite=False)
        inv = result["invariants"]
        if (inv["total_params"] != 873_438_784 or inv["frozen_tensors_verified"] != 153
                or [t["name"] for t in inv["changed_tensors"]] != [EMBED]
                or inv["changed_params"] != int((original.to(updated.dtype) != updated).sum())):
            raise AssertionError("Stored parameters, frozen vision tower, or edited support differ")
        metadata = {}
        for filename in ("config.json", "tokenizer.json", "tokenizer_config.json", "merges.txt",
                         "vocab.json", "chat_template.jinja"):
            digest = sha256_file(base / filename)
            if sha256_file(staged / filename) != digest:
                raise AssertionError(f"Checkpoint metadata changed: {filename}")
            metadata[filename] = digest
        inv["edited_dir"] = str(out)
        result["metadata_sha256"] = metadata
        result["toolbox"] = box.save_budget_report()
        result["patch_wall_seconds"] = time.perf_counter() - started
        with manifest.open("x", encoding="utf-8") as stream:
            try:
                stream.write(json.dumps(result, indent=2) + "\n")
                stream.flush()
                staged.rename(out)
            except BaseException:
                manifest.unlink()
                raise
    return result
