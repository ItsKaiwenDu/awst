"""Test the public T2 replay and C5 data path, without additional inference."""

import json
import tempfile
import time
from pathlib import Path

import torch
from safetensors import safe_open

from apply_t2 import apply_t2
from common import DONOR, EDITED, RECIPIENT, REC_SHA, RUN, SHARD, sha256_file, write_json
from src.interventions.patch_applier import verify_exact_reversion


class InaccessibleDonor:
    def __fspath__(self):
        raise AssertionError("An override must bypass the real donor directory")


def verify():
    selection = json.loads((RUN / "selection.json").read_text())
    label = selection["label"]
    original = json.loads((RUN / f"{label}_patch.json").read_text())
    tensors = list(original["provenance"]["sources"])
    start = time.perf_counter()
    with safe_open(DONOR / SHARD, framework="pt", device="cpu") as f:
        clean = {name: f.get_tensor(name) for name in tensors}
    # A common fixed permutation preserves the joint distribution of the two priors.
    permutation = torch.randperm(16, generator=torch.Generator().manual_seed(42))
    shuffled = {name: v[permutation] for name, v in clean.items()}
    source_bytes = sum(v.numel() * v.element_size() for v in clean.values())
    cases = {}
    for mode, override in (("regular", None), ("identical_override", clean), ("shuffled_override", shuffled)):
        with tempfile.TemporaryDirectory(prefix="awst-h003-verify-") as t:
            folder = Path(t)
            record = apply_t2(DONOR if override is None else InaccessibleDonor(), RECIPIENT,
                              folder / "checkpoint", folder / "patch.json", override)
            exact = record["edited_shard_sha256"] == selection["checkpoint_sha256"]
            if exact != (mode != "shuffled_override"):
                raise AssertionError(f"Unexpected replay digest in {mode}")
            cases[mode] = {"shard_sha256": record["edited_shard_sha256"], "equals_finalist": exact,
                           "changed_params": record["changed_params"],
                           "overall_relative_frobenius_norm": record["overall_relative_frobenius_norm"],
                           "sources": record["provenance"]["sources"], "runtime": record["runtime"],
                           "donor_bytes_sliced": record["provenance"]["donor_bytes_sliced"],
                           "override_bytes_after_storage_cast": record["provenance"]["override_bytes_after_storage_cast"],
                           "frozen_vision_tensors": record["invariants"]["frozen_tensors_verified"],
                           "wall_seconds": record["wall_seconds"]}
    with tempfile.TemporaryDirectory(prefix="awst-h003-revert-") as t:
        reverted = Path(t) / "reverted"
        inv = verify_exact_reversion(RECIPIENT, EDITED / f"h003_{label}", reverted)
        reversion = {"changed_params_vs_pristine": inv["changed_params"], "stored_params": inv["total_params"],
                     "frozen_vision_tensors": inv["frozen_tensors_verified"],
                     "pristine_tensor_contents_restored": inv["changed_params"] == 0,
                     "reverted_shard_sha256": sha256_file(reverted / SHARD),
                     "original_shard_sha256": REC_SHA}
    output = {"track": "T2", "selection": label, "donor_vectors_read_for_test_bytes": source_bytes,
              "permutation_new_head_to_original": permutation.tolist(), "replays": cases,
              "float64_delta_reversion": reversion, "donor_forward_passes": 0, "recipient_forward_passes": 0,
              "wall_seconds": time.perf_counter()-start}
    write_json(RUN / "replay.json", output)
    print("Verified real and identical replay, shuffled-value sensitivity and exact reversion.")


if __name__ == "__main__":
    verify()
