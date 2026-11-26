"""H002 donor-free digit-contrast edit of the pinned Qwen3.5-0.8B checkpoint."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runs.codex_h002.patch_utils import DIGITS, RADII, center, read_rows, verify_token_ids, write_patch
from src.agent_interface.tools import AgentWeightToolbox

SELECTED = "low"


def apply_c1(recipient_dir, out_dir, manifest_path, *, strength=None):
    started = time.perf_counter()
    radius = RADII[SELECTED if strength is None else strength]
    box = AgentWeightToolbox(donor_dir=None, recipient_dir=recipient_dir, track="C1")
    verify_token_ids(recipient_dir, zip("0123456789", DIGITS))
    original = read_rows(box, "recipient", DIGITS, 1024)
    delta = radius * center(original)
    provenance = {"arm": "C1", "radius": radius, "donor_tensors": [],
                  "formula": "BF16(R + radius * (R - mean_rows(R)))"}
    return write_patch(box, recipient_dir, out_dir, manifest_path, original, delta,
                       provenance, started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipient-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--strength", choices=tuple(RADII), default=SELECTED)
    args = parser.parse_args()
    apply_c1(args.recipient_dir, args.out_dir, args.manifest_path, strength=args.strength)
