"""H002 donor-to-recipient transfer of digit-to-arithmetic-token similarities."""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runs.codex_h002.patch_utils import (
    DIGITS, EMBED, RADII, center, read_rows, row_digest, verify_token_ids, write_patch,
)
from src.agent_interface.tools import AgentWeightToolbox

SELECTED = "low"
DONOR_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
RIDGE = 1 / 8
ANCHORS = (
    (" zero", 6942), (" one", 799), (" two", 1330), (" three", 2250), (" four", 2943),
    (" five", 4097), (" six", 4590), (" seven", 7840), (" eight", 7810), (" nine", 11292),
    ("+", 10), ("-", 12), ("*", 9), ("/", 14), ("=", 28), (" plus", 5346),
    (" minus", 26401), (" times", 2942), (" sum", 2542), (" total", 2702),
    (" digits", 17955), (" number", 1324), (" hundred", 7493), (" thousand", 15706),
    (" carry", 6564), (" borrow", 16686), (" equals", 16327), (" add", 884),
    (" subtract", 31192), (" multiply", 29283),
)


def geometry(rows):
    digits = center(rows[:len(DIGITS)])
    anchors = center(rows[len(DIGITS):])
    norms = torch.linalg.vector_norm(anchors, dim=1, keepdim=True)
    digit_norm = torch.linalg.vector_norm(digits)
    if not torch.isfinite(rows).all() or not digit_norm or (norms == 0).any():
        raise ValueError("Token geometry is nonfinite or degenerate")
    unit_anchors = anchors / norms
    return digits @ unit_anchors.T / digit_norm, unit_anchors, digit_norm


@torch.no_grad()
def apply_t1(donor_dir, recipient_dir, out_dir, manifest_path,
             donor_override: dict[str, torch.Tensor] | None = None, *, strength=None):
    started = time.perf_counter()
    radius = RADII[SELECTED if strength is None else strength]
    token_ids = list(zip("0123456789", DIGITS)) + list(ANCHORS)
    ids = [index for _, index in token_ids]
    verify_token_ids(recipient_dir, token_ids)
    if donor_override is None:
        box = AgentWeightToolbox(donor_dir=donor_dir, recipient_dir=recipient_dir, track="T1")
        donor = read_rows(box, "donor", ids, 2048)
        arm = "T1"
    else:
        # The override is the entire source, including the case of a missing key.
        # Use a recipient-only reader. Even constructing it cannot touch donor_dir.
        box = AgentWeightToolbox(donor_dir=None, recipient_dir=recipient_dir, track="C1")
        source = donor_override[EMBED]
        if source.shape != (248320, 2048) or source.dtype != torch.bfloat16:
            raise ValueError("C5 requires the full donor embedding shape and original bfloat16 dtype")
        donor = source[ids].detach().to(device="cpu", dtype=torch.float64)
        arm = "C5"
    recipient = read_rows(box, "recipient", ids, 1024)
    cr, anchors, digit_norm = geometry(recipient)
    cd, _, _ = geometry(donor)
    gram = anchors @ anchors.T
    gram.diagonal().add_(RIDGE)
    direction = center((cd - cr) @ torch.linalg.solve(gram, anchors))
    norm = torch.linalg.vector_norm(direction)
    if norm <= torch.finfo(direction.dtype).eps:
        raise ValueError("Donor and recipient yield a zero transfer direction")
    delta = (radius * digit_norm / norm) * direction
    original = recipient[:len(DIGITS)]
    after = recipient.clone()
    after[:len(DIGITS)] = (original + delta).to(torch.bfloat16).double()
    ca, _, _ = geometry(after)
    provenance = {
        "arm": arm, "radius": radius, "ridge": RIDGE, "donor_model": "Qwen/Qwen3.5-2B",
        "donor_revision": DONOR_REVISION, "source": "override" if arm == "C5" else "checkpoint",
        "donor_tensor": EMBED, "donor_shape": [248320, 2048], "source_dtype": "bfloat16",
        "row_order": [{"token": token, "id": idx} for token, idx in token_ids],
        "source_rows_sha256_bf16": row_digest(donor),
        "recipient_rows_sha256_bf16": row_digest(recipient),
        "override_row_bytes_read": (len(ids) * 2048 * 2) if arm == "C5" else 0,
        "formula": "C = center(digits) @ A.T / ||center(digits)||_F; A = row_unit(center(anchors)); "
                   "F = center((C_d-C_r) @ solve(A_r@A_r.T + ridge*I, A_r)); "
                   "digit_rows' = BF16(R + radius * ||center(R)||_F * F / ||F||_F)",
        "similarity_error_before": torch.linalg.vector_norm(cd - cr).item(),
        "similarity_error_after": torch.linalg.vector_norm(cd - ca).item(),
    }
    return write_patch(box, recipient_dir, out_dir, manifest_path, original, delta,
                       provenance, started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor-dir", required=True)
    parser.add_argument("--recipient-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--strength", choices=tuple(RADII), default=SELECTED)
    args = parser.parse_args()
    apply_t1(args.donor_dir, args.recipient_dir, args.out_dir, args.manifest_path,
             strength=args.strength)
