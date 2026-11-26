"""Regression check for the H002 shuffled-source contract and the selected row budget."""

from pathlib import Path

import torch

from runs.codex_h002 import apply_t1 as t1


def test_override_is_authoritative_and_uses_the_identical_formula(tmp_path, monkeypatch):
    rng = torch.Generator().manual_seed(5)
    ids = [*t1.DIGITS, *(index for _, index in t1.ANCHORS)]
    recipient = torch.randn(len(ids), 1024, generator=rng).to(torch.bfloat16).double()
    # A strided view supplies the public checkpoint shape using only ~0.5 MB of storage.
    backing = torch.randn(248320 + 2047, generator=rng).to(torch.bfloat16)
    source = backing.as_strided((248320, 2048), (1, 1))
    donor_reads = []

    def read_rows(box, role, requested_ids, width):
        assert requested_ids == ids
        if role == "recipient":
            assert width == 1024
            return recipient
        assert box.track == "T1" and width == 2048
        donor_reads.append(tuple(requested_ids))
        return source[ids].double()

    def capture(box, base, out, manifest, original, delta, provenance, started):
        return (original + delta).to(torch.bfloat16), provenance

    class InaccessibleDonor:
        def __fspath__(self):
            raise AssertionError("An override must bypass even resolving donor_dir")

    monkeypatch.setattr(t1, "read_rows", read_rows)
    monkeypatch.setattr(t1, "verify_token_ids", lambda *args: None)
    monkeypatch.setattr(t1, "write_patch", capture)
    args = (tmp_path, tmp_path / "out", tmp_path / "manifest.json")
    regular, p = t1.apply_t1(Path("fixture"), *args)
    overridden, q = t1.apply_t1(InaccessibleDonor(), *args, {t1.EMBED: source})
    assert torch.equal(regular, overridden)
    assert p["source_rows_sha256_bf16"] == q["source_rows_sha256_bf16"]
    assert q["override_row_bytes_read"] == 163_840
    alternative = backing.flip(0).as_strided(source.shape, source.stride())
    changed, _ = t1.apply_t1(InaccessibleDonor(), *args, {t1.EMBED: alternative})
    assert not torch.equal(regular, changed)
    assert len(donor_reads) == 1
