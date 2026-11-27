"""Automated verification tests for Milestone M4 manuscript, manifest, figures, and citations."""

from __future__ import annotations

from pathlib import Path

from src.paper.verify_paper_numbers import verify_paper_artifacts


ROOT = Path(__file__).resolve().parents[1]


def test_paper_numbers_citations_and_figures_verified() -> None:
    errors = verify_paper_artifacts()
    assert errors == [], f"Paper verification failures: {errors}"


def test_expected_publication_figures_present() -> None:
    fig_dir = ROOT / "paper" / "figures"
    expected_stems = [
        "fig1_split_trajectory",
        "fig2_m3_forest_contrasts",
        "fig3_edit_magnitude_vs_ood_ppl",
    ]
    for stem in expected_stems:
        for ext in (".png", ".pdf"):
            path = fig_dir / f"{stem}{ext}"
            assert path.exists(), f"Missing figure: {path}"
            assert path.stat().st_size > 1024, f"Unexpectedly small figure: {path}"
