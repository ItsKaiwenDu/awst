"""Generate publication figures directly from verified Project AWST JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / "paper" / "figures"

# Colorblind-safe Okabe-Ito derived palette
C_BLUE = "#0072B2"
C_ORANGE = "#D55E00"
C_TEAL = "#009E73"
C_AMBER = "#E69F00"
C_PURPLE = "#CC79A7"
C_SKY = "#56B4E9"
C_GRAY = "#5C6670"
C_LIGHT_GRAY = "#B8C0C8"
C_DARK = "#1F2933"


def _load_json(rel_path: str) -> dict:
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8"))


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 12.0,
            "figure.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )


def _save_both(fig: plt.Figure, stem: str) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{stem}.png"
    pdf_path = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png_path, pdf_path]


def make_fig1_split_trajectory(
    s0: dict, conv: dict, h2: dict, h3: dict, m3: dict, h2_dev: dict, h3_dev: dict
) -> list[Path]:
    """Compare blinded validation vs isolated test accuracy across Stage 1 arms."""
    # Gather validation values strictly from stored JSONs
    s0_runs = {(r["model_label"], r["dataset_label"]): r for r in s0["nothink_baselines"]["runs"]}
    c0_val_g = s0_runs[("qwen3.5_0.8b_raw", "gsm8k")]["metrics"]["pass_at_1"] * 100.0
    c0_val_c = (
        s0_runs[("qwen3.5_0.8b_raw", "synthetic")]["metrics"]["by_subskill"]["carry_add_sub"][
            "pass_at_1"
        ]
        * 100.0
    )
    donor_val_g = s0_runs[("qwen3.5_2b", "gsm8k")]["metrics"]["pass_at_1"] * 100.0
    donor_val_c = (
        s0_runs[("qwen3.5_2b", "synthetic")]["metrics"]["by_subskill"]["carry_add_sub"][
            "pass_at_1"
        ]
        * 100.0
    )

    val_data = {
        "C0_raw": (c0_val_g, c0_val_c),
        "C1": (
            h2["validation"]["C1"]["gsm8k_val"]["pass_at_1"] * 100.0,
            h2["validation"]["C1"]["synthetic_val"]["by_subskill"]["carry_add_sub"]["pass_at_1"]
            * 100.0,
        ),
        "C4_weights": (
            conv["validation"]["C4_weights"]["gsm8k_val"]["pass_at_1"] * 100.0,
            conv["validation"]["C4_weights"]["synthetic_val"]["by_subskill"]["carry_add_sub"][
                "pass_at_1"
            ]
            * 100.0,
        ),
        "C4_act": (
            conv["validation"]["C4_act"]["gsm8k_val"]["pass_at_1"] * 100.0,
            conv["validation"]["C4_act"]["synthetic_val"]["by_subskill"]["carry_add_sub"][
                "pass_at_1"
            ]
            * 100.0,
        ),
        "T1": (
            h2["validation"]["T1"]["gsm8k_val"]["pass_at_1"] * 100.0,
            h2["validation"]["T1"]["synthetic_val"]["by_subskill"]["carry_add_sub"]["pass_at_1"]
            * 100.0,
        ),
        "C5_T1_rowperm": (
            h2["validation"]["C5_rowperm"]["gsm8k_val"]["pass_at_1"] * 100.0,
            h2["validation"]["C5_rowperm"]["synthetic_val"]["by_subskill"]["carry_add_sub"][
                "pass_at_1"
            ]
            * 100.0,
        ),
        "T2": (
            h3["validation"]["T2"]["gsm8k_val"]["pass_at_1"] * 100.0,
            h3["validation"]["T2"]["synthetic_val"]["by_subskill"]["carry_add_sub"]["pass_at_1"]
            * 100.0,
        ),
        "C5_T2_norm_matched": (
            h3["validation"]["C5_norm_matched"]["gsm8k_val"]["pass_at_1"] * 100.0,
            h3["validation"]["C5_norm_matched"]["synthetic_val"]["by_subskill"]["carry_add_sub"][
                "pass_at_1"
            ]
            * 100.0,
        ),
        "Qwen3.5-2B": (donor_val_g, donor_val_c),
    }

    arms_order = [
        ("C0_raw", "C0 raw\n(0.8B)"),
        ("C1", "C1\n(donor-\nfree)"),
        ("C4_weights", "C4\nweights\n(SVD)"),
        ("C4_act", "C4 act\n(APM+\nridge)"),
        ("T1", "T1\n(H002\nwts-only)"),
        ("C5_T1_rowperm", "C5-T1\n(row-\nperm)"),
        ("T2", "T2\n(H003\nact)"),
        ("C5_T2_norm_matched", "C5-T2\n(norm-\nmatch)"),
        ("Qwen3.5-2B", "Qwen3.5\n-2B\n(donor)"),
    ]

    test_data = {}
    for arm_key, _ in arms_order:
        ds = m3["arms"][arm_key]["datasets"]
        test_data[arm_key] = (
            ds["gsm8k_test"]["metrics"]["pass_at_1"] * 100.0,
            ds["synthetic_test"]["metrics"]["by_subskill"]["carry_add_sub"]["pass_at_1"] * 100.0,
        )

    # Compute exact dev deltas for callout note directly from H002/H003 JSONs
    t1_dev_rem = (
        h2_dev["datasets"]["gsm8k_dev"]["remainder"]["correct"]["t1_low"]
        - h2_dev["datasets"]["gsm8k_dev"]["remainder"]["correct"]["c0"]
    ) / h2_dev["datasets"]["gsm8k_dev"]["remainder"]["n"] * 100.0
    t2_dev_fu = (
        h3_dev["scores"]["followup"]["half"]["gsm8k"]["pass_at_1"]
        - h3_dev["scores"]["followup"]["c0"]["gsm8k"]["pass_at_1"]
    ) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.1), sharey=False)
    x = np.arange(len(arms_order))
    width = 0.37

    panels = [
        (
            axes[0],
            0,
            "A. GSM8K Exact-Answer Pass@1 (%)",
            "Blinded Val (n=100) vs. Isolated Test (n=200)",
        ),
        (
            axes[1],
            1,
            "B. Carry/Borrow Arithmetic Pass@1 (%)",
            "Blinded Val (n=50) vs. Isolated Test (n=50)",
        ),
    ]

    for ax, idx, title, subtitle in panels:
        val_vals = [val_data[k][idx] for k, _ in arms_order]
        test_vals = [test_data[k][idx] for k, _ in arms_order]

        val_colors = []
        test_colors = []
        for k, _ in arms_order:
            if k == "T1":
                val_colors.append("#92C5DE")
                test_colors.append(C_BLUE)
            elif k == "T2":
                val_colors.append("#F4A582")
                test_colors.append(C_ORANGE)
            elif k == "Qwen3.5-2B":
                val_colors.append("#8CD17D")
                test_colors.append(C_TEAL)
            else:
                val_colors.append("#CBD2D9")
                test_colors.append(C_GRAY)

        bars_v = ax.bar(
            x - width / 2,
            val_vals,
            width,
            color=val_colors,
            edgecolor="#333333",
            linewidth=0.6,
            label="Blinded Validation",
        )
        bars_t = ax.bar(
            x + width / 2,
            test_vals,
            width,
            color=test_colors,
            edgecolor="#333333",
            linewidth=0.6,
            label="Isolated Test",
        )

        # Horizontal reference lines for C0_raw
        ax.axhline(val_vals[0], color="#88929B", linestyle=":", linewidth=1.0, zorder=1)
        ax.axhline(test_vals[0], color=C_DARK, linestyle="--", linewidth=1.0, zorder=1)

        label_bbox = dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none", alpha=0.82)
        for b in bars_v:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 1.3,
                f"{h:.0f}",
                ha="center",
                va="bottom",
                fontsize=7.3,
                color="#4B5563",
                bbox=label_bbox,
                zorder=5,
            )
        for b in bars_t:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 1.3,
                f"{h:.1f}" if h % 1 != 0 else f"{h:.0f}",
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight="bold",
                color=C_DARK,
                bbox=label_bbox,
                zorder=5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in arms_order], fontsize=7.6)
        ax.set_ylim(0, 98)
        ax.set_ylabel("Accuracy (% Pass@1)")
        ax.set_title(f"{title}\n{subtitle}", loc="left", pad=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35, color="#9AA5B1")
        ax.set_axisbelow(True)

    # Clear custom legend on Panel A
    legend_handles = [
        mpatches.Patch(facecolor="#CBD2D9", edgecolor="#333333", label="Blinded Validation (M2)"),
        mpatches.Patch(facecolor=C_GRAY, edgecolor="#333333", label="Isolated Test (M3, Controls)"),
        mpatches.Patch(facecolor=C_BLUE, edgecolor="#333333", label="T1 (Weights-Only AWST)"),
        mpatches.Patch(facecolor=C_ORANGE, edgecolor="#333333", label="T2 (Activation-Assisted AWST)"),
        mpatches.Patch(facecolor=C_TEAL, edgecolor="#333333", label="Qwen3.5-2B Donor"),
    ]
    axes[0].legend(
        handles=legend_handles,
        loc="upper left",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#D1D5DB",
        fontsize=7.6,
    )

    axes[0].annotate(
        f"Dev GSM8K gains (T1 +{t1_dev_rem:.1f} pp rem.;\nT2 +{t2_dev_fu:.1f} pp follow-up) vanished on val/test",
        xy=(4 + width / 2, test_data["T1"][0] + 6),
        xytext=(3.2, 69),
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFBEB", edgecolor="#D97706", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="#B45309", lw=0.9),
    )

    fig.suptitle(
        "Stage 1 Cross-Split Accuracy: Development-to-Held-Out Mirage & Control Overlap",
        y=1.03,
    )
    fig.tight_layout()
    return _save_both(fig, "fig1_split_trajectory")


def make_fig2_m3_forest_contrasts(m3: dict) -> list[Path]:
    """Plot M3 paired percentage-point differences and 95% bootstrap CIs across key contrasts."""
    pc = m3["paired_comparisons"]
    holm = m3["holm_bonferroni"]

    # Contrast specification: (key_in_pc, label, track, is_primary_holm, color)
    specs = [
        ("C0_raw_to_T1", "T1 - C0 raw", "T1", False, C_BLUE),
        ("C1_to_T1", "T1 - C1 (donor-free)*", "T1", True, C_BLUE),
        ("C4_weights_to_T1", "T1 - C4 weights*", "T1", True, C_BLUE),
        ("C5_T1_rowperm_to_T1", "T1 - C5 row-perm", "T1", False, C_SKY),
        ("C0_raw_to_T2", "T2 - C0 raw", "T2", False, C_ORANGE),
        ("C1_to_T2", "T2 - C1 (donor-free)*", "T2", True, C_ORANGE),
        ("C4_act_to_T2", "T2 - C4 activation*", "T2", True, C_ORANGE),
        ("C5_T2_norm_matched_to_T2", "T2 - C5 norm-matched", "T2", False, C_PURPLE),
    ]

    endpoints = [
        ("gsm8k_test", "A. GSM8K Test (n=200)", (-16, 20), 21.0),
        ("carry_add_sub", "B. Carry Arithmetic (n=50)", (-22, 30), 31.5),
        ("synthetic_ood_test", "C. Synthetic OOD (n=100)", (-20, 20), 21.0),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharey=True)
    y_positions = np.array([7.5, 6.5, 5.5, 4.5, 3.0, 2.0, 1.0, 0.0])

    for panel_idx, (ax, (ep_key, ep_title, xlim, text_x)) in enumerate(zip(axes, endpoints)):
        ax.axvline(0.0, color="#6B7280", linestyle="-", linewidth=1.0, zorder=1)
        if ep_key in ("gsm8k_test", "carry_add_sub"):
            ax.axvline(5.0, color=C_AMBER, linestyle="--", linewidth=1.1, zorder=1)
        ax.axhline(3.75, color="#D1D5DB", linestyle="-", linewidth=0.9)

        for row_idx, (pc_key, label, track, is_prim, color) in enumerate(specs):
            y = y_positions[row_idx]
            entry = pc[pc_key][ep_key]
            diff = entry["delta_pp"]
            lo, hi = entry["bootstrap_95ci_pp"]
            pval = entry["mcnemar_exact_p"]

            ax.errorbar(
                diff,
                y,
                xerr=[[diff - lo], [hi - diff]],
                fmt="o" if track == "T1" else "s",
                color=color,
                ecolor=color,
                elinewidth=1.8,
                capsize=3.5,
                markersize=6.0,
                zorder=3,
            )

            holm_key_map = {
                ("T1", "C1_to_T1"): "T1_vs_C1",
                ("T1", "C4_weights_to_T1"): "T1_vs_C4_weights",
                ("T2", "C1_to_T2"): "T2_vs_C1",
                ("T2", "C4_act_to_T2"): "T2_vs_C4_act",
            }
            if is_prim:
                hp = holm[track][ep_key][holm_key_map[(track, pc_key)]]
                stat_txt = f"{diff:+.1f} [{lo:+.1f}, {hi:+.1f}]  p={pval:.3f} (H={hp:.3f})"
            else:
                stat_txt = f"{diff:+.1f} [{lo:+.1f}, {hi:+.1f}]  p={pval:.3f}"

            ax.text(
                xlim[1] + 0.6,
                y,
                stat_txt,
                va="center",
                ha="left",
                fontsize=7.3,
                color=C_DARK,
                clip_on=False,
            )

        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(-0.7, 8.2)
        ax.set_xlabel("Paired Difference (percentage points)")
        ax.set_title(ep_title, loc="left", pad=8)
        ax.xaxis.grid(True, linestyle=":", alpha=0.45, color="#9AA5B1")
        ax.set_axisbelow(True)

        if ep_key == "gsm8k_test":
            ax.text(5.3, 8.0, "+5 pp target", color="#B45309", fontsize=7.5, va="top")

        if panel_idx == 0:
            ax.set_yticks(y_positions)
            ax.set_yticklabels([lbl for _, lbl, _, _, _ in specs], fontsize=8.4)
            ax.text(xlim[0] + 0.6, 7.95, "Track T1 (Weights-Only)", fontsize=8.0, fontweight="bold", color=C_BLUE)
            ax.text(xlim[0] + 0.6, 3.35, "Track T2 (Act-Assisted)", fontsize=8.0, fontweight="bold", color=C_ORANGE)

    fig.suptitle(
        "M3 Isolated Paired Contrasts (95% Bootstrap CI, Exact McNemar p, and Track-Holm Adjusted H)",
        y=1.02,
    )
    fig.subplots_adjust(wspace=0.92, right=0.88)
    return _save_both(fig, "fig2_m3_forest_contrasts")


def make_fig3_edit_magnitude_vs_ood_ppl(m3: dict) -> list[Path]:
    """Plot edit magnitude (rel. Frobenius norm) vs OOD accuracy and prose perplexity (H3)."""
    arms = [
        ("C1", "C1 (9.1K)", C_GRAY, "o", (-4, 8)),
        ("T1", "T1 (9.1K)", C_BLUE, "D", (6, -10)),
        ("C5_T1_rowperm", "C5-T1 (9.1K)", C_SKY, "^", (-18, -14)),
        ("T2", "T2 (2 params)", C_ORANGE, "s", (6, -10)),
        ("C5_T2_norm_matched", "C5-T2 (2 params)", C_PURPLE, "v", (6, 6)),
        ("C4_act", "C4 act (168.3M)", C_TEAL, "P", (-45, 7)),
        ("C4_weights", "C4 weights (172.5M)", C_AMBER, "X", (-68, -12)),
        ("C3", "C3 random slice (173.2M)", "#B91C1C", "*", (-105, 8)),
    ]

    c0_ood = m3["arms"]["C0_raw"]["datasets"]["synthetic_ood_test"]["metrics"]["pass_at_1"] * 100.0
    c0_ppl = m3["arms"]["C0_raw"]["prose_perplexity"]["perplexity"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.1))

    for ax, metric_name, base_val, ylabel, ylim, title in [
        (
            ax1,
            "ood",
            c0_ood,
            "Synthetic OOD Accuracy (%, n=100)",
            (54, 86),
            "A. Isolated OOD Generalization vs. Edit Magnitude",
        ),
        (
            ax2,
            "ppl",
            c0_ppl,
            "Held-Out Prose Perplexity (10 passages, lower = better)",
            (56.5, 84.5),
            "B. General Fluency Degradation (Perplexity) vs. Edit Magnitude",
        ),
    ]:
        ax.axhline(
            base_val,
            color=C_DARK,
            linestyle="--",
            linewidth=1.1,
            label=f"C0_raw unedited baseline ({base_val:.1f}{'%' if metric_name=='ood' else ''})",
        )

        for arm_key, label, color, marker, offset in arms:
            arm_obj = m3["arms"][arm_key]
            x_val = arm_obj["patch_invariants"]["overall_relative_frobenius_norm"]
            if metric_name == "ood":
                y_val = arm_obj["datasets"]["synthetic_ood_test"]["metrics"]["pass_at_1"] * 100.0
                custom_offsets = {
                    "C1": (-46, -26),
                    "T1": (22, -22),
                    "C5_T1_rowperm": (8, 16),
                    "T2": (12, -10),
                    "C5_T2_norm_matched": (10, 14),
                    "C4_act": (-42, 12),
                    "C4_weights": (-82, -14),
                    "C3": (-95, -18),
                }[arm_key]
            else:
                y_val = arm_obj["prose_perplexity"]["perplexity"]
                custom_offsets = {
                    "C1": (-16, 26),
                    "T1": (18, 14),
                    "C5_T1_rowperm": (-12, -28),
                    "T2": (12, -26),
                    "C5_T2_norm_matched": (12, 20),
                    "C4_act": (-55, 12),
                    "C4_weights": (-88, -8),
                    "C3": (-98, -12),
                }[arm_key]

            ms = 10.5 if marker == "*" else 7.2
            ax.plot(
                x_val,
                y_val,
                marker=marker,
                markersize=ms,
                color=color,
                markeredgecolor="#222222",
                markeredgewidth=0.6,
                linestyle="None",
                zorder=4,
            )
            ax.annotate(
                f"{label}\n({y_val:.1f}{'%' if metric_name=='ood' else ''})",
                xy=(x_val, y_val),
                xytext=custom_offsets,
                textcoords="offset points",
                fontsize=7.5,
                color=C_DARK,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color="#9AA5B1", lw=0.75),
                zorder=5,
            )

        ax.set_xscale("log")
        ax.set_xlim(1.6e-5, 2.5e-2)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Whole-Model Relative Frobenius Edit Norm (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=8)
        ax.grid(True, which="major", linestyle=":", alpha=0.45, color="#9AA5B1")
        ax.legend(loc="lower left" if metric_name == "ood" else "upper left", frameon=True, facecolor="white", edgecolor="#D1D5DB", fontsize=8.0)

    # Highlight H3 gap on Panel A
    c3_ood = m3["arms"]["C3"]["datasets"]["synthetic_ood_test"]["metrics"]["pass_at_1"] * 100.0
    c3_norm = m3["arms"]["C3"]["patch_invariants"]["overall_relative_frobenius_norm"]
    ax1.annotate(
        f"H3 OOD drop: {c3_ood - c0_ood:+.1f} pp\n(p = 3.59e-5 vs C0)",
        xy=(c3_norm, c3_ood),
        xytext=(-135, 34),
        textcoords="offset points",
        fontsize=7.8,
        color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2", edgecolor="#DC2626", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=0.9),
    )

    fig.suptitle(
        "Edit Scale vs. OOD Robustness & Prose Fluency (Sparse AWST vs. Dense C4 & Unaligned C3)",
        y=1.02,
    )
    fig.tight_layout()
    return _save_both(fig, "fig3_edit_magnitude_vs_ood_ppl")


def main() -> int:
    _configure_style()
    s0 = _load_json("results/m1_stage0_baselines.json")
    conv = _load_json("results/m2_conventional_baselines.json")
    h2 = _load_json("results/m2_h002_verification.json")
    h3 = _load_json("results/m2_h003_verification.json")
    m3 = _load_json("results/m3_isolated_evaluation.json")
    h2_dev = _load_json("runs/codex_h002/dev_results.json")
    h3_dev = _load_json("runs/codex_h003/dev_results.json")

    paths: list[Path] = []
    paths.extend(make_fig1_split_trajectory(s0, conv, h2, h3, m3, h2_dev, h3_dev))
    paths.extend(make_fig2_m3_forest_contrasts(m3))
    paths.extend(make_fig3_edit_magnitude_vs_ood_ppl(m3))

    for p in paths:
        print(f"generated: {p.relative_to(ROOT)} ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
