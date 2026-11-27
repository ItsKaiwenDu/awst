"""Programmatically verify numerical claims, citations, figures, and hygiene in Milestone M4."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_json(rel_path: str) -> dict:
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8"))


def verify_paper_artifacts() -> list[str]:
    """Run all automated checks against paper/ and stored results; raise AssertionError on mismatch."""
    errors: list[str] = []

    paper_path = ROOT / "paper" / "awst_paper.md"
    manifest_path = ROOT / "paper" / "artifact_manifest.md"
    paper_bib_path = ROOT / "paper" / "references.bib"
    docs_bib_path = ROOT / "docs" / "literature" / "bibliography.bib"
    progress_path = ROOT / "docs" / "progress.md"

    for p in (paper_path, manifest_path, paper_bib_path, docs_bib_path, progress_path):
        if not p.exists():
            errors.append(f"Missing required artifact: {p.relative_to(ROOT)}")
            return errors

    paper_text = paper_path.read_text(encoding="utf-8")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    paper_bib_text = paper_bib_path.read_text(encoding="utf-8")
    docs_bib_text = docs_bib_path.read_text(encoding="utf-8")
    progress_text = progress_path.read_text(encoding="utf-8")

    # 1. Bibliography sync and citation resolution
    if paper_bib_text != docs_bib_text:
        errors.append("paper/references.bib is not byte-identical to docs/literature/bibliography.bib")

    bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", paper_bib_text))
    # Find Pandoc citation keys [@key] inside prose (including multi-key [@a; @b])
    cited_keys: set[str] = set()
    for grp in re.findall(r"\[([^\]]*@[^\]]+)\]", paper_text):
        for key in re.findall(r"@([A-Za-z0-9_]+)", grp):
            cited_keys.add(key)

    if not cited_keys:
        errors.append("No [@key] citations found in paper/awst_paper.md")
    missing_keys = sorted(cited_keys - bib_keys)
    if missing_keys:
        errors.append(f"Citations in paper missing from references.bib: {missing_keys}")

    # 2. Figures exist (PNG + PDF) and are linked cleanly
    img_links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", paper_text)
    if len(img_links) < 3:
        errors.append(f"Expected >= 3 figure links in paper/awst_paper.md, found {len(img_links)}")
    for rel_img in img_links:
        img_file = (ROOT / "paper" / rel_img).resolve()
        if not img_file.exists() or img_file.stat().st_size == 0:
            errors.append(f"Broken or empty figure image link: {rel_img}")
        pdf_pair = img_file.with_suffix(".pdf")
        if not pdf_pair.exists() or pdf_pair.stat().st_size == 0:
            errors.append(f"Missing matching PDF figure: {pdf_pair.relative_to(ROOT)}")

    # 3. Hygiene: no uncalibrated 'null result confirmed' and no isolated test leaks
    combined_doc = (paper_text + "\n" + manifest_text + "\n" + progress_text).lower()
    if "null result confirmed" in combined_doc:
        errors.append("Forbidden uncalibrated phrase 'null result confirmed' found in docs/paper")
    if "usage reset" in combined_doc or "5-hour" in combined_doc:
        errors.append("Private operational remark leaked into docs/paper")

    # 4. Quantitative cross-checks against stored JSONs
    m3 = _load_json("results/m3_isolated_evaluation.json")
    s0 = _load_json("results/m1_stage0_baselines.json")
    remote = _load_json("results/m1_remote_reader_validation.json")
    audit = _load_json("runs/manifests/m3_frozen_checkpoint_audit.json")
    h003_acc = _load_json("runs/codex_h003/accounting.json")
    rq6 = _load_json("results/m2_rq6_competency_comparison.json")

    expected_tokens = [
        # Param counts & invariants
        f"{audit['persistent_arms']['C0_raw']['total_params']:,}",
        "852,985,920",
        # Remote reader HTTP 206
        f"{remote['bytes_read']:,}",
        f"{remote['total_file_bytes']:,}",
        f"{remote['byte_reduction_ratio']:,.2f}x",
        # Shadow & direct spend
        "$0.00",
        "$14.2912",
        # M3 key accuracies & stats
        f"{m3['arms']['C0_raw']['datasets']['gsm8k_test']['metrics']['pass_at_1'] * 100:.1f}%",
        f"{m3['arms']['Qwen3.5-2B']['datasets']['gsm8k_test']['metrics']['pass_at_1'] * 100:.1f}%",
        f"{m3['arms']['T1']['datasets']['gsm8k_test']['metrics']['pass_at_1'] * 100:.1f}%",
        f"{m3['arms']['T2']['datasets']['gsm8k_test']['metrics']['pass_at_1'] * 100:.1f}%",
        f"{m3['arms']['C3']['datasets']['synthetic_ood_test']['metrics']['pass_at_1'] * 100:.1f}%",
        f"{m3['arms']['C3']['prose_perplexity']['perplexity']:.2f}",
        f"{m3['arms']['C0_raw']['prose_perplexity']['perplexity']:.2f}",
        f"{m3['holm_bonferroni']['T1']['gsm8k_test']['T1_vs_C1']:.3f}",
        f"{m3['holm_bonferroni']['T1']['gsm8k_test']['T1_vs_C4_weights']:.3f}",
        f"{m3['holm_bonferroni']['T2']['gsm8k_test']['T2_vs_C1']:.3f}",
        f"{m3['holm_bonferroni']['T2']['gsm8k_test']['T2_vs_C4_act']:.3f}",
    ]
    for tok in expected_tokens:
        if tok not in paper_text:
            errors.append(f"Expected verified token missing from paper/awst_paper.md: {tok!r}")

    # Cross-check exact cumulative shadow arithmetic from stored receipts
    total_shadow = (
        rq6["gpt_6_2_astra"]["in_turn_shadow_usd"]
        + h003_acc["h002_closed_usd"]
        + h003_acc["h003_shadow_usd"]
        + 1.161018
    )
    if f"{total_shadow:.7f}" not in manifest_text:
        errors.append(f"Expected exact shadow total {total_shadow:.7f} in paper/artifact_manifest.md")
    if f"{s0['loopfix_patch_summary']['changed_params']:,}" not in paper_text:
        errors.append("Missing Stage 0 loopfix changed_params count in paper/awst_paper.md")

    return errors


def verify_latex_manuscript() -> list[str]:
    """Verify Overleaf-ready LaTeX manuscript (ASCII, braces, envs, citations, figures, numbers)."""
    errors: list[str] = []
    tex_path = ROOT / "paper" / "main.tex"
    bib_path = ROOT / "paper" / "references.bib"
    if not tex_path.exists():
        return [f"Missing required Overleaf LaTeX file: {tex_path.relative_to(ROOT)}"]
    if not bib_path.exists():
        return [f"Missing BibTeX file: {bib_path.relative_to(ROOT)}"]

    tex = tex_path.read_text(encoding="utf-8")
    bib = bib_path.read_text(encoding="utf-8")

    # 1. Pure ASCII encoding for clean Overleaf pdfLaTeX compatibility
    non_ascii = [(i + 1, ch) for i, ch in enumerate(tex) if ord(ch) >= 128]
    if non_ascii:
        errors.append(f"paper/main.tex contains non-ASCII characters: {non_ascii[:5]}")

    # 2. Required structural macros
    for req in (
        "\\documentclass",
        "\\begin{document}",
        "\\end{document}",
        "\\bibliographystyle{plainnat}",
        "\\bibliography{references}",
    ):
        if req not in tex:
            errors.append(f"paper/main.tex missing required LaTeX directive: {req}")

    # 3. Balanced unescaped braces '{' and '}' (ignoring \\{, \\}, and % comments)
    brace_depth = 0
    for lineno, raw_line in enumerate(tex.splitlines(), start=1):
        # Strip unescaped % comments
        i = 0
        line_chars: list[str] = []
        while i < len(raw_line):
            ch = raw_line[i]
            if ch == "\\" and i + 1 < len(raw_line):
                line_chars.append(raw_line[i : i + 2])
                i += 2
                continue
            if ch == "%":
                break
            line_chars.append(ch)
            i += 1
        stripped = "".join(line_chars)
        j = 0
        while j < len(stripped):
            if stripped[j] == "\\" and j + 1 < len(stripped):
                j += 2
                continue
            if stripped[j] == "{":
                brace_depth += 1
            elif stripped[j] == "}":
                brace_depth -= 1
                if brace_depth < 0:
                    errors.append(f"Unmatched '}}' in paper/main.tex at line {lineno}")
                    brace_depth = 0
            j += 1
    if brace_depth != 0:
        errors.append(f"Unbalanced '{{'/'}}' braces in paper/main.tex (final depth={brace_depth})")

    # 4. Balanced \begin{env} / \end{env} pairs
    env_stack: list[str] = []
    for match in re.finditer(r"\\(begin|end)\{([A-Za-z0-9_*]+)\}", tex):
        kind, env_name = match.group(1), match.group(2)
        if kind == "begin":
            env_stack.append(env_name)
        else:
            if not env_stack or env_stack[-1] != env_name:
                errors.append(
                    f"Mismatched \\end{{{env_name}}} in paper/main.tex (stack={env_stack})"
                )
            else:
                env_stack.pop()
    if env_stack:
        errors.append(f"Unclosed LaTeX environments in paper/main.tex: {env_stack}")

    # 5. BibTeX keys resolution & underscore hygiene in notes
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    cited_keys: set[str] = set()
    for grp in re.findall(r"\\cite[pt]?\{([^}]+)\}", tex):
        for k in grp.split(","):
            if k.strip():
                cited_keys.add(k.strip())
    missing_bib = sorted(cited_keys - bib_keys)
    if missing_bib:
        errors.append(f"LaTeX \\cite keys missing from paper/references.bib: {missing_bib}")

    for ln in bib.splitlines():
        if "note" in ln and "=" in ln:
            stripped_url = re.sub(r"\\url\{[^}]+\}", "", ln)
            if re.search(r"(?<!\\)_", stripped_url):
                errors.append(f"Unescaped '_' in BibTeX note outside \\url{{}}: {ln.strip()}")

    # 6. Figure includes exist as non-empty PDFs
    fig_refs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    if len(fig_refs) < 3:
        errors.append(f"Expected >= 3 \\includegraphics in paper/main.tex, found {len(fig_refs)}")
    for rel in fig_refs:
        fpath = (ROOT / "paper" / rel).resolve()
        if not fpath.exists() or fpath.stat().st_size == 0:
            errors.append(f"Missing or empty LaTeX figure: {rel}")

    # 7. Hygiene & core numeric consistency against JSONs
    tex_lower = tex.lower()
    if "null result confirmed" in tex_lower:
        errors.append("Forbidden uncalibrated phrase 'null result confirmed' found in paper/main.tex")
    if "usage reset" in tex_lower or "5-hour" in tex_lower:
        errors.append("Private operational remark leaked into paper/main.tex")

    m3 = _load_json("results/m3_isolated_evaluation.json")
    remote = _load_json("results/m1_remote_reader_validation.json")
    audit = _load_json("runs/manifests/m3_frozen_checkpoint_audit.json")
    expected_tex_tokens = [
        f"{audit['persistent_arms']['C0_raw']['total_params']:,}",
        "852,985,920",
        f"{remote['bytes_read']:,}",
        f"{remote['total_file_bytes']:,}",
        f"{remote['byte_reduction_ratio']:,.2f}",
        "\\$0.00",
        "\\$14.2912",
        f"{m3['arms']['C0_raw']['datasets']['gsm8k_test']['metrics']['pass_at_1'] * 100:.1f}\\%",
        f"{m3['arms']['Qwen3.5-2B']['datasets']['gsm8k_test']['metrics']['pass_at_1'] * 100:.1f}\\%",
        f"{m3['arms']['T1']['datasets']['gsm8k_test']['metrics']['pass_at_1'] * 100:.1f}\\%",
        f"{m3['arms']['T2']['datasets']['gsm8k_test']['metrics']['pass_at_1'] * 100:.1f}\\%",
        f"{m3['arms']['C3']['datasets']['synthetic_ood_test']['metrics']['pass_at_1'] * 100:.1f}\\%",
        f"{m3['arms']['C3']['prose_perplexity']['perplexity']:.2f}",
        f"{m3['arms']['C0_raw']['prose_perplexity']['perplexity']:.2f}",
        f"{m3['holm_bonferroni']['T1']['gsm8k_test']['T1_vs_C1']:.3f}",
        f"{m3['holm_bonferroni']['T1']['gsm8k_test']['T1_vs_C4_weights']:.3f}",
        f"{m3['holm_bonferroni']['T2']['gsm8k_test']['T2_vs_C1']:.3f}",
        f"{m3['holm_bonferroni']['T2']['gsm8k_test']['T2_vs_C4_act']:.3f}",
    ]
    for tok in expected_tex_tokens:
        if tok not in tex:
            errors.append(f"Expected verified token missing from paper/main.tex: {tok!r}")

    return errors


def main() -> int:
    errs = verify_paper_artifacts() + verify_latex_manuscript()
    if errs:
        for e in errs:
            print(f"ERROR: {e}")
        return 1
    print("All Milestone M4 paper, LaTeX, figure, citation, and JSON verification checks PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
