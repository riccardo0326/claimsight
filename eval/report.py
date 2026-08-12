"""Serialize eval reports to JSON and Markdown."""

from __future__ import annotations

from pathlib import Path

from eval.schema import EvalReport


def report_to_markdown(report: EvalReport) -> str:
    m = report.metrics
    prec = "n/a" if m.fraud_precision is None else f"{m.fraud_precision:.3f}"
    rec = "n/a" if m.fraud_recall is None else f"{m.fraud_recall:.3f}"
    lines = [
        "# ClaimSight Eval Report (Slice 6)",
        "",
        f"- **mode:** `{report.mode}`",
        f"- **prompt:** `{report.prompt_version}`",
        f"- **model:** `{report.model or 'n/a'}`",
        f"- **n_cases:** {m.n_cases}",
        "",
        "## Metrics",
        "",
        f"| Metric | Value |",
        f"| --- | --- |",
        f"| Decision accuracy | {m.decision_accuracy:.3f} |",
        f"| Citation hallucination rate (post-guardrail) | {m.hallucination_rate:.3f} |",
        f"| Fraud-flag precision | {prec} |",
        f"| Fraud-flag recall | {rec} |",
        f"| Fraud TP / FP / FN / TN | {m.fraud_true_positives} / "
        f"{m.fraud_false_positives} / {m.fraud_false_negatives} / "
        f"{m.fraud_true_negatives} |",
        "",
        "## Cases",
        "",
        "| claim_id | gt | pred | match | halluc | fraud_gt | fraud_pred |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in report.cases:
        lines.append(
            f"| {c.claim_id} | {c.ground_truth_decision} | {c.predicted_decision} | "
            f"{'Y' if c.decision_match else 'N'} | "
            f"{'Y' if c.hallucinated else 'N'} | "
            f"{c.ground_truth_fraud_flag} | {c.predicted_fraud_flag} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(
    report: EvalReport,
    *,
    out_dir: Path | str,
    basename: str = "latest",
) -> tuple[Path, Path]:
    """Write report JSON + Markdown; return (json_path, md_path)."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{basename}.json"
    md_path = directory / f"{basename}.md"
    json_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    md_path.write_text(report_to_markdown(report), encoding="utf-8")
    return json_path, md_path
