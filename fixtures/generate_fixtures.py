"""Generate synthetic sample policy and estimate PDFs for ClaimSight slice 1.

Synthetic data only — no real customer PII. Re-run this script to regenerate
fixtures/sample_policy.pdf, fixtures/sample_estimate.pdf, and fixtures/expected.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

FIXTURES_DIR = Path(__file__).resolve().parent

EXPECTED = {
    "policy_id": "POL-2024-0098213",
    "coverage_limits": {
        "collision": 50000.0,
        "comprehensive": 25000.0,
        "liability": 100000.0,
    },
    "deductible": 1000.0,
    "vin": "1HGCM82633A004352",
    "incident_date": "2024-03-14",
    "line_items": [
        {"description": "Front bumper replacement", "cost": 850.0},
        {"description": "Headlight assembly", "cost": 420.5},
        {"description": "Labor - body repair", "cost": 682.5},
        {"description": "Paint and materials", "cost": 310.0},
    ],
}


def build_policy_pdf(path: Path) -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "PolicyTitle",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=12,
    )
    body = styles["Normal"]

    doc = SimpleDocTemplate(
        str(path),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    story = [
        Paragraph("ClaimSight Auto Insurance — Declarations Page", title),
        Paragraph(f"Policy ID: {EXPECTED['policy_id']}", body),
        Spacer(1, 0.2 * inch),
        Paragraph("Named Insured: Alex Rivera", body),
        Paragraph("Vehicle Identification Number (VIN): 1HGCM82633A004352", body),
        Paragraph("Incident Date: March 14, 2024", body),
        Spacer(1, 0.3 * inch),
        Paragraph("Coverage Limits", styles["Heading2"]),
    ]

    coverage_data = [
        ["Coverage", "Limit"],
        ["Collision", "$50,000"],
        ["Comprehensive", "$25,000"],
        ["Liability", "$100,000"],
    ]
    coverage_table = Table(coverage_data, colWidths=[3.5 * inch, 2 * inch])
    coverage_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(coverage_table)
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Deductible: $1,000.00", body))
    story.append(
        Paragraph(
            "This declarations page summarizes coverage for the policy period. "
            "All figures are synthetic and for demo purposes only.",
            body,
        )
    )
    doc.build(story)


def build_estimate_pdf(path: Path) -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "EstimateTitle",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=12,
    )
    body = styles["Normal"]

    doc = SimpleDocTemplate(
        str(path),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    story = [
        Paragraph("Repair Estimate — Metro Auto Body", title),
        Paragraph(f"Related Policy ID: {EXPECTED['policy_id']}", body),
        Paragraph(f"VIN: {EXPECTED['vin']}", body),
        Paragraph(f"Incident Date: {EXPECTED['incident_date']}", body),
        Spacer(1, 0.3 * inch),
        Paragraph("Line Items", styles["Heading2"]),
    ]

    table_data = [["Description", "Cost"]]
    for item in EXPECTED["line_items"]:
        table_data.append([item["description"], f"${item['cost']:,.2f}"])

    total = sum(item["cost"] for item in EXPECTED["line_items"])
    table_data.append(["Total", f"${total:,.2f}"])

    estimate_table = Table(table_data, colWidths=[4.5 * inch, 1.5 * inch])
    estimate_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.whitesmoke, colors.lightgrey]),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(estimate_table)
    doc.build(story)


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    policy_path = FIXTURES_DIR / "sample_policy.pdf"
    estimate_path = FIXTURES_DIR / "sample_estimate.pdf"
    expected_path = FIXTURES_DIR / "expected.json"

    build_policy_pdf(policy_path)
    build_estimate_pdf(estimate_path)
    expected_path.write_text(json.dumps(EXPECTED, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {policy_path}")
    print(f"Wrote {estimate_path}")
    print(f"Wrote {expected_path}")


if __name__ == "__main__":
    main()
