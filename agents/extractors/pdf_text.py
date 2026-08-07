"""PDF text / table helpers using pdfplumber.

word_boxes are normalized to the 0..1000 coordinate space expected by
LayoutLM DocVQA. Table extraction for estimate line items is a pragmatic
placeholder for a proper Table Question Answering model in a later task.
"""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

from agents.schemas import LineItem

WordBox = tuple[str, list[int]]


def _normalize_box(
    x0: float, top: float, x1: float, bottom: float, page_width: float, page_height: float
) -> list[int]:
    return [
        int(1000 * (x0 / page_width)),
        int(1000 * (top / page_height)),
        int(1000 * (x1 / page_width)),
        int(1000 * (bottom / page_height)),
    ]


def extract_word_boxes(pdf_path: str | Path) -> list[list[WordBox]]:
    """Return per-page lists of (word, [x0,y0,x1,y1]) in LayoutLM 0..1000 space."""
    pages: list[list[WordBox]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            width = float(page.width) or 1.0
            height = float(page.height) or 1.0
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
            page_boxes: list[WordBox] = []
            for word in words:
                text = (word.get("text") or "").strip()
                if not text:
                    continue
                box = _normalize_box(
                    float(word["x0"]),
                    float(word["top"]),
                    float(word["x1"]),
                    float(word["bottom"]),
                    width,
                    height,
                )
                page_boxes.append((text, box))
            pages.append(page_boxes)
    return pages


_CURRENCY_RE = re.compile(r"[^0-9.\-]")


def _parse_cost(raw: str | None) -> float | None:
    if raw is None:
        return None
    cleaned = _CURRENCY_RE.sub("", raw.strip())
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_line_items(pdf_path: str | Path) -> list[LineItem]:
    """Extract estimate line items via pdfplumber table extraction.

    NOTE: This is a placeholder for a proper Table Question Answering model
    in a later task; do not over-engineer this part now.
    """
    items: list[LineItem] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables() or []
            for table in tables:
                if not table:
                    continue
                # Skip header row if it looks like Description / Cost.
                start = 0
                header = [((c or "").strip().lower()) for c in table[0]]
                if header and ("description" in header[0] or "cost" in (header[-1] if header else "")):
                    start = 1
                for row in table[start:]:
                    if not row or len(row) < 2:
                        continue
                    description = (row[0] or "").strip()
                    cost_raw = (row[-1] or "").strip()
                    if not description:
                        continue
                    if description.lower() == "total":
                        continue
                    cost = _parse_cost(cost_raw)
                    if cost is None:
                        continue
                    items.append(LineItem(description=description, cost=cost))
    return items
