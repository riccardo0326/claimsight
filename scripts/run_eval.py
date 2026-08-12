"""Run Slice 6 golden eval (Adjudicator + guardrails on canned upstream).

Examples:
  python scripts/run_eval.py --mode fake
  python scripts/run_eval.py --mode live
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.config import get_settings
from eval.report import write_report
from eval.runner import load_manifest, run_eval


def main() -> int:
    parser = argparse.ArgumentParser(description="ClaimSight Slice 6 golden eval")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "fixtures" / "golden" / "manifest.jsonl",
        help="Path to golden JSONL manifest",
    )
    parser.add_argument(
        "--mode",
        choices=("fake", "live"),
        default="fake",
        help="fake = oracle stub LLM; live = OpenAI via OPENAI_API_KEY",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "eval" / "reports",
        help="Directory for latest.json / latest.md",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of cases (smoke)",
    )
    args = parser.parse_args()

    cases = load_manifest(args.manifest)
    if args.limit is not None:
        cases = cases[: max(0, args.limit)]

    settings = get_settings()
    model = settings.adjudicator_model if args.mode == "live" else "oracle-fake"
    report = run_eval(cases, mode=args.mode, model=model)
    json_path, md_path = write_report(report, out_dir=args.out_dir)

    m = report.metrics
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        f"n={m.n_cases} decision_accuracy={m.decision_accuracy:.3f} "
        f"hallucination_rate={m.hallucination_rate:.3f} "
        f"fraud_P={m.fraud_precision} fraud_R={m.fraud_recall}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
