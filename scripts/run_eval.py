"""Run Slice 6/7 golden eval (Adjudicator + guardrails on canned upstream).

Examples:
  python scripts/run_eval.py --mode fake
  python scripts/run_eval.py --mode fake --gate
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
from eval.gate import check_gates, load_baseline
from eval.report import write_report
from eval.runner import load_manifest, run_eval


def main() -> int:
    parser = argparse.ArgumentParser(description="ClaimSight golden eval + optional CI gate")
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
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Fail (exit 1) if hallucination or accuracy gates trip vs baseline",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "eval" / "reports" / "baseline_fake.json",
        help="Checked-in EvalReport JSON used when --gate is set",
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

    if args.gate:
        baseline = load_baseline(args.baseline)
        failures = check_gates(m, baseline)
        if failures:
            print("GATE FAILED:", file=sys.stderr)
            for reason in failures:
                print(f"  - {reason}", file=sys.stderr)
            return 1
        print(
            f"GATE PASSED vs {args.baseline} "
            f"(hallucination_rate={m.hallucination_rate:.3f}, "
            f"decision_accuracy={m.decision_accuracy:.3f})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
