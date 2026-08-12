"""Build fixtures/golden/manifest.jsonl — synthetic Slice 6 golden set.

Synthetic data only (no real customer PII). Re-run to regenerate the manifest:

    python fixtures/golden/build_manifest.py
"""

from __future__ import annotations

from pathlib import Path

from agents.schemas import (
    Detection,
    DocumentOutput,
    LineItem,
    NHTSARecall,
    RAGOutput,
    RetrievedClause,
    RiskFlag,
    RiskOutput,
    VerifierOutput,
    VisionOutput,
    WeatherAtIncident,
)
from eval.schema import GoldenCase, GroundTruth, UpstreamSnapshot

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "manifest.jsonl"

POLICY = "fixtures/sample_policy.pdf"
ESTIMATE = "fixtures/sample_estimate.pdf"

CLAUSES = {
    "COL-001": (
        "Collision coverage pays for direct and accidental loss to your covered "
        "auto caused by collision with another object or by upset of the auto, "
        "subject to the deductible shown on the declarations page."
    ),
    "COL-002": (
        "Under collision coverage, we will pay the lesser of the actual cash value "
        "of the damaged property or the amount necessary to repair or replace it."
    ),
    "COM-001": (
        "Comprehensive coverage (other than collision) pays for loss to your covered "
        "auto caused by fire, theft, vandalism, falling objects, explosion, "
        "earthquake, windstorm, hail, water, flood, or contact with a bird or animal."
    ),
    "COM-002": (
        "Glass breakage is covered under comprehensive coverage unless it occurs as "
        "a result of a collision, in which case collision coverage applies if purchased."
    ),
    "LIA-001": (
        "Liability coverage pays damages for bodily injury or property damage for "
        "which any insured becomes legally responsible because of an auto accident."
    ),
    "EXC-001": (
        "We do not provide coverage for any insured who intentionally causes bodily "
        "injury or property damage. Intentional acts and fraud void coverage."
    ),
    "EXC-002": (
        "Coverage does not apply to damage due to wear and tear, freezing, mechanical "
        "or electrical breakdown, or road damage to tires."
    ),
    "CLM-001": (
        "In the event of an accident or loss, you must promptly notify us or our "
        "authorized agent."
    ),
    # Cross-policy ids — must never appear as legal cites for sample policy cases.
    "OTHER-COL-001": (
        "Collision coverage under POL-OTHER-0001 pays for front-end and rear-end "
        "collision damage to the insured vehicle after the applicable deductible."
    ),
}


def base_document(**overrides) -> DocumentOutput:
    data = {
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
            LineItem(description="Front bumper replacement", cost=850.0),
            LineItem(description="Headlight assembly", cost=420.5),
        ],
    }
    data.update(overrides)
    return DocumentOutput.model_validate(data)


def clean_extraction() -> dict:
    return {
        "confidences": {
            "policy_id": 0.99,
            "coverage_limits.collision": 0.95,
            "vin": 0.98,
            "incident_date": 0.97,
        },
        "low_confidence_fields": [],
        "min_confidence": 0.5,
    }


def rag_of(*clause_ids: str, score: float = 0.9) -> RAGOutput:
    return RAGOutput(
        retrieved_clauses=[
            RetrievedClause(
                clause_id=cid,
                text=CLAUSES[cid],
                similarity_score=score - 0.02 * i,
            )
            for i, cid in enumerate(clause_ids)
        ]
    )


def risk_clean() -> RiskOutput:
    return RiskOutput(flags=[], risk_score=0.05)


def risk_flag(flag_type: str, severity: str = "medium") -> RiskOutput:
    return RiskOutput(
        flags=[
            RiskFlag(
                flag_type=flag_type,
                rationale=f"Synthetic golden signal: {flag_type}",
                severity=severity,
            )
        ],
        risk_score=0.55 if severity == "medium" else 0.8,
    )


def verifiers_ok(**overrides) -> VerifierOutput:
    data = {
        "make": "HONDA",
        "model": "Accord",
        "model_year": 2003,
        "nhtsa_recalls": [],
        "nhtsa_complaints": [],
        "weather_at_incident": None,
        "sources_failed": [],
    }
    data.update(overrides)
    return VerifierOutput.model_validate(data)


def vision_damage() -> VisionOutput:
    return VisionOutput(
        detections=[
            Detection(
                label="bumper_dent",
                confidence=0.82,
                image_path="fixtures/images/synthetic/bumper.jpg",
            )
        ],
        severity_tier="moderate",
        severity_confidence=0.7,
        vqa_answers={"what_is_damaged": "front bumper"},
        low_confidence=False,
    )


def vision_empty() -> VisionOutput:
    return VisionOutput(
        detections=[],
        severity_tier="unknown",
        severity_confidence=0.2,
        vqa_answers={},
        low_confidence=True,
    )


def case(
    claim_id: str,
    *,
    narrative: str,
    decision: str,
    clause_ids: list[str],
    fraud_flag: bool,
    notes: str,
    rag: RAGOutput,
    risk: RiskOutput | None = None,
    document: DocumentOutput | None = None,
    extraction_meta: dict | None = None,
    vision: VisionOutput | None = None,
    verifiers: VerifierOutput | None = None,
    images: list[str] | None = None,
    incident_location: str | None = None,
) -> GoldenCase:
    return GoldenCase(
        claim_id=claim_id,
        narrative=narrative,
        policy_pdf=POLICY,
        estimate_pdf=ESTIMATE,
        images=images or [],
        incident_location=incident_location,
        upstream=UpstreamSnapshot(
            document_agent=document or base_document(),
            extraction_meta=extraction_meta if extraction_meta is not None else clean_extraction(),
            vision=vision,
            verifiers=verifiers or verifiers_ok(),
            rag=rag,
            risk=risk or risk_clean(),
        ),
        ground_truth=GroundTruth(
            decision=decision,  # type: ignore[arg-type]
            clause_ids=clause_ids,
            fraud_flag=fraud_flag,
        ),
        notes=notes,
    )


def build_cases() -> list[GoldenCase]:
    cases: list[GoldenCase] = []

    # --- Approve (~12) ---
    approve_specs = [
        ("g001_collision_approve", "Front-end collision damaged bumper and headlight.", ["COL-001"], rag_of("COL-001", "COL-002"), vision_damage()),
        ("g002_collision_approve_multicite", "Rear collision; bumper and trunk lid damaged.", ["COL-001", "COL-002"], rag_of("COL-001", "COL-002"), vision_damage()),
        ("g003_glass_comp_approve", "Windshield cracked from falling branch; no collision.", ["COM-001", "COM-002"], rag_of("COM-001", "COM-002"), None),
        ("g004_hail_comp_approve", "Hail dented the hood and roof panels overnight.", ["COM-001"], rag_of("COM-001", "COL-001"), None),
        ("g005_animal_comp_approve", "Hit a deer; comprehensive animal contact claimed.", ["COM-001"], rag_of("COM-001"), vision_damage()),
        ("g006_theft_comp_approve", "Vehicle stolen from driveway and recovered with damage.", ["COM-001"], rag_of("COM-001", "CLM-001"), None),
        ("g007_vandalism_approve", "Someone keyed the doors overnight in a parking lot.", ["COM-001"], rag_of("COM-001"), None),
        ("g008_collision_approve_empty_vision", "Minor parking-lot bump to rear bumper; no photos uploaded.", ["COL-001"], rag_of("COL-001"), vision_empty()),
        ("g009_collision_approve_null_vision", "Side-swipe on highway; photos not attached.", ["COL-001"], rag_of("COL-001"), None),
        ("g010_collision_approve_with_weather_ok", "Collision during light rain; weather matches narrative.", ["COL-001"], rag_of("COL-001"), vision_damage()),
        ("g011_glass_only_approve", "Chip in windshield from road debris; glass claim only.", ["COM-002"], rag_of("COM-002", "COM-001"), None),
        ("g012_collision_approve_low_risk", "Low-speed fender bender at stop sign.", ["COL-001"], rag_of("COL-001", "CLM-001"), vision_damage()),
    ]
    for cid, narrative, clauses, rag, vision in approve_specs:
        loc = "Washington, DC" if "weather" in cid else None
        ver = verifiers_ok(
            weather_at_incident=WeatherAtIncident(
                condition="Light Rain",
                precipitation_mm=2.0,
                had_storm_event=False,
            )
        ) if loc else verifiers_ok()
        cases.append(
            case(
                cid,
                narrative=narrative,
                decision="approve",
                clause_ids=clauses,
                fraud_flag=False,
                notes="Clear coverage; no material risk",
                rag=rag,
                vision=vision,
                verifiers=ver,
                incident_location=loc,
                images=["fixtures/images/synthetic/bumper.jpg"] if vision and vision.detections else [],
            )
        )

    # --- Deny (~8) ---
    deny_specs = [
        ("g013_intentional_deny", "I intentionally rammed the gate after an argument.", ["EXC-001"], rag_of("EXC-001", "COL-001")),
        ("g014_fraud_void_deny", "Insured admits fabricating the loss to collect payment.", ["EXC-001"], rag_of("EXC-001")),
        ("g015_wear_tear_deny", "Tires worn bald from age; requesting new tires under collision.", ["EXC-002"], rag_of("EXC-002", "COL-001")),
        ("g016_mechanical_deny", "Engine seized from neglected oil changes; claiming comprehensive.", ["EXC-002"], rag_of("EXC-002", "COM-001")),
        ("g017_freezing_deny", "Coolant system froze after owner left water in radiator.", ["EXC-002"], rag_of("EXC-002")),
        ("g018_road_tire_deny", "Normal road wear destroyed the tires; no accident occurred.", ["EXC-002"], rag_of("EXC-002", "CLM-001")),
        ("g019_intentional_fire_deny", "Owner set fire to the car to get out of the loan.", ["EXC-001"], rag_of("EXC-001", "COM-001")),
        ("g020_intentional_vandal_self_deny", "I smashed my own windshield after a fight and want glass coverage.", ["EXC-001"], rag_of("EXC-001", "COM-002")),
    ]
    for cid, narrative, clauses, rag in deny_specs:
        cases.append(
            case(
                cid,
                narrative=narrative,
                decision="deny",
                clause_ids=clauses,
                fraud_flag=False,
                notes="Exclusion evidence retrieved; deny supported",
                rag=rag,
            )
        )

    # --- Needs review: empty RAG / missing evidence / low confidence (~15) ---
    review_specs = [
        ("g021_empty_rag_review", "Collision claim but no clauses retrieved for this policy.", [], rag_of(), False, "Empty RAG → cannot approve/deny"),
        ("g022_sources_failed_review", "Storm damage claimed but weather API failed.", ["COM-001"], rag_of("COM-001"), False, "sources_failed is missing evidence"),
        ("g023_low_conf_policy_review", "Collision with unreliable policy_id extraction.", ["COL-001"], rag_of("COL-001"), False, "Critical low-confidence extraction"),
        ("g024_low_conf_limits_review", "Estimate present but coverage_limits.collision low confidence.", ["COL-001"], rag_of("COL-001"), False, "coverage_limits low confidence"),
        ("g025_conflicting_story_review", "Narrative says hail; estimate lists collision bumper only.", ["COL-001", "COM-001"], rag_of("COL-001", "COM-001"), False, "Conflicting coverage story"),
        ("g026_missing_date_review", "Loss date missing from documents; weather check impossible.", ["COL-001"], rag_of("COL-001"), False, "Missing incident date"),
        ("g027_vin_decode_failed_review", "VIN present but NHTSA decode failed.", ["COL-001"], rag_of("COL-001"), False, "VIN source failed"),
        ("g028_ambiguous_glass_collision_review", "Glass broke; unclear if collision or comprehensive.", ["COM-002", "COL-001"], rag_of("COM-002", "COL-001"), False, "Ambiguous glass vs collision"),
        ("g029_high_estimate_review", "Repair estimate far exceeds ACV guidance; needs adjuster.", ["COL-002"], rag_of("COL-002", "COL-001"), False, "High severity estimate"),
        ("g030_no_narrative_detail_review", "Please process claim.", ["COL-001"], rag_of("COL-001"), False, "Narrative too thin"),
        ("g031_partial_rag_unclear_review", "Claim mentions flood and collision together without clarity.", ["COM-001", "COL-001"], rag_of("COM-001"), False, "Mixed peril narrative"),
        ("g032_liability_third_party_review", "Third-party liability demand letter; coverage unclear.", ["LIA-001"], rag_of("LIA-001"), False, "Liability demand needs review"),
        ("g033_recall_context_review", "Collision plus open recall on airbags; not automatic deny.", ["COL-001"], rag_of("COL-001"), False, "Recall is context not proof"),
        ("g034_empty_vision_and_weak_docs_review", "No photos and sparse estimate line items.", ["COL-001"], rag_of("COL-001"), False, "Weak multimodal evidence"),
        ("g035_policy_id_null_review", "Documents failed to yield a policy_id.", ["COL-001"], rag_of("COL-001"), False, "Null policy_id extraction"),
    ]
    for cid, narrative, clauses, rag, fraud, notes in review_specs:
        document = base_document()
        extraction = clean_extraction()
        ver = verifiers_ok()
        risk = risk_clean()
        vision: VisionOutput | None = None

        if cid == "g022_sources_failed_review":
            ver = verifiers_ok(sources_failed=["nws_observations"])
            incident_location = "Miami, FL"
        else:
            incident_location = None

        if cid == "g023_low_conf_policy_review":
            extraction = {
                "confidences": {"policy_id": 0.2},
                "low_confidence_fields": ["policy_id"],
                "min_confidence": 0.5,
            }
        if cid == "g024_low_conf_limits_review":
            extraction = {
                "confidences": {"coverage_limits.collision": 0.1},
                "low_confidence_fields": ["coverage_limits.collision"],
                "min_confidence": 0.5,
            }
        if cid == "g026_missing_date_review":
            document = base_document(incident_date=None)
        if cid == "g027_vin_decode_failed_review":
            ver = verifiers_ok(make=None, model=None, model_year=None, sources_failed=["nhtsa_vin"])
        if cid == "g033_recall_context_review":
            ver = verifiers_ok(
                nhtsa_recalls=[
                    NHTSARecall(
                        campaign_number="22V-999",
                        component="AIR BAGS",
                        summary="Synthetic recall for golden set",
                    )
                ]
            )
        if cid == "g034_empty_vision_and_weak_docs_review":
            vision = vision_empty()
            document = base_document(line_items=[])
        if cid == "g035_policy_id_null_review":
            document = base_document(policy_id=None)
            extraction = {
                "confidences": {"policy_id": 0.0},
                "low_confidence_fields": ["policy_id"],
                "min_confidence": 0.5,
            }

        # For oracle consistency: GT needs_review. Oracle proposes needs_review.
        # Material-risk cases are separate below.
        cases.append(
            case(
                cid,
                narrative=narrative,
                decision="needs_review",
                clause_ids=clauses,
                fraud_flag=fraud,
                notes=notes,
                rag=rag,
                risk=risk,
                document=document,
                extraction_meta=extraction,
                vision=vision,
                verifiers=ver,
                incident_location=incident_location,
            )
        )

    # --- Fraud-signal / material risk (~8) — GT needs_review + fraud_flag true ---
    fraud_specs = [
        ("g036_weather_mismatch_review", "Claimed hail storm destroyed roof; weather shows clear skies.", ["COM-001"], rag_of("COM-001"), "weather mismatch"),
        ("g037_staged_damage_review", "Narrative reads like possible staged damage after prior claim.", ["COL-001"], rag_of("COL-001"), "possible staged damage"),
        ("g038_inconsistent_claim_review", "Dates, VIN, and location conflict across documents.", ["COL-001"], rag_of("COL-001"), "inconsistent claim"),
        ("g039_weather_mismatch_with_vision", "Flood claim but dry weather; photos show water line.", ["COM-001"], rag_of("COM-001"), "weather mismatch"),
        ("g040_staged_high_severity", "Brand-new damage pattern suggests staging.", ["COL-001"], rag_of("COL-001", "EXC-001"), "possible staged damage"),
        ("g041_inconsistent_plus_recall", "Story inconsistent; also unrelated open recall.", ["COL-001"], rag_of("COL-001"), "inconsistent claim"),
        ("g042_weather_mismatch_deny_attempt", "Insured wants deny-level fraud call; system should review.", ["EXC-001"], rag_of("EXC-001", "COM-001"), "weather mismatch"),
        ("g043_staged_empty_rag_review", "Staged-damage signal with empty RAG → review.", [], rag_of(), "possible staged damage"),
    ]
    for cid, narrative, clauses, rag, flag_type in fraud_specs:
        cases.append(
            case(
                cid,
                narrative=narrative,
                decision="needs_review",
                clause_ids=clauses,
                fraud_flag=True,
                notes=f"Material risk flag: {flag_type}",
                rag=rag,
                risk=risk_flag(flag_type, severity="high" if "staged" in flag_type else "medium"),
                vision=vision_damage() if "vision" in cid else None,
                incident_location="Dallas, TX" if "weather" in flag_type else None,
                verifiers=verifiers_ok(
                    weather_at_incident=WeatherAtIncident(
                        condition="Clear",
                        precipitation_mm=0.0,
                        had_storm_event=False,
                    )
                )
                if "weather" in flag_type
                else verifiers_ok(),
            )
        )

    # --- Edge (~7) ---
    cases.append(
        case(
            "g044_cross_policy_cite_trap",
            narrative="Collision claim; RAG correctly scoped — OTHER-* must not be cited.",
            decision="approve",
            clause_ids=["COL-001"],
            fraud_flag=False,
            notes="Cross-policy hygiene: only sample-policy clauses retrieved",
            rag=rag_of("COL-001", "COL-002"),
            vision=vision_damage(),
        )
    )
    cases.append(
        case(
            "g045_other_clause_not_in_rag",
            narrative="If model invents OTHER-COL-001 it must be rejected by guardrails.",
            decision="approve",
            clause_ids=["COL-001"],
            fraud_flag=False,
            notes="Hallucination trap for live mode; oracle cites COL-001 only",
            rag=rag_of("COL-001"),
        )
    )
    cases.append(
        case(
            "g046_multicite_approve_edge",
            narrative="Multi-panel collision; cite collision repair valuation clauses.",
            decision="approve",
            clause_ids=["COL-001", "COL-002"],
            fraud_flag=False,
            notes="Multi-cite happy path",
            rag=rag_of("COL-001", "COL-002", "CLM-001"),
            vision=vision_damage(),
        )
    )
    cases.append(
        case(
            "g047_deny_with_collision_also_retrieved",
            narrative="Intentional act; collision clauses retrieved but exclusion controls.",
            decision="deny",
            clause_ids=["EXC-001"],
            fraud_flag=False,
            notes="Deny cites exclusion even when collision also retrieved",
            rag=rag_of("EXC-001", "COL-001", "COL-002"),
        )
    )
    cases.append(
        case(
            "g048_empty_detections_not_deny",
            narrative="Damage claimed; vision ran but detections empty — not automatic deny.",
            decision="approve",
            clause_ids=["COL-001"],
            fraud_flag=False,
            notes="Empty detections = no signal; approve still allowed with RAG cite",
            rag=rag_of("COL-001"),
            vision=vision_empty(),
        )
    )
    cases.append(
        case(
            "g049_sources_failed_not_negative",
            narrative="Hail claimed; weather source failed — missing evidence, not no-storm.",
            decision="needs_review",
            clause_ids=["COM-001"],
            fraud_flag=False,
            notes="sources_failed must not become deny",
            rag=rag_of("COM-001"),
            verifiers=verifiers_ok(sources_failed=["nws_observations", "nominatim"]),
            incident_location="Denver, CO",
        )
    )
    cases.append(
        case(
            "g050_claims_procedure_only_review",
            narrative="Late notice possible; only claims-procedure clauses retrieved.",
            decision="needs_review",
            clause_ids=["CLM-001"],
            fraud_flag=False,
            notes="Procedure-only RAG is weak for approve/deny",
            rag=rag_of("CLM-001"),
        )
    )

    assert len(cases) == 50, f"expected 50 golden cases, got {len(cases)}"
    ids = [c.claim_id for c in cases]
    assert len(ids) == len(set(ids)), "duplicate claim_id in golden set"
    return cases


def main() -> None:
    cases = build_cases()
    lines = [c.model_dump_json() for c in cases]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
