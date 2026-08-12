# ClaimSight Eval Report (Slice 6)

- **mode:** `fake`
- **prompt:** `prompts/adjudicator_v1.md`
- **model:** `oracle-fake`
- **n_cases:** 50

## Metrics

| Metric | Value |
| --- | --- |
| Decision accuracy | 1.000 |
| Citation hallucination rate (post-guardrail) | 0.000 |
| Fraud-flag precision | 1.000 |
| Fraud-flag recall | 1.000 |
| Fraud TP / FP / FN / TN | 8 / 0 / 0 / 42 |

## Cases

| claim_id | gt | pred | match | halluc | fraud_gt | fraud_pred |
| --- | --- | --- | --- | --- | --- | --- |
| g001_collision_approve | approve | approve | Y | N | False | False |
| g002_collision_approve_multicite | approve | approve | Y | N | False | False |
| g003_glass_comp_approve | approve | approve | Y | N | False | False |
| g004_hail_comp_approve | approve | approve | Y | N | False | False |
| g005_animal_comp_approve | approve | approve | Y | N | False | False |
| g006_theft_comp_approve | approve | approve | Y | N | False | False |
| g007_vandalism_approve | approve | approve | Y | N | False | False |
| g008_collision_approve_empty_vision | approve | approve | Y | N | False | False |
| g009_collision_approve_null_vision | approve | approve | Y | N | False | False |
| g010_collision_approve_with_weather_ok | approve | approve | Y | N | False | False |
| g011_glass_only_approve | approve | approve | Y | N | False | False |
| g012_collision_approve_low_risk | approve | approve | Y | N | False | False |
| g013_intentional_deny | deny | deny | Y | N | False | False |
| g014_fraud_void_deny | deny | deny | Y | N | False | False |
| g015_wear_tear_deny | deny | deny | Y | N | False | False |
| g016_mechanical_deny | deny | deny | Y | N | False | False |
| g017_freezing_deny | deny | deny | Y | N | False | False |
| g018_road_tire_deny | deny | deny | Y | N | False | False |
| g019_intentional_fire_deny | deny | deny | Y | N | False | False |
| g020_intentional_vandal_self_deny | deny | deny | Y | N | False | False |
| g021_empty_rag_review | needs_review | needs_review | Y | N | False | False |
| g022_sources_failed_review | needs_review | needs_review | Y | N | False | False |
| g023_low_conf_policy_review | needs_review | needs_review | Y | N | False | False |
| g024_low_conf_limits_review | needs_review | needs_review | Y | N | False | False |
| g025_conflicting_story_review | needs_review | needs_review | Y | N | False | False |
| g026_missing_date_review | needs_review | needs_review | Y | N | False | False |
| g027_vin_decode_failed_review | needs_review | needs_review | Y | N | False | False |
| g028_ambiguous_glass_collision_review | needs_review | needs_review | Y | N | False | False |
| g029_high_estimate_review | needs_review | needs_review | Y | N | False | False |
| g030_no_narrative_detail_review | needs_review | needs_review | Y | N | False | False |
| g031_partial_rag_unclear_review | needs_review | needs_review | Y | N | False | False |
| g032_liability_third_party_review | needs_review | needs_review | Y | N | False | False |
| g033_recall_context_review | needs_review | needs_review | Y | N | False | False |
| g034_empty_vision_and_weak_docs_review | needs_review | needs_review | Y | N | False | False |
| g035_policy_id_null_review | needs_review | needs_review | Y | N | False | False |
| g036_weather_mismatch_review | needs_review | needs_review | Y | N | True | True |
| g037_staged_damage_review | needs_review | needs_review | Y | N | True | True |
| g038_inconsistent_claim_review | needs_review | needs_review | Y | N | True | True |
| g039_weather_mismatch_with_vision | needs_review | needs_review | Y | N | True | True |
| g040_staged_high_severity | needs_review | needs_review | Y | N | True | True |
| g041_inconsistent_plus_recall | needs_review | needs_review | Y | N | True | True |
| g042_weather_mismatch_deny_attempt | needs_review | needs_review | Y | N | True | True |
| g043_staged_empty_rag_review | needs_review | needs_review | Y | N | True | True |
| g044_cross_policy_cite_trap | approve | approve | Y | N | False | False |
| g045_other_clause_not_in_rag | approve | approve | Y | N | False | False |
| g046_multicite_approve_edge | approve | approve | Y | N | False | False |
| g047_deny_with_collision_also_retrieved | deny | deny | Y | N | False | False |
| g048_empty_detections_not_deny | approve | approve | Y | N | False | False |
| g049_sources_failed_not_negative | needs_review | needs_review | Y | N | False | False |
| g050_claims_procedure_only_review | needs_review | needs_review | Y | N | False | False |
