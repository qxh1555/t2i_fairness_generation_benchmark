# Metadata Design Notes

Each prompt item is not just a sentence. It is an evaluation unit.

## Core fields

- `prompt_id`: unique id, e.g. `nurse_A1`
- `module`: one of A/B/C/D/E/F
- `slice`: detailed subtype, e.g. `neutral_occupation`, `pair_occupation_contrast_ood`
- `target_occupation`: occupation under evaluation
- `prompt`: text-to-image prompt
- `expected_roles`: roles expected to appear in the image
- `evaluation_tasks`: what a VLM/classifier should judge
- `primary_metrics`: metrics aggregated from evaluation outputs
- `validity_filters`: filters before fairness statistics
- `distribution_evaluation`: whether to compute demographic distribution
- `binding_evaluation`: whether to compute role/gender binding correctness
- `side_effect_evaluation`: whether to compute off-target side effects

## Module-specific logic

A / Neutral:
- single target occupation
- no explicit gender/race/age
- compute demographic distribution after validity filtering

B / Group:
- multiple target-occupation people
- require at least two visible people/faces
- compute group-level demographic distribution

C1 / Common pair:
- realistic high-cooccurrence pair
- may have high role ambiguity
- useful for ecological validity, not strict role-binding proof

C2-C3 / Contrast OOD pair:
- visually distinguishable pair
- mark `is_ood_pair=true`
- useful for occupation-role binding and stereotype transfer diagnostics

D / Contextual trigger:
- no occupation word
- action/tool/workflow implies target occupation
- record confusable occupations

E / Role binding:
- explicit male/female
- do not evaluate distribution fairness
- evaluate gender preservation, occupation preservation, role swap, over-debias

F / Side effects:
- shared F1-F9 are fixed across all occupations
- optional pseudo-trigger is occupation-specific if available
- evaluate unexpected human/occupation generation and prompt alignment
