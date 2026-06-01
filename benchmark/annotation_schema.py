from __future__ import annotations

import json
from typing import Any, Dict, List


ANNOTATION_TOP_LEVEL_KEYS = {
    "annotation_id",
    "prompt_id",
    "image_id",
    "image_path",
    "prompt",
    "module",
    "slice",
    "target_occupation",
    "eval_task",
    "status",
    "expected",
    "annotations",
    "required_annotations",
    "metrics_enabled",
    "quality_gate",
    "evaluator_outputs",
}


def validate_annotation_record(record: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    missing_top = sorted(ANNOTATION_TOP_LEVEL_KEYS - set(record))
    if missing_top:
        errors.append(f"missing annotation top-level keys: {missing_top}")
        return errors

    status = record.get("status")
    annotations = record.get("annotations")
    expected = record.get("expected")
    if not isinstance(status, dict):
        errors.append("status must be an object")
    if not isinstance(annotations, dict):
        errors.append("annotations must be an object")
        return errors
    if not isinstance(expected, dict):
        errors.append("expected must be an object")

    required_names = [ann.get("name") for ann in record.get("required_annotations", []) if ann.get("required")]
    if not required_names:
        required_names = _fallback_required_names(record)

    missing_annotations = [name for name in required_names if name not in annotations]
    if missing_annotations:
        errors.append(f"missing required annotations: {missing_annotations}")

    eval_task = record.get("eval_task")
    if eval_task == "group_occupation_fairness":
        persons = annotations.get("persons")
        if not isinstance(persons, list):
            errors.append("group_occupation_fairness requires annotations.persons list")

    if eval_task in {"neutral_pair_role_fairness", "explicit_pair_role_gender_binding"}:
        roles = annotations.get("roles")
        if not isinstance(roles, list) or len(roles) != 2:
            errors.append(f"{eval_task} requires exactly two annotations.roles")

    if eval_task == "contextual_trigger_bias":
        if expected.get("explicit_occupation_word_present") is not False:
            errors.append("contextual_trigger_bias expected.explicit_occupation_word_present must be false")

    if eval_task == "irrelevant_side_effect":
        forbidden_metrics = {
            "gender_distribution_fairness",
            "race_distribution",
            "age_distribution",
            "spurious_association",
            "explicit_gender_accuracy",
            "role_binding_accuracy",
        }
        enabled = set(record.get("metrics_enabled") or [])
        forbidden = sorted(enabled & forbidden_metrics)
        if forbidden:
            errors.append(f"irrelevant_side_effect has forbidden metrics: {forbidden}")

    try:
        json.dumps(record, ensure_ascii=False)
    except TypeError as exc:
        errors.append(f"annotation record is not JSON serializable: {exc}")

    return errors


def _fallback_required_names(record: Dict[str, Any]) -> List[str]:
    plan = record.get("eval_plan") or {}
    return [
        ann.get("name")
        for ann in plan.get("required_annotations", [])
        if isinstance(ann, dict) and ann.get("required") and ann.get("name")
    ]
