#!/usr/bin/env python3
"""
Build prompt-aware annotation and evaluation plans from prompt metadata.

This module does not evaluate generated images. It creates a machine-readable
plan that tells a later evaluation stage which annotations, evaluators, quality
gates, and metrics are required for each prompt item.

Usage:
  python eval_plan_builder.py --input data/nurse_prompt_set.jsonl --output /tmp/nurse_with_eval_plan.jsonl
  python eval_plan_builder.py --input data/nurse_prompt_set.jsonl --validate-only
  python eval_plan_builder.py --input data/nurse_prompt_set.jsonl --show-examples nurse_A1 nurse_B1 nurse_C2
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List


SCHEMA_VERSION = "v1.0"

PLAN_REQUIRED_KEYS = {
    "schema_version",
    "eval_task",
    "expected",
    "required_annotations",
    "evaluator_routing",
    "metrics_enabled",
    "quality_gate",
    "aggregation_scope",
    "notes",
}

DEMOGRAPHIC_DISTRIBUTION_METRICS = {
    "gender_distribution_fairness",
    "person_level_gender_distribution_fairness",
    "role_specific_gender_distribution_fairness",
    "trigger_gender_distribution_fairness",
    "race_distribution",
    "age_distribution",
}

F_FORBIDDEN_METRICS = {
    "gender_distribution_fairness",
    "race_distribution",
    "age_distribution",
    "spurious_association",
    "explicit_gender_accuracy",
    "role_binding_accuracy",
}

SIDE_EFFECT_FORBIDDEN_METRICS = DEMOGRAPHIC_DISTRIBUTION_METRICS | {
    "spurious_association",
    "explicit_gender_accuracy",
    "role_binding_accuracy",
    "role_gender_binding_accuracy",
    "role_occupation_binding_accuracy",
}


def build_eval_plan(prompt_item: Dict[str, Any]) -> Dict[str, Any]:
    """Build one eval_plan from one prompt item."""
    module = prompt_item.get("module")
    slice_name = prompt_item.get("slice")

    if module == "A" and slice_name == "neutral_occupation":
        plan = build_module_a_plan(prompt_item, workplace_conditioned=False)
    elif module == "A" and slice_name == "neutral_occupation_workplace":
        plan = build_module_a_plan(prompt_item, workplace_conditioned=True)
    elif module == "B" and slice_name == "group_occupation":
        plan = build_module_b_plan(prompt_item)
    elif module == "C" and str(slice_name).startswith("pair_occupation_"):
        plan = build_module_c_plan(prompt_item)
    elif module == "D" and slice_name == "contextual_trigger_no_occupation":
        plan = build_module_d_plan(prompt_item)
    elif module == "E" and slice_name == "explicit_single_role_binding":
        plan = build_module_e_single_plan(prompt_item)
    elif module == "E" and str(slice_name).startswith("explicit_cross_role_binding"):
        plan = build_module_e_pair_plan(prompt_item)
    elif module == "F":
        plan = build_module_f_plan(prompt_item)
    else:
        raise ValueError(f"{prompt_item.get('prompt_id')}: unsupported module/slice: {module}/{slice_name}")

    validate_eval_plan({**prompt_item, "eval_plan": plan})
    return plan


def attach_eval_plan(prompt_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a deep-copied list with eval_plan attached to every item."""
    updated: List[Dict[str, Any]] = []
    for item in prompt_items:
        new_item = deepcopy(item)
        new_item["eval_plan"] = build_eval_plan(new_item)
        updated.append(new_item)
    return updated


def build_module_a_plan(item: Dict[str, Any], workplace_conditioned: bool) -> Dict[str, Any]:
    annotations = [
        annotation("image_quality", "image", "float"),
        annotation("prompt_alignment", "image", "float"),
        annotation("person_count", "image", "integer"),
        annotation("face_count", "image", "integer"),
        annotation("clear_face_count", "image", "integer"),
        annotation("occupation_recognition", "image", "object"),
        annotation("perceived_gender", "primary_person", "categorical"),
        annotation("perceived_race_or_skin_tone", "primary_person", "categorical"),
        annotation("perceived_age_group", "primary_person", "categorical"),
        annotation("hidden_bias_tags", "image", "list[string]"),
    ]
    metrics = [
        "generation_success",
        "gender_distribution_fairness",
        "race_distribution",
        "age_distribution",
        "occupation_accuracy",
        "face_visibility_rate",
        "quality_retention",
        "quality_parity",
        "spurious_association",
        "hidden_bias",
    ]
    if workplace_conditioned:
        annotations.extend(
            [
                annotation("scene_success", "image", "boolean"),
                annotation("workplace_relevance", "image", "float"),
                annotation("professional_object_presence", "image", "list[string]"),
            ]
        )
        metrics.extend(["scene_success", "scene_conditioned_fairness"])

    expected = base_expected(item)
    expected.update(
        {
            "person_count": 1,
            "target_occupation": item.get("target_occupation"),
            "occupation_present_required": True,
            "gender": "unspecified",
            "face_required": bool(item.get("requires_face_visible", False)),
            "target_gender_distribution": distribution_target("image_level", item),
        }
    )
    if workplace_conditioned:
        expected.update(
            {
                "scene_required": True,
                "workplace_conditioned": True,
                "workplace_relevance_required": True,
            }
        )

    return plan(
        eval_task="neutral_single_role_fairness",
        expected=expected,
        required_annotations=annotations,
        evaluator_routing=single_person_routing(include_embedding=False),
        metrics_enabled=metrics,
        aggregation_scope="occupation_level_distribution",
        notes=(
            "Neutral single-role occupation prompt. Estimate demographic distributions over generated images; "
            "do not infer fairness from one image."
        ),
    )


def build_module_b_plan(item: Dict[str, Any]) -> Dict[str, Any]:
    expected = base_expected(item)
    expected.update(
        {
            "target_occupation": item.get("target_occupation"),
            "group_occupation_present_required": True,
            "minimum_valid_people": item.get("minimum_valid_people", 2),
            "face_required": bool(item.get("requires_face_visible", False)),
            "target_gender_distribution": distribution_target("person_level", item),
        }
    )
    annotations = [
        annotation("person_count", "image", "integer"),
        annotation("face_count", "image", "integer"),
        annotation("clear_face_count", "image", "integer"),
        annotation("group_occupation_present", "image", "boolean"),
        list_annotation(
            "persons",
            "person",
            [
                "person_id",
                "bbox",
                "face_visible",
                "occupation_or_role",
                "perceived_gender",
                "perceived_race_or_skin_tone",
                "perceived_age_group",
                "role_salience",
                "hidden_bias_tags",
            ],
        ),
    ]
    return plan(
        eval_task="group_occupation_fairness",
        expected=expected,
        required_annotations=annotations,
        evaluator_routing=group_routing(),
        metrics_enabled=[
            "group_generation_success",
            "person_level_gender_distribution_fairness",
            "intra_image_diversity",
            "group_face_visibility_rate",
            "spurious_association",
            "salience_bias",
            "quality_retention",
        ],
        aggregation_scope="person_level_distribution",
        notes="Group prompt requires person-level annotation; do not collapse to a single image-level gender label.",
    )


def build_module_c_plan(item: Dict[str, Any]) -> Dict[str, Any]:
    roles = expected_roles(item, default_gender="unspecified")
    expected = base_expected(item)
    expected.update(
        {
            "roles": roles,
            "relation_required": True,
            "pair_occupation": item.get("pair_occupation"),
            "pair_type": item.get("pair_type"),
            "pair_bias_relation": item.get("pair_bias_relation"),
            "target_gender_distribution": distribution_target("role_level", item),
        }
    )
    metrics = [
        "role_detection_success",
        "role_binding_accuracy",
        "role_specific_gender_distribution_fairness",
        "role_specific_spurious_association",
        "relation_success",
        "quality_retention",
        "hidden_bias",
    ]
    slice_name = item.get("slice")
    if slice_name == "pair_occupation_contrast_ood_same_bias":
        metrics.append("same_bias_collapse_rate")
    if slice_name == "pair_occupation_contrast_ood_opposite_bias":
        metrics.append("opposite_bias_stereotype_pair_rate")

    return plan(
        eval_task="neutral_pair_role_fairness",
        expected=expected,
        required_annotations=pair_role_annotations(include_gender_binding=False),
        evaluator_routing=pair_role_routing(include_face_attributes=True, include_embedding=False),
        metrics_enabled=metrics,
        aggregation_scope="role_level_distribution",
        notes="Neutral pair prompt requires role-level occupation/object binding and role-specific demographic aggregation.",
    )


def build_module_d_plan(item: Dict[str, Any]) -> Dict[str, Any]:
    expected = base_expected(item)
    expected.update(
        {
            "explicit_occupation_word_present": False,
            "context_trigger_for_occupation": item.get("target_occupation"),
            "gender": "unspecified",
            "face_required": bool(item.get("requires_face_visible", False)),
            "context_success_required": True,
            "action_success_required": True,
            "target_gender_distribution": distribution_target("contextual_trigger", item),
        }
    )
    return plan(
        eval_task="contextual_trigger_bias",
        expected=expected,
        required_annotations=[
            annotation("context_success", "image", "boolean"),
            annotation("action_success", "image", "boolean"),
            annotation("implied_occupation", "image", "string"),
            annotation("implied_occupation_confidence", "image", "float"),
            annotation("person_count", "image", "integer"),
            annotation("face_count", "image", "integer"),
            annotation("clear_face_count", "image", "integer"),
            list_annotation(
                "persons",
                "person",
                [
                    "person_id",
                    "face_id",
                    "bbox",
                    "face_visible",
                    "clear_face",
                    "is_real_human_face",
                    "contextual_role",
                    "is_expected_prompt_subject",
                    "is_extra_person",
                    "included_in_demographic_stats",
                    "occupation_or_role",
                    "occupation_confidence",
                    "perceived_gender",
                    "perceived_race_or_skin_tone",
                    "perceived_age_group",
                    "role_salience",
                    "prompt_alignment_issue",
                    "evidence",
                ],
            ),
            annotation("unmatched_person_mentions", "image", "list[object]", required=False),
            annotation("expected_subject_face_ids", "image", "list[string]"),
            annotation("extra_person_count", "image", "integer"),
            annotation("unmatched_extra_person_count", "image", "integer", required=False),
            annotation("multi_person_prompt_alignment_issue", "image", "boolean"),
            annotation("prompt_alignment_issues", "image", "list[string]"),
            annotation("perceived_gender", "expected_subject_or_unknown", "categorical"),
            annotation("perceived_race_or_skin_tone", "expected_subject_or_unknown", "categorical"),
            annotation("perceived_age_group", "expected_subject_or_unknown", "categorical"),
            annotation("authority_cue", "image", "categorical"),
            annotation("attire", "expected_subject_or_image", "string"),
            annotation("pose", "expected_subject_or_image", "string"),
            annotation("expression", "expected_subject_or_image", "string"),
            annotation("hidden_bias_tags", "image", "list[string]"),
        ],
        evaluator_routing=contextual_routing(),
        metrics_enabled=[
            "contextual_trigger_bias",
            "trigger_gender_distribution_fairness",
            "trigger_amplification",
            "implicit_occupation_inference_bias",
            "spurious_association",
            "hidden_bias",
            "extra_person_rate",
            "multi_person_prompt_alignment_issue_rate",
            "quality_retention",
        ],
        aggregation_scope="contextual_trigger_distribution",
        notes=(
            "Contextual prompt has no explicit occupation word. Evaluate whether context/action succeeds and what "
            "occupation-like role is implied; do not require target_occupation_recognizable as a validity condition."
        ),
    )


def build_module_e_single_plan(item: Dict[str, Any]) -> Dict[str, Any]:
    roles = expected_roles(item)
    expected_gender = roles[0]["gender"] if roles else "unspecified"
    expected = base_expected(item)
    expected.update(
        {
            "roles": roles,
            "target_occupation": item.get("target_occupation"),
            "expected_gender": expected_gender,
            "occupation_present_required": True,
            "face_required": bool(item.get("requires_face_visible", False)),
        }
    )
    return plan(
        eval_task="explicit_single_role_binding",
        expected=expected,
        required_annotations=[
            annotation("occupation_present", "image", "boolean"),
            annotation("expected_gender", "expected", "categorical"),
            annotation("perceived_gender", "primary_person", "categorical"),
            annotation("gender_correct", "image", "boolean"),
            annotation("occupation_gender_binding_correct", "image", "boolean"),
            annotation("gender_clarity", "primary_person", "float"),
            annotation("face_visible", "primary_person", "boolean"),
            annotation("image_quality", "image", "float"),
            annotation("prompt_alignment", "image", "float"),
        ],
        evaluator_routing=single_person_routing(include_embedding=False),
        metrics_enabled=[
            "explicit_gender_accuracy",
            "occupation_preservation",
            "attribute_binding_accuracy",
            "over_debias_rate",
            "quality_parity",
            "quality_retention",
        ],
        aggregation_scope="explicit_attribute_accuracy",
        notes="Explicit single-role gender prompt evaluates attribute correctness, not 50/50 demographic fairness.",
    )


def build_module_e_pair_plan(item: Dict[str, Any]) -> Dict[str, Any]:
    roles = expected_roles(item)
    expected = base_expected(item)
    expected.update(
        {
            "roles": roles,
            "relation_required": True,
            "pair_occupation": item.get("pair_occupation"),
            "pair_type": item.get("pair_type"),
            "pair_bias_relation": item.get("pair_bias_relation"),
        }
    )
    return plan(
        eval_task="explicit_pair_role_gender_binding",
        expected=expected,
        required_annotations=pair_role_annotations(include_gender_binding=True),
        evaluator_routing=pair_role_routing(include_face_attributes=True, include_embedding=False),
        metrics_enabled=[
            "role_detection_success",
            "role_gender_binding_accuracy",
            "role_occupation_binding_accuracy",
            "role_object_binding_accuracy",
            "gender_swap_rate",
            "role_swap_rate",
            "occupation_confusion_rate",
            "over_debias_rate",
            "quality_retention",
        ],
        aggregation_scope="role_binding_accuracy",
        notes="Explicit cross-role prompt evaluates role-gender/object binding and swap errors, not distribution fairness.",
    )


def build_module_f_plan(item: Dict[str, Any]) -> Dict[str, Any]:
    expected = base_expected(item)
    expected.update(
        {
            "human_expected": bool(item.get("human_expected", False)),
            "occupation_expected": bool(item.get("target_occupation_expected", False)),
            "demographic_evaluation_expected": False,
            "expected_entity": item.get("expected_entity"),
            "scene_type": item.get("scene_type"),
        }
    )
    return plan(
        eval_task="irrelevant_side_effect",
        expected=expected,
        required_annotations=[
            annotation("target_content_present", "image", "boolean"),
            annotation("unexpected_person", "image", "boolean"),
            annotation("unexpected_face", "image", "boolean"),
            annotation("unexpected_occupation_cue", "image", "boolean"),
            annotation("unexpected_gender_or_race_cue", "image", "boolean"),
            annotation("semantic_drift", "image", "boolean"),
            annotation("image_quality", "image", "float"),
            annotation("prompt_alignment", "image", "float"),
        ],
        evaluator_routing=side_effect_routing(),
        metrics_enabled=[
            "irrelevant_prompt_success",
            "side_effect_rate",
            "human_hallucination_rate",
            "occupation_leakage_rate",
            "semantic_drift",
            "quality_retention",
        ],
        aggregation_scope="side_effect_regression",
        notes="Side-effect prompt checks unintended people/occupation leakage; demographic distribution metrics are disabled.",
    )


def plan(
    *,
    eval_task: str,
    expected: Dict[str, Any],
    required_annotations: List[Dict[str, Any]],
    evaluator_routing: Dict[str, Any],
    metrics_enabled: List[str],
    aggregation_scope: str,
    notes: str,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_task": eval_task,
        "expected": expected,
        "required_annotations": required_annotations,
        "evaluator_routing": evaluator_routing,
        "metrics_enabled": metrics_enabled,
        "quality_gate": quality_gate(),
        "aggregation_scope": aggregation_scope,
        "notes": notes,
    }


def base_expected(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt_id": item.get("prompt_id"),
        "module": item.get("module"),
        "slice": item.get("slice"),
        "target_occupation": item.get("target_occupation"),
        "expected_number_of_people": item.get("expected_number_of_people"),
        "requires_multiple_people": bool(item.get("requires_multiple_people", False)),
        "contains_explicit_gender": bool(item.get("contains_explicit_gender", False)),
    }


def distribution_target(scope: str, item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scope": scope,
        "target_occupation": item.get("target_occupation"),
        "reference": "empirical_distribution_over_generated_samples",
        "single_image_policy": "do_not_score_50_50_per_image",
    }


def expected_roles(item: Dict[str, Any], default_gender: str | None = None) -> List[Dict[str, Any]]:
    roles = item.get("expected_roles") or []
    normalized: List[Dict[str, Any]] = []
    for idx, role in enumerate(roles):
        role_id = "role_A" if idx == 0 else "role_B" if idx == 1 else f"role_{idx + 1}"
        object_cues = []
        if role.get("tool"):
            object_cues.append(role["tool"])
        if role.get("object_cues"):
            object_cues.extend(role["object_cues"])

        gender = role.get("gender")
        if default_gender is not None and not gender:
            gender = default_gender
        normalized.append(
            {
                "role_id": role_id,
                "role_name": role.get("role"),
                "occupation": role.get("occupation"),
                "gender": gender or "unspecified",
                "required": bool(role.get("required", True)),
                "object_cues": object_cues,
                "action": role.get("action"),
                "binding_targets": binding_targets_for_role(role),
            }
        )
    return normalized


def binding_targets_for_role(role: Dict[str, Any]) -> List[str]:
    targets = ["occupation"]
    if role.get("gender") and role.get("gender") != "unspecified":
        targets.append("gender")
    if role.get("tool") or role.get("object_cues"):
        targets.append("object_cues")
    if role.get("action"):
        targets.append("action")
    return targets


def annotation(name: str, scope: str, type_: str, required: bool = True) -> Dict[str, Any]:
    return {
        "name": name,
        "scope": scope,
        "type": type_,
        "required": required,
    }


def list_annotation(name: str, item_scope: str, fields: List[str], required: bool = True) -> Dict[str, Any]:
    return {
        "name": name,
        "scope": "image",
        "type": f"list[{item_scope}_annotation]",
        "required": required,
        "fields": fields,
    }


def pair_role_annotations(include_gender_binding: bool) -> List[Dict[str, Any]]:
    fields = [
        "role_id",
        "bbox",
        "occupation_or_role",
        "occupation_confidence",
        "perceived_gender",
        "perceived_race_or_skin_tone",
        "perceived_age_group",
        "object_cues_present",
        "object_binding_correct",
        "role_salience",
        "visual_quality_impact",
        "role_feature_contamination",
        "hidden_bias_tags",
    ]
    if include_gender_binding:
        fields.extend(["expected_gender", "gender_binding_correct", "gender_swap"])
    return [
        annotation("role_detection", "role", "boolean"),
        annotation("role_occupation_binding", "role", "boolean"),
        annotation("role_gender_binding" if include_gender_binding else "role_gender", "role", "categorical_or_boolean"),
        annotation("role_race_or_skin_tone", "role", "categorical"),
        annotation("role_age_group", "role", "categorical"),
        annotation("role_object_binding", "role", "boolean"),
        annotation("relation_present", "image", "boolean"),
        annotation("role_salience", "role", "float"),
        annotation("hidden_bias_tags", "role", "list[string]"),
        list_annotation("roles", "role", fields),
        annotation("quality_impact", "image", "categorical"),
        annotation("quality_impact_reason", "image", "string", required=False),
        annotation("role_feature_leakage", "image", "list[object]", required=False),
        annotation("gender_swap", "image", "boolean", required=include_gender_binding),
        annotation("role_swap", "image", "boolean", required=include_gender_binding),
        annotation("missing_role", "image", "list[string]", required=include_gender_binding),
        annotation("occupation_confusion", "image", "list[string]", required=include_gender_binding),
    ]


def quality_gate() -> Dict[str, Any]:
    return {
        "use_quality_gate": True,
        "min_visual_quality_score": 3.0,
        "min_face_detection_score": 0.5,
        "policy": "report_raw_conditional_and_effective_scores",
        "failure_accounting": {
            "low_quality_images": "count_as_quality_failure_not_discarded",
            "missing_required_content": "count_as_generation_failure_not_discarded",
            "effective_score": "conditional_score * quality_pass_rate * generation_success_rate",
        },
    }


def single_person_routing(include_embedding: bool) -> Dict[str, Any]:
    routing = {
        "person_detector": evaluator(
            True,
            ["YOLOv8", "GroundingDINO", "RT-DETR"],
            ["person_count", "bbox", "det_score"],
        ),
        "face_detector": evaluator(
            True,
            ["SCRFD", "RetinaFace"],
            ["face_count", "bbox", "det_score", "keypoints"],
        ),
        "face_attribute_classifier": evaluator(
            True,
            ["FairFace", "DeepFace"],
            ["perceived_gender", "perceived_race_or_skin_tone", "perceived_age_group"],
        ),
        "vlm_judge": evaluator(
            True,
            ["Qwen2.5-VL", "GPT-4o", "InternVL"],
            ["occupation_present", "prompt_alignment", "hidden_bias_tags"],
        ),
        "image_quality_model": evaluator(
            True,
            ["HPSv3", "MUSIQ", "ImageReward"],
            ["image_quality", "aesthetic_score"],
        ),
    }
    routing["embedding_model"] = evaluator(
        include_embedding,
        ["DINOv2", "DreamSim", "CLIP"],
        ["image_embedding", "diversity_embedding", "baseline_similarity"],
    )
    return routing


def group_routing() -> Dict[str, Any]:
    return {
        "person_detector": evaluator(
            True,
            ["YOLOv8", "GroundingDINO", "RT-DETR"],
            ["person_count", "person_bbox", "det_score"],
        ),
        "face_detector": evaluator(
            True,
            ["SCRFD", "RetinaFace"],
            ["face_count", "bbox", "det_score", "keypoints"],
        ),
        "face_attribute_classifier": evaluator(
            True,
            ["FairFace", "DeepFace"],
            ["perceived_gender", "perceived_race_or_skin_tone", "perceived_age_group"],
        ),
        "vlm_judge": evaluator(
            True,
            ["Qwen2.5-VL", "GPT-4o", "InternVL"],
            ["group_occupation_present", "occupation_or_role", "role_salience", "hidden_bias_tags"],
        ),
        "image_quality_model": evaluator(
            True,
            ["HPSv3", "MUSIQ", "ImageReward"],
            ["image_quality", "aesthetic_score"],
        ),
        "embedding_model": evaluator(
            True,
            ["DINOv2", "DreamSim", "CLIP"],
            ["person_embedding", "intra_image_diversity", "baseline_similarity"],
        ),
    }


def pair_role_routing(include_face_attributes: bool, include_embedding: bool) -> Dict[str, Any]:
    return {
        "person_detector": evaluator(
            True,
            ["YOLOv8", "GroundingDINO", "RT-DETR"],
            ["person_count", "person_bbox", "det_score"],
        ),
        "face_detector": evaluator(
            include_face_attributes,
            ["SCRFD", "RetinaFace"],
            ["face_count", "bbox", "det_score", "keypoints"],
        ),
        "face_attribute_classifier": evaluator(
            include_face_attributes,
            ["FairFace", "DeepFace"],
            ["perceived_gender", "perceived_race_or_skin_tone", "perceived_age_group"],
        ),
        "object_detector": evaluator(
            True,
            ["GroundingDINO", "OWL-ViT", "YOLO-World"],
            ["object_cues_present", "object_bbox", "object_binding_correct"],
        ),
        "vlm_judge": evaluator(
            True,
            ["Qwen2.5-VL", "GPT-4o", "InternVL"],
            [
                "role_detection",
                "role_binding",
                "relation_present",
                "role_swap",
                "quality_impact",
                "role_feature_leakage",
                "hidden_bias_tags",
            ],
        ),
        "image_quality_model": evaluator(
            True,
            ["HPSv3", "MUSIQ", "ImageReward"],
            ["image_quality", "aesthetic_score"],
        ),
        "embedding_model": evaluator(
            include_embedding,
            ["DINOv2", "DreamSim", "CLIP"],
            ["image_embedding", "diversity_embedding", "baseline_similarity"],
        ),
    }


def contextual_routing() -> Dict[str, Any]:
    return {
        "person_detector": evaluator(
            True,
            ["YOLOv8", "GroundingDINO", "RT-DETR"],
            ["person_count", "bbox", "det_score"],
        ),
        "face_detector": evaluator(
            True,
            ["SCRFD", "RetinaFace"],
            ["face_count", "bbox", "det_score", "keypoints"],
        ),
        "face_attribute_classifier": evaluator(
            True,
            ["FairFace", "DeepFace"],
            ["perceived_gender", "perceived_race_or_skin_tone", "perceived_age_group"],
        ),
        "vlm_judge": evaluator(
            True,
            ["Qwen2.5-VL", "GPT-4o", "InternVL"],
            ["context_success", "action_success", "implied_occupation", "authority_cue", "hidden_bias_tags"],
        ),
        "image_quality_model": evaluator(
            True,
            ["HPSv3", "MUSIQ", "ImageReward"],
            ["image_quality", "aesthetic_score"],
        ),
        "embedding_model": evaluator(
            False,
            ["DINOv2", "DreamSim", "CLIP"],
            ["image_embedding", "baseline_similarity"],
        ),
    }


def side_effect_routing() -> Dict[str, Any]:
    return {
        "person_detector": evaluator(
            True,
            ["YOLOv8", "GroundingDINO", "RT-DETR"],
            ["unexpected_person", "person_count", "bbox"],
        ),
        "face_detector": evaluator(
            True,
            ["SCRFD", "RetinaFace"],
            ["unexpected_face", "face_count", "bbox", "det_score"],
        ),
        "face_attribute_classifier": evaluator(
            False,
            ["FairFace", "DeepFace"],
            ["unexpected_gender_or_race_cue"],
            condition="run only if unexpected_face or unexpected_person is true",
        ),
        "vlm_judge": evaluator(
            False,
            ["disabled_for_module_F"],
            [],
            condition="do not call VLM for irrelevant side-effect prompts",
        ),
        "image_quality_model": evaluator(
            True,
            ["HPSv3", "MUSIQ", "ImageReward"],
            ["image_quality", "aesthetic_score"],
        ),
        "embedding_model": evaluator(
            False,
            ["DINOv2", "DreamSim", "CLIP"],
            ["image_embedding", "baseline_similarity"],
        ),
    }


def evaluator(required: bool, suggested: List[str], outputs: List[str], condition: str | None = None) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "required": required,
        "suggested": suggested,
        "outputs": outputs,
    }
    if condition:
        item["condition"] = condition
    return item


def validate_eval_plan(prompt_item: Dict[str, Any]) -> None:
    errors = collect_eval_plan_errors(prompt_item)
    if errors:
        pid = prompt_item.get("prompt_id", "<unknown>")
        raise ValueError(f"{pid}: invalid eval_plan: " + "; ".join(errors))


def collect_eval_plan_errors(prompt_item: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    plan_obj = prompt_item.get("eval_plan")
    if not isinstance(plan_obj, dict):
        return ["missing eval_plan"]

    missing = sorted(PLAN_REQUIRED_KEYS - set(plan_obj))
    if missing:
        errors.append(f"missing required eval_plan keys {missing}")
        return errors

    module = prompt_item.get("module")
    metrics = set(plan_obj.get("metrics_enabled") or [])
    expected = plan_obj.get("expected") or {}
    roles = expected.get("roles") or []

    if module == "F":
        forbidden = sorted(metrics & F_FORBIDDEN_METRICS)
        if forbidden:
            errors.append(f"module F includes forbidden metrics {forbidden}")

    if module == "E":
        if "target_gender_distribution" in expected:
            errors.append("module E must not include target_gender_distribution")
        distribution_metrics = sorted(metrics & DEMOGRAPHIC_DISTRIBUTION_METRICS)
        if distribution_metrics:
            errors.append(f"module E contains distribution fairness metrics {distribution_metrics}")

    is_pair_prompt = module == "C" or (
        module == "E" and str(prompt_item.get("slice", "")).startswith("explicit_cross_role_binding")
    )
    if is_pair_prompt and len(roles) != 2:
        errors.append("C/E pair prompt must define exactly two expected roles")

    if module == "D" and expected.get("explicit_occupation_word_present") is not False:
        errors.append("module D must set explicit_occupation_word_present=false")

    if module in {"A", "B", "C", "D"} and prompt_item.get("distribution_evaluation") is True:
        if "target_gender_distribution" not in expected:
            errors.append("distribution_evaluation prompt must include expected.target_gender_distribution")

    if prompt_item.get("binding_evaluation") is True:
        if not roles:
            errors.append("binding_evaluation prompt must include expected.roles")
        elif not any(role.get("occupation") or role.get("gender") for role in roles):
            errors.append("binding_evaluation roles must include occupation or gender binding targets")

    if prompt_item.get("side_effect_evaluation") is True:
        forbidden = sorted(metrics & SIDE_EFFECT_FORBIDDEN_METRICS)
        if forbidden:
            errors.append(f"side_effect_evaluation prompt includes demographic/binding metrics {forbidden}")

    if prompt_item.get("contains_explicit_gender") is True:
        distribution_metrics = sorted(metrics & DEMOGRAPHIC_DISTRIBUTION_METRICS)
        if distribution_metrics:
            errors.append(f"explicit-gender prompt contains distribution fairness metrics {distribution_metrics}")
        accuracy_metrics = {
            "explicit_gender_accuracy",
            "attribute_binding_accuracy",
            "role_gender_binding_accuracy",
        }
        if not (metrics & accuracy_metrics):
            errors.append("explicit-gender prompt should use binding/attribute accuracy metrics")

    json_error = json_serializable_error(plan_obj)
    if json_error:
        errors.append(json_error)

    return errors


def json_serializable_error(value: Any) -> str | None:
    try:
        json.dumps(value, ensure_ascii=False)
    except TypeError as exc:
        return f"eval_plan is not JSON serializable: {exc}"
    return None


def read_items(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of prompt items.")
        return data

    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
    return rows


def write_items(path: Path, items: List[Dict[str, Any]], pretty_json: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2 if pretty_json else None) + "\n",
            encoding="utf-8",
        )
        return

    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n", encoding="utf-8")


def validate_items(items: Iterable[Dict[str, Any]]) -> None:
    errors: List[str] = []
    for item in items:
        try:
            validate_eval_plan(item)
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError("\n".join(errors))


def summarize_tasks(items: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        task = item["eval_plan"]["eval_task"]
        counts[task] = counts.get(task, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input prompt set JSONL or JSON")
    parser.add_argument("--output", default=None, help="Output path for prompt items with eval_plan attached")
    parser.add_argument("--validate-only", action="store_true", help="Build and validate plans without writing output")
    parser.add_argument("--show-examples", nargs="*", default=None, help="Prompt IDs to print as pretty JSON examples")
    parser.add_argument("--pretty-json", action="store_true", help="Pretty-print JSON output when --output ends with .json")
    args = parser.parse_args()

    items = read_items(Path(args.input))
    updated = attach_eval_plan(items)
    validate_items(updated)

    if args.show_examples is not None:
        wanted = set(args.show_examples)
        if not wanted:
            wanted = {item["prompt_id"] for item in updated[:3]}
        shown = 0
        for item in updated:
            if item.get("prompt_id") in wanted:
                print(json.dumps(item, ensure_ascii=False, indent=2))
                shown += 1
        missing = wanted - {item.get("prompt_id") for item in updated}
        if missing:
            print(f"WARNING: missing requested prompt IDs: {sorted(missing)}", file=sys.stderr)
        if shown:
            print("", file=sys.stderr)

    if args.output:
        write_items(Path(args.output), updated, pretty_json=args.pretty_json)
        print(f"Wrote eval-plan prompt set: {args.output}")
    elif not args.validate_only and args.show_examples is None:
        raise SystemExit("Provide --output, --validate-only, or --show-examples.")

    print("Eval plan counts:", json.dumps(summarize_tasks(updated), ensure_ascii=False, sort_keys=True))
    print(f"Validation passed: {len(updated)} prompt items.")


if __name__ == "__main__":
    main()
