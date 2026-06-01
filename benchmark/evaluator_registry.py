from __future__ import annotations

import hashlib
from typing import Any, Dict, List


class MockEvaluator:
    """Deterministic placeholder evaluator for end-to-end pipeline testing.

    It does not inspect pixels. It fills the annotation schema from prompt
    metadata so downstream schema validation, metrics, and aggregation can run
    before real local/API evaluators are integrated.
    """

    name = "mock"
    version = "v0.1"

    def annotate(self, record: Dict[str, Any]) -> Dict[str, Any]:
        item = record["prompt_item"]
        plan = item["eval_plan"]
        image_exists = bool(record.get("image_exists"))
        eval_task = plan["eval_task"]
        annotations = self._annotations_for_task(eval_task, item, image_exists)

        quality_score = annotations.get("image_quality", 4.0 if image_exists else 0.0)
        generation_success = image_exists
        quality_pass = image_exists and quality_score >= plan["quality_gate"]["min_visual_quality_score"]
        content_pass = image_exists and self._content_pass(eval_task, annotations)

        return {
            "annotation_id": record["image_id"],
            "prompt_id": record["prompt_id"],
            "image_id": record["image_id"],
            "image_path": record["image_path"],
            "sample_index": record.get("sample_index"),
            "sample_seed": record.get("sample_seed"),
            "prompt": item.get("prompt"),
            "module": item.get("module"),
            "slice": item.get("slice"),
            "target_occupation": item.get("target_occupation"),
            "eval_task": eval_task,
            "aggregation_scope": plan.get("aggregation_scope"),
            "status": {
                "generation_success": generation_success,
                "quality_pass": quality_pass,
                "content_pass": content_pass,
                "annotation_complete": True,
                "evaluator_mode": self.name,
            },
            "expected": plan.get("expected", {}),
            "annotations": annotations,
            "required_annotations": plan.get("required_annotations", []),
            "metrics_enabled": plan.get("metrics_enabled", []),
            "quality_gate": plan.get("quality_gate", {}),
            "evaluator_outputs": {
                "mock_evaluator": {
                    "model": self.name,
                    "version": self.version,
                    "raw_output_path": None,
                    "note": "Synthetic annotations generated from metadata only; no image pixels were evaluated.",
                }
            },
        }

    def _annotations_for_task(self, eval_task: str, item: Dict[str, Any], image_exists: bool) -> Dict[str, Any]:
        if eval_task == "neutral_single_role_fairness":
            return self._single_role_annotations(item, image_exists)
        if eval_task == "group_occupation_fairness":
            return self._group_annotations(item, image_exists)
        if eval_task == "neutral_pair_role_fairness":
            return self._pair_annotations(item, image_exists, explicit_gender=False)
        if eval_task == "contextual_trigger_bias":
            return self._contextual_annotations(item, image_exists)
        if eval_task == "explicit_single_role_binding":
            return self._explicit_single_annotations(item, image_exists)
        if eval_task == "explicit_pair_role_gender_binding":
            return self._pair_annotations(item, image_exists, explicit_gender=True)
        if eval_task == "irrelevant_side_effect":
            return self._side_effect_annotations(item, image_exists)
        raise ValueError(f"Unsupported eval_task for mock evaluator: {eval_task}")

    def _single_role_annotations(self, item: Dict[str, Any], image_exists: bool) -> Dict[str, Any]:
        target = item.get("target_occupation")
        return {
            "image_quality": 4.0 if image_exists else 0.0,
            "prompt_alignment": 0.95 if image_exists else 0.0,
            "person_count": 1 if image_exists else 0,
            "face_count": 1 if image_exists else 0,
            "clear_face_count": 1 if image_exists else 0,
            "occupation_recognition": {
                "expected_occupation": target,
                "detected_occupation": target if image_exists else None,
                "present": image_exists,
                "confidence": 0.95 if image_exists else 0.0,
            },
            "occupation_present": image_exists,
            "perceived_gender": self._stable_unknown_or_gender(item["prompt_id"]),
            "perceived_race_or_skin_tone": "unknown",
            "perceived_age_group": "unknown",
            "hidden_bias_tags": [],
            "scene_success": image_exists if item.get("slice") == "neutral_occupation_workplace" else None,
            "workplace_relevance": 0.9 if image_exists and item.get("slice") == "neutral_occupation_workplace" else None,
            "professional_object_presence": [],
        }

    def _group_annotations(self, item: Dict[str, Any], image_exists: bool) -> Dict[str, Any]:
        min_people = int(item.get("minimum_valid_people", 2) or 2)
        persons = []
        if image_exists:
            for idx in range(min_people):
                persons.append(
                    {
                        "person_id": f"person_{idx + 1}",
                        "bbox": None,
                        "face_visible": True,
                        "occupation_or_role": item.get("target_occupation"),
                        "perceived_gender": self._stable_unknown_or_gender(f"{item['prompt_id']}:{idx}"),
                        "perceived_race_or_skin_tone": "unknown",
                        "perceived_age_group": "unknown",
                        "role_salience": 1.0 / min_people,
                        "hidden_bias_tags": [],
                    }
                )
        return {
            "person_count": len(persons),
            "face_count": len(persons),
            "clear_face_count": len(persons),
            "group_occupation_present": image_exists,
            "persons": persons,
            "image_quality": 4.0 if image_exists else 0.0,
            "prompt_alignment": 0.95 if image_exists else 0.0,
        }

    def _pair_annotations(self, item: Dict[str, Any], image_exists: bool, explicit_gender: bool) -> Dict[str, Any]:
        expected_roles = item["eval_plan"]["expected"].get("roles", [])
        roles: List[Dict[str, Any]] = []
        for role in expected_roles:
            expected_gender = role.get("gender", "unspecified")
            perceived_gender = expected_gender if explicit_gender and expected_gender != "unspecified" else "unknown"
            roles.append(
                {
                    "role_id": role.get("role_id"),
                    "bbox": None,
                    "expected_occupation": role.get("occupation"),
                    "detected_occupation": role.get("occupation") if image_exists else None,
                    "occupation_or_role": role.get("occupation") if image_exists else None,
                    "occupation_confidence": 0.9 if image_exists else 0.0,
                    "expected_gender": expected_gender,
                    "perceived_gender": perceived_gender,
                    "gender_binding_correct": bool(image_exists and (not explicit_gender or perceived_gender == expected_gender)),
                    "perceived_race_or_skin_tone": "unknown",
                    "perceived_age_group": "unknown",
                    "object_cues_present": role.get("object_cues", []) if image_exists else [],
                    "object_binding_correct": image_exists,
                    "role_occupation_binding_correct": image_exists,
                    "role_salience": 0.5 if image_exists else 0.0,
                    "visual_quality_impact": "无影响",
                    "role_feature_contamination": [],
                    "hidden_bias_tags": [],
                }
            )

        return {
            "role_detection": image_exists and len(roles) == 2,
            "role_occupation_binding": image_exists,
            "role_gender": [role["perceived_gender"] for role in roles],
            "role_gender_binding": image_exists if explicit_gender else None,
            "role_race_or_skin_tone": [role["perceived_race_or_skin_tone"] for role in roles],
            "role_age_group": [role["perceived_age_group"] for role in roles],
            "role_object_binding": image_exists,
            "relation_present": image_exists,
            "role_salience": [role["role_salience"] for role in roles],
            "hidden_bias_tags": [],
            "roles": roles,
            "gender_swap": False,
            "role_swap": False,
            "missing_role": [] if image_exists else [role.get("role_id") for role in expected_roles],
            "occupation_confusion": [],
            "quality_impact": "无影响",
            "quality_impact_reason": "",
            "role_feature_leakage": [],
            "image_quality": 4.0 if image_exists else 0.0,
            "prompt_alignment": 0.95 if image_exists else 0.0,
        }

    def _contextual_annotations(self, item: Dict[str, Any], image_exists: bool) -> Dict[str, Any]:
        gender = self._stable_unknown_or_gender(item["prompt_id"])
        persons = [
            {
                "person_id": "person_0",
                "face_id": "face_0",
                "bbox": None,
                "face_visible": image_exists,
                "clear_face": image_exists,
                "is_real_human_face": image_exists,
                "contextual_role": "expected_subject" if image_exists else "unknown",
                "is_expected_prompt_subject": image_exists,
                "is_extra_person": False,
                "included_in_demographic_stats": image_exists,
                "occupation_or_role": item.get("target_occupation") if image_exists else "unknown",
                "occupation_confidence": 0.85 if image_exists else 0.0,
                "perceived_gender": gender if image_exists else "unknown",
                "perceived_race_or_skin_tone": "unknown",
                "perceived_age_group": "unknown",
                "role_salience": 1.0 if image_exists else 0.0,
                "prompt_alignment_issue": False,
                "evidence": [],
            }
        ] if image_exists else []
        return {
            "context_success": image_exists,
            "action_success": image_exists,
            "implied_occupation": item.get("target_occupation") if image_exists else None,
            "implied_occupation_confidence": 0.85 if image_exists else 0.0,
            "person_count": len(persons),
            "face_count": len(persons),
            "clear_face_count": len(persons),
            "persons": persons,
            "unmatched_person_mentions": [],
            "expected_subject_face_ids": ["face_0"] if image_exists else [],
            "extra_person_count": 0,
            "unmatched_extra_person_count": 0,
            "multi_person_prompt_alignment_issue": False,
            "prompt_alignment_issues": [],
            "perceived_gender": gender if image_exists else "unknown",
            "perceived_race_or_skin_tone": "unknown",
            "perceived_age_group": "unknown",
            "authority_cue": "unknown",
            "attire": "unknown",
            "pose": "unknown",
            "expression": "unknown",
            "hidden_bias_tags": [],
            "image_quality": 4.0 if image_exists else 0.0,
            "prompt_alignment": 0.95 if image_exists else 0.0,
        }

    def _explicit_single_annotations(self, item: Dict[str, Any], image_exists: bool) -> Dict[str, Any]:
        role = (item["eval_plan"]["expected"].get("roles") or [{}])[0]
        expected_gender = role.get("gender", "unspecified")
        perceived_gender = expected_gender if image_exists else "unknown"
        gender_correct = image_exists and expected_gender == perceived_gender
        return {
            "occupation_present": image_exists,
            "expected_gender": expected_gender,
            "perceived_gender": perceived_gender,
            "gender_correct": gender_correct,
            "occupation_gender_binding_correct": gender_correct,
            "gender_clarity": 1.0 if gender_correct else 0.0,
            "face_visible": image_exists,
            "image_quality": 4.0 if image_exists else 0.0,
            "prompt_alignment": 0.95 if image_exists else 0.0,
        }

    def _side_effect_annotations(self, item: Dict[str, Any], image_exists: bool) -> Dict[str, Any]:
        return {
            "target_content_present": image_exists,
            "unexpected_person": False,
            "unexpected_face": False,
            "unexpected_occupation_cue": False,
            "unexpected_gender_or_race_cue": False,
            "semantic_drift": False,
            "image_quality": 4.0 if image_exists else 0.0,
            "prompt_alignment": 0.95 if image_exists else 0.0,
        }

    def _content_pass(self, eval_task: str, annotations: Dict[str, Any]) -> bool:
        if eval_task == "irrelevant_side_effect":
            return bool(annotations.get("target_content_present")) and not bool(annotations.get("semantic_drift"))
        if eval_task == "contextual_trigger_bias":
            return bool(annotations.get("context_success")) and bool(annotations.get("action_success"))
        if eval_task == "group_occupation_fairness":
            return bool(annotations.get("group_occupation_present")) and bool(annotations.get("persons"))
        if eval_task in {"neutral_pair_role_fairness", "explicit_pair_role_gender_binding"}:
            return bool(annotations.get("role_detection")) and bool(annotations.get("relation_present"))
        if eval_task == "explicit_single_role_binding":
            return bool(annotations.get("occupation_present"))
        return bool(annotations.get("occupation_recognition", {}).get("present"))

    def _stable_unknown_or_gender(self, key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return ["male", "female", "unknown"][int(digest[:2], 16) % 3]


def get_evaluator(name: str, **kwargs: Any) -> Any:
    if name == "mock":
        return MockEvaluator()
    if name in {"qwen_vl", "dashscope_vlm"}:
        from benchmark.vlm_evaluator import DashScopeVLMEvaluator

        return DashScopeVLMEvaluator(**kwargs)
    raise ValueError(f"Unknown evaluator backend: {name}. Available: mock, qwen_vl")
