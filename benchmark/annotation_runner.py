from __future__ import annotations

from typing import Any, Dict, Iterable, List

from benchmark.annotation_schema import validate_annotation_record
from benchmark.evaluator_registry import get_evaluator


def run_annotations(
    records: Iterable[Dict[str, Any]],
    evaluator_name: str = "mock",
    evaluator_kwargs: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    evaluator = get_evaluator(evaluator_name, **(evaluator_kwargs or {}))
    annotations: List[Dict[str, Any]] = []
    for record in records:
        annotation_record = evaluator.annotate(record)
        errors = validate_annotation_record(annotation_record)
        annotation_record["validation_errors"] = errors
        if errors:
            annotation_record["status"]["annotation_complete"] = False
        annotations.append(annotation_record)
    return annotations
