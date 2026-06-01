from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List

from benchmark.metric_calculator import METRIC_REGISTRY, METRIC_TIERS, metric_tier


GROUP_KEYS = ["overall", "module", "slice", "eval_task", "prompt_id", "aggregation_scope"]


def aggregate_metrics(per_image_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "schema_version": "v1.0",
        "total_images": len(per_image_metrics),
        "metric_tiers": list(METRIC_TIERS),
        "metric_definitions": METRIC_REGISTRY,
        "groups": {group_key: aggregate_by(per_image_metrics, group_key) for group_key in GROUP_KEYS},
    }


def aggregate_by(rows: List[Dict[str, Any]], group_key: str) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = "overall" if group_key == "overall" else str(row.get(group_key, "unknown"))
        grouped[key].append(row)
    return {key: aggregate_rows(group_rows) for key, group_rows in sorted(grouped.items())}


def aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_values: Dict[str, List[Any]] = defaultdict(list)
    for row in rows:
        for metric, value in (row.get("metrics") or {}).items():
            metric_values[metric].append(value)

    metrics = {}
    for metric, values in sorted(metric_values.items()):
        metrics[metric] = summarize_values(values)

    return {
        "count": len(rows),
        "status_rates": status_rates(rows),
        "metrics": metrics,
        "metric_groups": group_metric_summaries(metrics),
        "demographic_counts": demographic_counts(rows),
    }


def group_metric_summaries(metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {tier: {} for tier in METRIC_TIERS}
    for metric, summary in metrics.items():
        grouped[metric_tier(metric)][metric] = summary
    return grouped


def summarize_values(values: List[Any]) -> Dict[str, Any]:
    numeric = [float(value) for value in values if isinstance(value, (int, float)) and value is not None]
    missing = len(values) - len(numeric)
    summary: Dict[str, Any] = {
        "n": len(values),
        "valid_n": len(numeric),
        "missing_n": missing,
        "mean": sum(numeric) / len(numeric) if numeric else None,
    }
    if numeric:
        summary["min"] = min(numeric)
        summary["max"] = max(numeric)
    return summary


def status_rates(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    fields = ["generation_success", "quality_pass", "content_pass", "annotation_complete"]
    result = {}
    for field in fields:
        values = [bool((row.get("status") or {}).get(field)) for row in rows]
        result[field] = sum(values) / len(values) if values else None
    return result


def demographic_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    counters = {
        "gender": Counter(),
        "race_or_skin_tone": Counter(),
        "age_group": Counter(),
    }
    for row in rows:
        observations = row.get("demographic_observations") or {}
        for field in counters:
            counters[field].update(value for value in observations.get(field, []) if value)
    return {field: dict(counter) for field, counter in counters.items()}


def top_level_summary(aggregate: Dict[str, Any]) -> Dict[str, Any]:
    overall = aggregate["groups"]["overall"]["overall"]
    module_counts = {
        module: data["count"]
        for module, data in aggregate["groups"].get("module", {}).items()
    }
    return {
        "total_images": aggregate["total_images"],
        "module_counts": module_counts,
        "status_rates": overall["status_rates"],
        "metric_groups": overall.get("metric_groups", {}),
        "demographic_counts": overall.get("demographic_counts", {}),
    }
