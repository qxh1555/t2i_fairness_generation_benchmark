from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from benchmark.aggregator import top_level_summary
from benchmark.metric_calculator import metric_definition


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    *,
    prompt_set_path: str,
    manifest_path: str,
    evaluator: str,
    annotations: List[Dict[str, Any]],
    aggregate: Dict[str, Any],
    issues: List[str],
) -> None:
    summary = top_level_summary(aggregate)
    lines = [
        "# Benchmark Report",
        "",
        f"- Prompt set: `{prompt_set_path}`",
        f"- Manifest: `{manifest_path}`",
        f"- Evaluator backend: `{evaluator}`",
        f"- Total images: `{summary['total_images']}`",
        "",
        "## Status Rates",
        "",
        "| Field | Rate |",
        "|---|---:|",
    ]
    for field, value in summary["status_rates"].items():
        rendered = "n/a" if value is None else f"{value:.4f}"
        lines.append(f"| {field} | {rendered} |")

    append_metric_table(
        lines,
        "Core Metrics",
        summary.get("metric_groups", {}).get("core", {}),
        (
            "Core metrics are the default report view. Proxy-only quality/occupation fields "
            "remain in JSON outputs but are hidden here."
        ),
    )
    append_metric_table(
        lines,
        "Diagnostic Metrics",
        summary.get("metric_groups", {}).get("diagnostic", {}),
        "Diagnostic metrics explain common failure modes and should not be mixed into a single score.",
    )

    demographic_counts = summary.get("demographic_counts", {})
    if demographic_counts:
        lines.extend(["", "## Demographic Counts", ""])
        for field in ("gender", "race_or_skin_tone", "age_group"):
            counts = demographic_counts.get(field) or {}
            if not counts:
                continue
            rendered = ", ".join(f"{key}: {value}" for key, value in sorted(counts.items()))
            lines.append(f"- `{field}`: {rendered}")

    lines.extend(["", "## Module Counts", "", "| Module | Images |", "|---|---:|"])
    for module, count in sorted(summary["module_counts"].items()):
        lines.append(f"| {module} | {count} |")

    if issues:
        lines.extend(["", "## Dataset Issues", ""])
        for issue in issues:
            lines.append(f"- {issue}")

    validation_errors = [
        (row["annotation_id"], row.get("validation_errors", []))
        for row in annotations
        if row.get("validation_errors")
    ]
    if validation_errors:
        lines.extend(["", "## Annotation Validation Errors", ""])
        for annotation_id, errors in validation_errors[:20]:
            lines.append(f"- `{annotation_id}`: {'; '.join(errors)}")
        if len(validation_errors) > 20:
            lines.append(f"- ... {len(validation_errors) - 20} more")

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `annotations.jsonl`: per-image normalized annotation records.",
            "- `per_image_metrics.jsonl`: per-image metric values, metric groups, and demographic observations.",
            "- `aggregate_metrics.json`: grouped metric aggregation, including core/diagnostic/auxiliary/planned metric groups.",
            "- `benchmark_report.md`: this summary.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def append_metric_table(
    lines: List[str],
    title: str,
    metrics: Dict[str, Dict[str, Any]],
    note: str,
) -> None:
    if not metrics:
        return
    lines.extend(
        [
            "",
            f"## {title}",
            "",
            note,
            "",
            "| Metric | Mean | Valid N | Direction | Source |",
            "|---|---:|---:|---|---|",
        ]
    )
    for metric, summary in sorted(metrics.items()):
        definition = metric_definition(metric)
        mean = summary.get("mean")
        rendered_mean = "n/a" if mean is None else f"{float(mean):.4f}"
        valid_n = summary.get("valid_n", 0)
        direction = definition.get("direction", "unknown")
        source = definition.get("source", "unknown")
        display_name = definition.get("display_name", metric)
        if definition.get("proxy"):
            display_name = f"{display_name} (proxy)"
        lines.append(f"| `{display_name}` | {rendered_mean} | {valid_n} | {direction} | {source} |")
