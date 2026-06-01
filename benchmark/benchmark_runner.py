#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.aggregator import aggregate_metrics
from benchmark.annotation_runner import run_annotations
from benchmark.auxiliary_results import attach_auxiliary_results
from benchmark.dataset_loader import load_benchmark_dataset
from benchmark.metric_calculator import calculate_per_image_metrics
from benchmark.report_writer import write_json, write_jsonl, write_report


def filter_records(
    records: list[dict],
    *,
    prompt_ids: list[str] | None = None,
    modules: list[str] | None = None,
    slices: list[str] | None = None,
    sample_indices: list[int] | None = None,
) -> list[dict]:
    prompt_id_set = set(prompt_ids or [])
    module_set = set(modules or [])
    slice_set = set(slices or [])
    sample_index_set = set(sample_indices or [])

    filtered = []
    for record in records:
        item = record["prompt_item"]
        if prompt_id_set and record.get("prompt_id") not in prompt_id_set:
            continue
        if module_set and item.get("module") not in module_set:
            continue
        if slice_set and item.get("slice") not in slice_set:
            continue
        if sample_index_set and record.get("sample_index") not in sample_index_set:
            continue
        filtered.append(record)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run metadata-driven benchmark evaluation from prompt eval plans and image manifest."
    )
    parser.add_argument("--prompt-set", required=True, help="Prompt set JSONL/JSON, with or without eval_plan")
    parser.add_argument("--manifest", required=True, help="Image generation manifest.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for benchmark outputs")
    parser.add_argument("--evaluator", default="mock", help="Evaluator backend. Available: mock, qwen_vl")
    parser.add_argument("--prompt-id", action="append", help="Filter by prompt_id; can be repeated")
    parser.add_argument("--module", action="append", help="Filter by module; can be repeated, e.g. --module C")
    parser.add_argument("--slice", action="append", help="Filter by slice; can be repeated")
    parser.add_argument("--sample-index", type=int, action="append", help="Filter by sample index; can be repeated")
    parser.add_argument("--face-results", default=None, help="Optional SCRFD all_face_results.json for face_id/bbox context")
    parser.add_argument("--face-attribute-results", default=None, help="Optional FairFace all_face_attribute_results.json for VLM context and post-VLM fusion")
    parser.add_argument(
        "--vlm-image-mode",
        choices=["auto", "original", "annotated"],
        default="auto",
        help="Image sent to VLM. auto sends SCRFD-annotated images for pair-role tasks when available.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of joined image records")
    parser.add_argument("--strict-images", action="store_true", help="Fail if any joined image path is missing")
    parser.add_argument("--dry-run", action="store_true", help="Load and join dataset without writing outputs")
    args = parser.parse_args()

    records, prompt_items, issues = load_benchmark_dataset(Path(args.prompt_set), Path(args.manifest))
    records = filter_records(
        records,
        prompt_ids=args.prompt_id,
        modules=args.module,
        slices=args.slice,
        sample_indices=args.sample_index,
    )
    if not records:
        raise SystemExit("No image records matched the selected filters.")
    if args.limit is not None:
        records = records[: args.limit]

    aux_issues = attach_auxiliary_results(
        records,
        face_results_path=Path(args.face_results) if args.face_results else None,
        face_attribute_results_path=Path(args.face_attribute_results) if args.face_attribute_results else None,
    )
    issues.extend(aux_issues)

    missing_images = [record["image_path"] for record in records if not record.get("image_exists")]
    if missing_images:
        issues.extend(f"Missing image file: {path}" for path in missing_images)
        if args.strict_images:
            raise SystemExit("Missing image files:\n" + "\n".join(missing_images))

    print(f"Prompt items: {len(prompt_items)}")
    print(f"Joined image records: {len(records)}")
    print(f"Evaluator backend: {args.evaluator}")
    if issues:
        print(f"Dataset issues: {len(issues)}")

    if args.dry_run:
        preview = [
            {
                "image_id": record["image_id"],
                "prompt_id": record["prompt_id"],
                "image_path": record["image_path"],
                "module": record["prompt_item"].get("module"),
                "slice": record["prompt_item"].get("slice"),
                "eval_task": record["prompt_item"]["eval_plan"]["eval_task"],
                "image_exists": record["image_exists"],
            }
            for record in records[:10]
        ]
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    annotations = run_annotations(
        records,
        evaluator_name=args.evaluator,
        evaluator_kwargs={"image_mode": args.vlm_image_mode},
    )
    per_image_metrics = calculate_per_image_metrics(annotations)
    aggregate = aggregate_metrics(per_image_metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "annotations.jsonl", annotations)
    write_jsonl(output_dir / "per_image_metrics.jsonl", per_image_metrics)
    write_json(output_dir / "aggregate_metrics.json", aggregate)
    write_json(output_dir / "dataset_issues.json", {"issues": issues})
    write_report(
        output_dir / "benchmark_report.md",
        prompt_set_path=args.prompt_set,
        manifest_path=args.manifest,
        evaluator=args.evaluator,
        annotations=annotations,
        aggregate=aggregate,
        issues=issues,
    )

    print(f"Wrote annotations: {output_dir / 'annotations.jsonl'}")
    print(f"Wrote per-image metrics: {output_dir / 'per_image_metrics.jsonl'}")
    print(f"Wrote aggregate metrics: {output_dir / 'aggregate_metrics.json'}")
    print(f"Wrote report: {output_dir / 'benchmark_report.md'}")


if __name__ == "__main__":
    main()
