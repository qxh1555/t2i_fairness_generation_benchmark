from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"JSON input must be a list: {path}")
        return data

    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object at {path}:{line_no}")
        rows.append(row)
    return rows


def ensure_eval_plans(prompt_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if all(isinstance(item.get("eval_plan"), dict) for item in prompt_items):
        return prompt_items

    from eval_plan_builder import attach_eval_plan

    return attach_eval_plan(prompt_items)


def load_benchmark_dataset(
    prompt_set_path: Path,
    manifest_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    prompt_items = ensure_eval_plans(read_json_or_jsonl(prompt_set_path))
    manifest_rows = read_json_or_jsonl(manifest_path)
    prompt_by_id = {item["prompt_id"]: item for item in prompt_items}

    records: List[Dict[str, Any]] = []
    issues: List[str] = []
    for manifest_row in manifest_rows:
        prompt_id = manifest_row.get("prompt_id")
        if prompt_id not in prompt_by_id:
            issues.append(f"Manifest prompt_id not found in prompt set: {prompt_id}")
            continue
        images = manifest_row.get("images") or []
        if not images:
            issues.append(f"Manifest row has no images: {prompt_id}")
            continue

        sample_seeds = manifest_row.get("sample_seeds") or []
        for idx, image_path in enumerate(images, start=1):
            image_path_obj = Path(image_path)
            sample_seed = sample_seeds[idx - 1] if idx - 1 < len(sample_seeds) else None
            records.append(
                {
                    "image_id": f"{prompt_id}__sample_{idx:03d}",
                    "prompt_id": prompt_id,
                    "sample_index": idx,
                    "sample_seed": sample_seed,
                    "image_path": str(image_path_obj),
                    "image_exists": image_path_obj.exists(),
                    "prompt_item": prompt_by_id[prompt_id],
                    "manifest": manifest_row,
                }
            )

    if not records:
        raise ValueError("No image records could be joined from prompt set and manifest.")
    return records, prompt_items, issues

