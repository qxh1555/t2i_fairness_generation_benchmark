#!/usr/bin/env python3
"""
Batch image generation for one occupation/model from a JSON config.

This wrapper expands one occupation prompt set into per-prompt generation jobs
and delegates actual API calls to scripts/generate_images_from_prompt_set.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "generate_images_from_prompt_set.py"


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return data


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def slugify(text: str) -> str:
    import re

    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def find_prompt_set(occupation: str, explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Prompt set not found: {path}")
        return path

    occ_slug = slugify(occupation)
    candidates = [
        ROOT / "data" / "generated" / f"{occ_slug}_prompt_set.jsonl",
        ROOT / "data" / f"{occ_slug}_prompt_set.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Prompt set not found. Expected one of: "
        + ", ".join(str(path) for path in candidates)
        + ". Generate the prompt set first or set prompt_set in the batch config."
    )


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def select_prompts(
    rows: Iterable[Dict[str, Any]],
    *,
    prompt_ids: Iterable[str] = (),
    modules: Iterable[str] = (),
    slices: Iterable[str] = (),
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    prompt_id_set = set(prompt_ids)
    module_set = set(modules)
    slice_set = set(slices)
    selected: List[Dict[str, Any]] = []

    for row in rows:
        if prompt_id_set and row.get("prompt_id") not in prompt_id_set:
            continue
        if module_set and row.get("module") not in module_set:
            continue
        if slice_set and row.get("slice") not in slice_set:
            continue
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break

    return selected


def merge_options(base: Dict[str, Any], override: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = deepcopy(base)
    if override:
        for key, value in override.items():
            merged[key] = value
    return merged


def job_options(config: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    options = deepcopy(config.get("defaults", {}))
    for key in [
        "model",
        "size",
        "n",
        "seed",
        "prompt_extend",
        "watermark",
        "skip_existing",
        "output_dir",
        "api_base",
        "api_key",
        "negative_prompt",
        "timeout",
        "raw_response_dir",
    ]:
        if key in config:
            options[key] = config[key]

    module_overrides = config.get("module_overrides", {})
    slice_overrides = config.get("slice_overrides", {})
    prompt_overrides = config.get("prompt_overrides", {})

    options = merge_options(options, module_overrides.get(item.get("module")))
    options = merge_options(options, slice_overrides.get(item.get("slice")))
    options = merge_options(options, prompt_overrides.get(item.get("prompt_id")))
    return options


def format_option_strings(options: Dict[str, Any], occupation: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """Expand simple placeholders in string options such as output paths."""
    formatted = deepcopy(options)
    model = str(formatted.get("model") or "")
    seed = formatted.get("seed")
    context = {
        "occupation": occupation,
        "occupation_slug": slugify(occupation),
        "model": model,
        "model_slug": slugify(model) if model else "",
        "seed": str(seed) if seed is not None else "random",
        "prompt_id": item.get("prompt_id", ""),
        "module": item.get("module", ""),
        "slice": item.get("slice", ""),
    }
    for key, value in list(formatted.items()):
        if isinstance(value, str):
            formatted[key] = value.format(**context)
    return formatted


def append_flag(command: List[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_command(occupation: str, prompt_set: Path | None, item: Dict[str, Any], options: Dict[str, Any]) -> List[str]:
    command = [
        sys.executable,
        str(GENERATOR),
        "--occupation",
        occupation,
        "--prompt-id",
        item["prompt_id"],
    ]
    if prompt_set is not None:
        command.extend(["--prompt-set", str(prompt_set)])

    append_flag(command, "--model", options.get("model"))
    append_flag(command, "--size", options.get("size"))
    append_flag(command, "--n", options.get("n"))
    append_flag(command, "--seed", options.get("seed"))
    append_flag(command, "--output-dir", options.get("output_dir"))
    append_flag(command, "--api-base", options.get("api_base"))
    append_flag(command, "--api-key", options.get("api_key"))
    append_flag(command, "--negative-prompt", options.get("negative_prompt"))
    append_flag(command, "--timeout", options.get("timeout"))
    append_flag(command, "--raw-response-dir", options.get("raw_response_dir"))

    if options.get("prompt_extend") is False:
        command.append("--no-prompt-extend")
    if options.get("watermark"):
        command.append("--watermark")
    if options.get("skip_existing", True):
        command.append("--skip-existing")
    return command


def validate_options(item: Dict[str, Any], options: Dict[str, Any]) -> None:
    n = int(options.get("n", 1))
    seed = options.get("seed")
    if n < 1:
        raise ValueError(f"{item['prompt_id']}: n must be at least 1.")
    if seed is not None:
        seed = int(seed)
        if not 0 <= seed <= 2147483647:
            raise ValueError(f"{item['prompt_id']}: seed must be in [0, 2147483647].")
        if seed + n - 1 > 2147483647:
            raise ValueError(f"{item['prompt_id']}: seed + n - 1 must be <= 2147483647.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Batch generation JSON config.")
    parser.add_argument("--dry-run", action="store_true", help="Print expanded jobs without calling the API.")
    parser.add_argument("--module", action="append", help="Extra module filter; can be repeated.")
    parser.add_argument("--slice", action="append", help="Extra slice filter; can be repeated.")
    parser.add_argument("--prompt-id", action="append", help="Extra prompt_id filter; can be repeated.")
    parser.add_argument("--limit", type=int, default=None, help="Limit expanded jobs after filters.")
    parser.add_argument("--start-at", default=None, help="Skip jobs until this prompt_id is reached.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = read_json(config_path)

    occupation = config.get("occupation")
    if not occupation:
        raise SystemExit("Config must include occupation.")

    prompt_set = find_prompt_set(occupation, config.get("prompt_set"))
    rows = read_jsonl(prompt_set)

    prompt_ids = as_list(config.get("prompt_ids")) + as_list(args.prompt_id)
    modules = as_list(config.get("modules")) + as_list(args.module)
    slices = as_list(config.get("slices")) + as_list(args.slice)
    limit = args.limit if args.limit is not None else config.get("limit")

    selected = select_prompts(rows, prompt_ids=prompt_ids, modules=modules, slices=slices, limit=limit)
    if args.start_at:
        start_seen = False
        selected_after_start = []
        for item in selected:
            if item.get("prompt_id") == args.start_at:
                start_seen = True
            if start_seen:
                selected_after_start.append(item)
        if not start_seen:
            raise SystemExit(f"--start-at prompt_id not found after filters: {args.start_at}")
        selected = selected_after_start

    if not selected:
        raise SystemExit("No prompts selected.")

    print(f"Config: {config_path}")
    print(f"Prompt set: {prompt_set}")
    print(f"Occupation: {occupation}")
    print(f"Selected prompts: {len(selected)}")

    commands: List[List[str]] = []
    for item in selected:
        options = format_option_strings(job_options(config, item), occupation, item)
        validate_options(item, options)
        commands.append(build_command(occupation, prompt_set, item, options))

    if args.dry_run:
        for index, command in enumerate(commands, start=1):
            print(f"[{index}/{len(commands)}] {' '.join(command)}")
        return

    for index, command in enumerate(commands, start=1):
        prompt_id = command[command.index("--prompt-id") + 1]
        print(f"[{index}/{len(commands)}] running {prompt_id}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
