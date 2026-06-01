#!/usr/bin/env python3
"""
Generate images for a target occupation prompt set with DashScope Qwen-Image.

Usage:
  python scripts/generate_images_from_prompt_set.py --occupation nurse --limit 3
  python scripts/generate_images_from_prompt_set.py --occupation CEO --module A --n 2
  python scripts/generate_images_from_prompt_set.py --occupation nurse --prompt-id nurse_A1 --seed 42
  python scripts/generate_images_from_prompt_set.py --occupation firefighter --make-prompt-set --prompt-api

Environment variables:
  DASHSCOPE_API_KEY=...
  DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/api/v1
  DASHSCOPE_IMAGE_MODEL=qwen-image-2.0-pro
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse

try:
    import dashscope
    from dashscope import MultiModalConversation
except ImportError:
    dashscope = None
    MultiModalConversation = None

try:
    import requests
except ImportError:
    requests = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEGATIVE_PROMPT = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，"
    "人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
)
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]


def slugify(text: str) -> str:
    import re

    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def resolve_model_name(args: argparse.Namespace) -> str:
    return args.model or os.environ.get("DASHSCOPE_IMAGE_MODEL") or "qwen-image-2.0"


def run_folder_name(occupation_slug: str, seed: int | None) -> str:
    seed_label = str(seed) if seed is not None else "random"
    return f"{occupation_slug}_seed{seed_label}"


def sample_stem(prompt_id: str, sample_idx: int, total_samples: int) -> str:
    suffix = f"_{sample_idx}" if total_samples > 1 else ""
    return f"{prompt_id}{suffix}"


def find_existing_image(output_dir: Path, prompt_id: str, sample_idx: int, total_samples: int) -> Path | None:
    stem = sample_stem(prompt_id, sample_idx, total_samples)
    for ext in IMAGE_EXTENSIONS:
        path = output_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl_append(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_prompt_set(occupation: str, explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
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
        + ". Use --make-prompt-set to generate it first."
    )


def make_prompt_set(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "generate_prompt_set.py"),
        "--occupation",
        args.occupation,
        "--output-dir",
        str(ROOT / "data" / "generated"),
    ]
    if args.prompt_profile:
        command.extend(["--profile", args.prompt_profile])
    elif args.prompt_api:
        command.append("--use-api")
        if args.pair_pool:
            command.extend(["--pair-pool", args.pair_pool])
        if args.prompt_seed is not None:
            command.extend(["--seed", str(args.prompt_seed)])
    else:
        raise SystemExit("When using --make-prompt-set, provide either --prompt-profile or --prompt-api.")

    if args.no_prompt_review:
        command.append("--no-review-prompts")

    subprocess.run(command, cwd=ROOT, check=True)


def filter_prompts(rows: Iterable[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected = []
    prompt_ids = set(args.prompt_id or [])
    modules = set(args.module or [])
    slices = set(args.slice or [])

    for row in rows:
        if prompt_ids and row.get("prompt_id") not in prompt_ids:
            continue
        if modules and row.get("module") not in modules:
            continue
        if slices and row.get("slice") not in slices:
            continue
        selected.append(row)

    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def api_config(args: argparse.Namespace) -> Dict[str, str]:
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("IMAGE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set DASHSCOPE_API_KEY or pass --api-key.")

    return {
        "api_key": api_key,
        "api_base": (
            args.api_base
            or os.environ.get("DASHSCOPE_API_BASE")
            or "https://dashscope.aliyuncs.com/api/v1"
        ).rstrip("/"),
        "model": resolve_model_name(args),
    }


def response_to_dict(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "to_dict"):
        return response.to_dict()
    try:
        return json.loads(json.dumps(response, ensure_ascii=False))
    except TypeError:
        return {"raw_response": str(response)}


def request_image_once(prompt: str, args: argparse.Namespace, config: Dict[str, str]) -> Dict[str, Any]:
    if dashscope is None or MultiModalConversation is None:
        raise RuntimeError("Missing dependency: dashscope. Run `pip install -r requirements.txt`.")

    dashscope.base_http_api_url = config["api_base"]
    messages = [
        {
            "role": "user",
            "content": [
                {"text": prompt},
            ],
        }
    ]
    call_kwargs = {
        "api_key": config["api_key"],
        "model": config["model"],
        "messages": messages,
        "result_format": "message",
        "stream": False,
        "watermark": args.watermark,
        "prompt_extend": not args.no_prompt_extend,
        "negative_prompt": args.negative_prompt,
        "size": args.size,
    }
    if args.current_seed is not None:
        call_kwargs["seed"] = args.current_seed
    response = MultiModalConversation.call(**call_kwargs)

    response_dict = response_to_dict(response)
    status_code = getattr(response, "status_code", response_dict.get("status_code"))
    if status_code != 200:
        code = getattr(response, "code", response_dict.get("code"))
        message = getattr(response, "message", response_dict.get("message"))
        raise RuntimeError(
            f"DashScope image API error. HTTP={status_code}; code={code}; message={message}; "
            "see https://help.aliyun.com/zh/model-studio/developer-reference/error-code"
        )
    return response_dict


def extract_image_payloads(value: Any) -> List[Dict[str, str]]:
    payloads: List[Dict[str, str]] = []
    if isinstance(value, dict):
        for key in ("url", "image_url", "image"):
            item = value.get(key)
            if isinstance(item, str):
                if item.startswith("http://") or item.startswith("https://"):
                    payloads.append({"type": "url", "value": item})
                elif item.startswith("data:image"):
                    payloads.append({"type": "data_url", "value": item})
        for key in ("b64_json", "base64"):
            item = value.get(key)
            if isinstance(item, str):
                payloads.append({"type": "base64", "value": item})
        for item in value.values():
            payloads.extend(extract_image_payloads(item))
    elif isinstance(value, list):
        for item in value:
            payloads.extend(extract_image_payloads(item))
    return payloads


def extract_prompt_rewrites(value: Any) -> List[Dict[str, str]]:
    """Best-effort extraction of prompt rewrite fields from API responses."""
    rewrite_keys = {
        "prompt",
        "actual_prompt",
        "revised_prompt",
        "expanded_prompt",
        "enhanced_prompt",
        "prompt_extend",
        "prompt_extended",
        "rewritten_prompt",
    }
    rewrites: List[Dict[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = key.lower()
            if lowered in rewrite_keys and isinstance(item, str) and item.strip():
                rewrites.append({"field": key, "value": item})
            elif "prompt" in lowered and isinstance(item, str) and item.strip():
                rewrites.append({"field": key, "value": item})
            else:
                rewrites.extend(extract_prompt_rewrites(item))
    elif isinstance(value, list):
        for item in value:
            rewrites.extend(extract_prompt_rewrites(item))
    return rewrites


def save_raw_response(
    response_dict: Dict[str, Any],
    raw_response_dir: Path | None,
    prompt_id: str,
    sample_idx: int,
) -> str | None:
    if raw_response_dir is None:
        return None
    raw_response_dir.mkdir(parents=True, exist_ok=True)
    path = raw_response_dir / f"{prompt_id}_{sample_idx}_response.json"
    path.write_text(json.dumps(response_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def extension_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return suffix
    return ".png"


def save_payload(payload: Dict[str, str], image_path: Path, timeout: int) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if payload["type"] == "url":
        if requests is None:
            raise RuntimeError("Missing dependency: requests. Run `pip install -r requirements.txt`.")
        response = requests.get(payload["value"], timeout=timeout)
        if response.status_code >= 400:
            raise RuntimeError(f"Failed to download image: HTTP {response.status_code}; url={payload['value']}")
        image_path.write_bytes(response.content)
    elif payload["type"] == "data_url":
        _, encoded = payload["value"].split(",", 1)
        image_path.write_bytes(base64.b64decode(encoded))
    elif payload["type"] == "base64":
        image_path.write_bytes(base64.b64decode(payload["value"]))
    else:
        raise RuntimeError(f"Unsupported image payload type: {payload['type']}")


def generate_and_save_images(
    item: Dict[str, Any],
    args: argparse.Namespace,
    config: Dict[str, str],
    output_dir: Path,
) -> tuple[List[Path], List[Dict[str, Any]], List[Dict[str, Any]]]:
    saved_paths: List[Path] = []
    responses: List[Dict[str, Any]] = []
    sample_records: List[Dict[str, Any]] = []
    for sample_idx in range(1, args.n + 1):
        args.current_seed = args.seed + sample_idx - 1 if args.seed is not None else None
        existing_image = find_existing_image(output_dir, item["prompt_id"], sample_idx, args.n)
        if args.skip_existing and existing_image is not None:
            saved_paths.append(existing_image)
            sample_records.append(
                {
                    "sample_index": sample_idx,
                    "seed": args.current_seed,
                    "image": str(existing_image),
                    "raw_response": None,
                    "prompt_rewrites": [],
                    "status": "skipped_existing",
                }
            )
            print(f"  [{sample_idx}/{args.n}] skip existing {existing_image} seed={args.current_seed}", flush=True)
            continue

        print(f"  [{sample_idx}/{args.n}] request seed={args.current_seed}", flush=True)
        response_dict = request_image_once(item["prompt"], args, config)
        responses.append(response_dict)
        raw_response_path = save_raw_response(
            response_dict,
            args.raw_response_dir,
            item["prompt_id"],
            sample_idx,
        )
        payloads = extract_image_payloads(response_dict)
        if not payloads:
            raise RuntimeError(f"No image URL/base64 payload found in response for {item['prompt_id']}: {response_dict}")

        payload = payloads[0]
        ext = extension_from_url(payload["value"]) if payload["type"] == "url" else ".png"
        image_path = output_dir / f"{sample_stem(item['prompt_id'], sample_idx, args.n)}{ext}"
        save_payload(payload, image_path, args.timeout)
        saved_paths.append(image_path)
        prompt_rewrites = extract_prompt_rewrites(response_dict)
        sample_records.append(
            {
                "sample_index": sample_idx,
                "seed": args.current_seed,
                "image": str(image_path),
                "raw_response": raw_response_path,
                "prompt_rewrites": prompt_rewrites,
                "status": "generated",
            }
        )
        print(f"  [{sample_idx}/{args.n}] saved {image_path}", flush=True)
        if prompt_rewrites:
            for rewrite in prompt_rewrites:
                print(f"    response prompt field {rewrite['field']}: {rewrite['value']}", flush=True)
    return saved_paths, responses, sample_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--occupation", required=True, help="Target occupation, e.g. nurse or taxi_driver")
    parser.add_argument("--prompt-set", default=None, help="Path to an existing prompt_set.jsonl")
    parser.add_argument("--make-prompt-set", action="store_true", help="Generate the prompt set first if needed")
    parser.add_argument("--prompt-api", action="store_true", help="Use LLM API when --make-prompt-set is set")
    parser.add_argument("--prompt-profile", default=None, help="Profile JSON used by --make-prompt-set instead of --prompt-api")
    parser.add_argument("--pair-pool", default=None, help="Optional pair pool forwarded to generate_prompt_set.py")
    parser.add_argument("--no-prompt-review", action="store_true", help="Forward --no-review-prompts to generate_prompt_set.py")
    parser.add_argument("--api-key", default=None, help="DashScope API key; defaults to DASHSCOPE_API_KEY")
    parser.add_argument("--api-base", default=None, help="DashScope API base URL")
    parser.add_argument("--model", default=None, help="DashScope image model; defaults to qwen-image-2.0-pro")
    parser.add_argument("--size", default="2048*2048", help="Image size, e.g. 1024*1024 or 2048*2048")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt passed to DashScope")
    parser.add_argument("--no-prompt-extend", action="store_true", help="Disable DashScope prompt_extend")
    parser.add_argument("--watermark", action="store_true", help="Enable DashScope watermark")
    parser.add_argument("--n", type=int, default=1, help="Images per prompt; implemented as repeated API calls")
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N selected prompts")
    parser.add_argument("--module", action="append", help="Filter by module; can be repeated, e.g. --module A --module C")
    parser.add_argument("--slice", action="append", help="Filter by slice; can be repeated")
    parser.add_argument("--prompt-id", action="append", help="Filter by prompt_id; can be repeated")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "images"), help="Directory for generated images")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing sample images and continue missing seeds")
    parser.add_argument("--dry-run", action="store_true", help="Print selected prompts without calling the image API")
    parser.add_argument("--timeout", type=int, default=180, help="HTTP timeout in seconds")
    parser.add_argument("--seed", type=int, default=None, help="DashScope image seed; with --n > 1, uses seed, seed+1, ...")
    parser.add_argument("--prompt-seed", type=int, default=None, help="Seed forwarded to generate_prompt_set.py when --make-prompt-set --prompt-api is used")
    parser.add_argument("--raw-response-dir", default=None, type=Path, help="Optional directory for per-sample raw API responses")
    args = parser.parse_args()

    if args.n < 1:
        raise SystemExit("--n must be at least 1.")
    if args.seed is not None and not 0 <= args.seed <= 2147483647:
        raise SystemExit("--seed must be in [0, 2147483647].")
    if args.seed is not None and args.seed + args.n - 1 > 2147483647:
        raise SystemExit("--seed + --n - 1 must be <= 2147483647.")

    if args.make_prompt_set:
        make_prompt_set(args)

    prompt_set_path = find_prompt_set(args.occupation, args.prompt_set)
    rows = read_jsonl(prompt_set_path)
    selected = filter_prompts(rows, args)
    if not selected:
        raise SystemExit("No prompts selected.")

    occ_slug = slugify(args.occupation)
    model_slug = slugify(resolve_model_name(args))
    output_dir = Path(args.output_dir) / model_slug / run_folder_name(occ_slug, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    print(f"Prompt set: {prompt_set_path}")
    print(f"Selected prompts: {len(selected)}")
    print(f"Output dir: {output_dir}")

    if args.dry_run:
        for item in selected:
            print(f"{item['prompt_id']}: {item['prompt']}")
        return

    config = api_config(args)
    for index, item in enumerate(selected, start=1):
        prompt_id = item["prompt_id"]
        print(f"[{index}/{len(selected)}] generating {prompt_id}")
        saved_paths, responses, sample_records = generate_and_save_images(item, args, config, output_dir)
        write_jsonl_append(
            manifest_path,
            {
                "prompt_id": prompt_id,
                "prompt": item["prompt"],
                "module": item.get("module"),
                "slice": item.get("slice"),
                "target_occupation": item.get("target_occupation"),
                "model": config["model"],
                "size": args.size,
                "n": args.n,
                "seed": args.seed,
                "sample_seeds": [args.seed + idx for idx in range(args.n)] if args.seed is not None else None,
                "prompt_extend": not args.no_prompt_extend,
                "watermark": args.watermark,
                "images": [str(path) for path in saved_paths],
                "samples": sample_records,
                "responses": responses,
            },
        )

    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
