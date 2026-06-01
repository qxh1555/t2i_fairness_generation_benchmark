#!/usr/bin/env python3
"""Compute DINOv2 embeddings and pairwise diversity for image sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(input_path: Path, pattern: str | None, recursive: bool) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Input file is not a supported image: {input_path}")
        return [input_path]

    globber = input_path.rglob if recursive else input_path.glob
    candidates = globber(pattern or "*")
    images = [path for path in candidates if path.is_file() and path.suffix.lower() in IMAGE_EXTS]
    return sorted(images)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def embed_images(
    image_paths: List[Path],
    model_path: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
    model.to(device)
    model.eval()

    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            images = [load_image(path) for path in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            pooled = outputs.pooler_output
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.detach().cpu().numpy().astype(np.float32))
            print(f"[DINOv2] embedded {min(start + batch_size, len(image_paths))}/{len(image_paths)}", flush=True)

    return np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 0), dtype=np.float32)


def diversity_summary(embeddings: np.ndarray, image_paths: List[Path]) -> Dict[str, Any]:
    count = int(embeddings.shape[0])
    dim = int(embeddings.shape[1]) if embeddings.ndim == 2 and count else 0
    summary: Dict[str, Any] = {
        "image_count": count,
        "embedding_dim": dim,
        "metrics": {},
    }
    if count < 2:
        summary["metrics"] = {
            "mean_pairwise_cosine_similarity": None,
            "mean_pairwise_cosine_distance": None,
            "diversity_score": None,
            "nearest_neighbor_similarity_mean": None,
            "nearest_neighbor_similarity_max": None,
        }
        return summary

    sim = embeddings @ embeddings.T
    tri = np.triu_indices(count, k=1)
    pairwise = sim[tri]
    distance = 1.0 - pairwise

    sim_no_diag = sim.copy()
    np.fill_diagonal(sim_no_diag, -np.inf)
    nearest = np.max(sim_no_diag, axis=1)

    summary["metrics"] = {
        "mean_pairwise_cosine_similarity": float(np.mean(pairwise)),
        "std_pairwise_cosine_similarity": float(np.std(pairwise)),
        "min_pairwise_cosine_similarity": float(np.min(pairwise)),
        "max_pairwise_cosine_similarity": float(np.max(pairwise)),
        "mean_pairwise_cosine_distance": float(np.mean(distance)),
        "std_pairwise_cosine_distance": float(np.std(distance)),
        "diversity_score": float(np.mean(distance)),
        "nearest_neighbor_similarity_mean": float(np.mean(nearest)),
        "nearest_neighbor_similarity_max": float(np.max(nearest)),
        "nearest_neighbor_similarity_min": float(np.min(nearest)),
    }
    summary["most_similar_pair"] = pair_record(image_paths, sim, mode="max")
    summary["least_similar_pair"] = pair_record(image_paths, sim, mode="min")
    return summary


def pair_record(image_paths: List[Path], sim: np.ndarray, mode: str) -> Dict[str, Any]:
    count = len(image_paths)
    masked = sim.copy()
    if mode == "max":
        np.fill_diagonal(masked, -np.inf)
        flat_idx = int(np.argmax(masked))
    elif mode == "min":
        np.fill_diagonal(masked, np.inf)
        flat_idx = int(np.argmin(masked))
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    i, j = np.unravel_index(flat_idx, (count, count))
    return {
        "image_a": str(image_paths[i]),
        "image_b": str(image_paths[j]),
        "cosine_similarity": float(sim[i, j]),
        "cosine_distance": float(1.0 - sim[i, j]),
    }


def write_outputs(
    output_dir: Path,
    image_paths: List[Path],
    embeddings: np.ndarray,
    summary: Dict[str, Any],
    model_path: Path,
    input_path: Path,
    pattern: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings)

    records = [
        {
            "index": idx,
            "image_path": str(path),
            "embedding_index": idx,
        }
        for idx, path in enumerate(image_paths)
    ]
    (output_dir / "embedding_records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    payload = {
        "model_path": str(model_path),
        "input_path": str(input_path),
        "pattern": pattern,
        **summary,
    }
    (output_dir / "diversity_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] embeddings: {output_dir / 'embeddings.npy'}")
    print(f"[DONE] records:    {output_dir / 'embedding_records.json'}")
    print(f"[DONE] summary:    {output_dir / 'diversity_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute DINOv2 embedding diversity for an image directory.")
    parser.add_argument("--input", required=True, help="Image file or directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for embeddings and diversity summary.")
    parser.add_argument("--model-path", default="models/dinov2-base", help="Local DINOv2 model directory.")
    parser.add_argument("--pattern", default=None, help="Optional glob pattern, e.g. '*_faces.png'.")
    parser.add_argument("--recursive", action="store_true", help="Recursively collect images.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model_path)
    device = choose_device(args.device)

    image_paths = collect_images(input_path, args.pattern, args.recursive)
    if not image_paths:
        raise SystemExit(f"No images found under {input_path} with pattern={args.pattern!r}")

    print(f"input: {input_path}")
    print(f"model: {model_path}")
    print(f"device: {device}")
    print(f"images: {len(image_paths)}")
    embeddings = embed_images(image_paths, model_path, device, args.batch_size)
    summary = diversity_summary(embeddings, image_paths)
    write_outputs(output_dir, image_paths, embeddings, summary, model_path, input_path, args.pattern)


if __name__ == "__main__":
    main()
