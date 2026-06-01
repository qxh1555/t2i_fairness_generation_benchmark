from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]


def attach_auxiliary_results(
    records: List[Dict[str, Any]],
    *,
    face_results_path: Path | None = None,
    face_attribute_results_path: Path | None = None,
) -> List[str]:
    """Attach SCRFD and FairFace outputs to joined benchmark records.

    FairFace attributes are attached as structured auxiliary context and are
    also fused after role-level VLM face_id -> role_id assignment.
    """
    issues: List[str] = []

    face_index: ResultIndex | None = None
    face_attr_index: ResultIndex | None = None

    if face_results_path is not None:
        face_rows = read_result_rows(face_results_path)
        face_index = ResultIndex(face_rows)

    if face_attribute_results_path is not None:
        attr_rows = read_result_rows(face_attribute_results_path)
        face_attr_index = ResultIndex(attr_rows)

    for record in records:
        if face_index is not None:
            result = face_index.find(record["image_path"])
            if result is None:
                issues.append(f"Missing SCRFD face result for image: {record['image_path']}")
            else:
                record["face_detection"] = normalize_face_detection_result(result, face_results_path)

        if face_attr_index is not None:
            result = face_attr_index.find(record["image_path"])
            if result is None:
                issues.append(f"Missing FairFace attribute result for image: {record['image_path']}")
            else:
                record["face_attributes"] = normalize_face_attribute_result(result, face_attribute_results_path)

    return issues


def read_result_rows(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported auxiliary result format: {path}")


class ResultIndex:
    def __init__(self, rows: Iterable[Dict[str, Any]]) -> None:
        self.by_key: Dict[str, Dict[str, Any]] = {}
        self.by_name: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            image_path = row.get("image_path")
            image_name = row.get("image_name") or (Path(image_path).name if image_path else None)
            for key in path_keys(image_path):
                self.by_key[key] = row
            if image_name:
                self.by_name[str(image_name)] = row

    def find(self, image_path: str) -> Dict[str, Any] | None:
        for key in path_keys(image_path):
            if key in self.by_key:
                return self.by_key[key]
        return self.by_name.get(Path(image_path).name)


def path_keys(value: Any) -> List[str]:
    if not value:
        return []
    raw = Path(str(value))
    keys = [str(raw)]
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(ROOT / raw)
    for candidate in candidates:
        try:
            keys.append(str(candidate.resolve()))
        except OSError:
            keys.append(str(candidate.absolute()))
    return list(dict.fromkeys(keys))


def normalize_face_detection_result(result: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
    image_path = Path(result["image_path"])
    annotated_image_path = source_path.parent / f"{image_path.stem}_faces{image_path.suffix}"
    faces = [normalize_face(face) for face in result.get("faces", []) if isinstance(face, dict)]
    return {
        "source_path": str(source_path),
        "image_path": result.get("image_path"),
        "image_name": result.get("image_name"),
        "width": result.get("width"),
        "height": result.get("height"),
        "face_count": int(result.get("face_count", len(faces)) or 0),
        "clear_face_count": int(result.get("clear_face_count", 0) or 0),
        "faces": faces,
        "annotated_image_path": str(annotated_image_path),
        "annotated_image_exists": annotated_image_path.exists(),
    }


def normalize_face_attribute_result(result: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
    faces = [normalize_face_attribute(face) for face in result.get("faces", []) if isinstance(face, dict)]
    return {
        "source_path": str(source_path),
        "image_path": result.get("image_path"),
        "image_name": result.get("image_name"),
        "source_face_count": int(result.get("source_face_count", len(faces)) or 0),
        "source_clear_face_count": int(result.get("source_clear_face_count", 0) or 0),
        "classified_face_count": int(result.get("classified_face_count", len(faces)) or 0),
        "faces": faces,
    }


def normalize_face(face: Dict[str, Any]) -> Dict[str, Any]:
    face_id = face.get("face_id")
    return {
        "face_id": face_label(face_id),
        "source_face_id": face_id,
        "bbox": face.get("bbox"),
        "bbox_unclipped": face.get("bbox_unclipped"),
        "det_score": face.get("det_score"),
        "keypoints": face.get("keypoints", []),
        "clear_face": bool(face.get("clear_face", False)),
        "face_visible": bool(face.get("face_visible", True)),
        "area_ratio": face.get("area_ratio"),
    }


def normalize_face_attribute(face: Dict[str, Any]) -> Dict[str, Any]:
    face_id = face.get("face_id")
    return {
        "face_id": face_label(face_id),
        "source_face_id": face_id,
        "bbox": face.get("bbox"),
        "clear_face": bool(face.get("clear_face", False)),
        "perceived_gender": normalize_unknown(face.get("perceived_gender")),
        "gender_confidence": face.get("gender_confidence"),
        "perceived_race_or_skin_tone": normalize_unknown(face.get("perceived_race_or_skin_tone")),
        "race_confidence": face.get("race_confidence"),
        "perceived_age_group": normalize_unknown(face.get("perceived_age_group")),
        "age_confidence": face.get("age_confidence"),
        "crop_path": face.get("crop_path"),
    }


def face_label(value: Any) -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered.startswith("face_"):
            return lowered
        if lowered.isdigit():
            return f"face_{int(lowered)}"
        return lowered or "face_unknown"
    try:
        return f"face_{int(value)}"
    except (TypeError, ValueError):
        return "face_unknown"


def normalize_unknown(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "unknown"


def face_maps(record: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    detections = {
        face["face_id"]: face
        for face in (record.get("face_detection") or {}).get("faces", [])
        if isinstance(face, dict) and face.get("face_id")
    }
    attributes = {
        face["face_id"]: face
        for face in (record.get("face_attributes") or {}).get("faces", [])
        if isinstance(face, dict) and face.get("face_id")
    }
    return detections, attributes
