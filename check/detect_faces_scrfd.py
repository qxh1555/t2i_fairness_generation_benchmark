import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

# ------------------------------------------------------------
# 关键：必须在 import insightface 之前预加载 CUDA/cuDNN 动态库
# ------------------------------------------------------------
import torch
import onnxruntime as ort

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls(cuda=True, cudnn=True, directory="")

from insightface.app import FaceAnalysis


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_image(image_path: str):
    """
    读取图片。
    使用 cv2.imdecode 而不是 cv2.imread，可以更好地兼容中文路径。
    """
    image_path = str(image_path)
    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def save_image(image_path: str, img):
    """
    保存图片。
    使用 cv2.imencode + tofile，更好地兼容中文路径。
    """
    image_path = str(image_path)
    ext = Path(image_path).suffix
    success, encoded = cv2.imencode(ext, img)
    if not success:
        raise RuntimeError(f"Failed to encode image: {image_path}")
    encoded.tofile(image_path)


def pad_image_for_detection(img, padding_ratio: float, mode: str, color: int):
    if padding_ratio <= 0:
        return img, 0, 0

    height, width = img.shape[:2]
    pad_x = int(round(width * padding_ratio))
    pad_y = int(round(height * padding_ratio))
    if pad_x <= 0 and pad_y <= 0:
        return img, 0, 0

    border_types = {
        "reflect": cv2.BORDER_REFLECT_101,
        "replicate": cv2.BORDER_REPLICATE,
        "constant": cv2.BORDER_CONSTANT,
    }
    if mode not in border_types:
        raise ValueError(f"Unsupported padding mode: {mode}")

    value = [int(color), int(color), int(color)] if mode == "constant" else None
    padded = cv2.copyMakeBorder(
        img,
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        border_types[mode],
        value=value,
    )
    return padded, pad_x, pad_y


def build_face_app(det_size=(640, 640), use_gpu=True, attributes=False):
    """
    初始化 InsightFace FaceAnalysis。

    allowed_modules=['detection'] 表示只加载人脸检测模型 SCRFD，
    不加载人脸识别、年龄性别等模型，速度更快。

    如果你之后想检测年龄/性别，可以把 allowed_modules 改成：
    allowed_modules=None
    或者去掉这个参数。

    注意：SCRFD 本身只做人脸检测，不做人脸属性分类。
    attributes=True 会让 InsightFace 额外加载 buffalo_l 里的属性模型，
    这不是 SCRFD 输出，且通常只包含二值 gender 和 age，不包含 race/skin tone。
    """
    print("torch cuda available:", torch.cuda.is_available())
    print("torch cuda version:", torch.version.cuda)
    print("onnxruntime providers:", ort.get_available_providers())

    if use_gpu and not torch.cuda.is_available():
        print("[WARN] --cpu was not set, but torch.cuda.is_available() is false. Using CPUExecutionProvider.")
        use_gpu = False

    if use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1

    allowed_modules = None if attributes else ["detection"]
    app = FaceAnalysis(name="buffalo_l", allowed_modules=allowed_modules, providers=providers)

    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app


def face_area_ratio(bbox, width: int, height: int) -> float:
    x1, y1, x2, y2 = bbox
    area = max(0, x2 - x1) * max(0, y2 - y1)
    image_area = max(1, width * height)
    return float(area / image_area)


def is_clear_face(face_info, min_score: float, min_area_ratio: float) -> bool:
    return (
        face_info["det_score"] >= min_score
        and face_info["area_ratio"] >= min_area_ratio
        and bool(face_info.get("keypoints"))
    )


def insightface_gender_to_label(value):
    if value is None:
        return None
    # InsightFace genderage convention is commonly 0=female, 1=male.
    return "male" if int(value) == 1 else "female"


def detect_one_image(
    app,
    image_path: str,
    output_dir: str,
    score_threshold: float = 0.5,
    clear_score_threshold: float = 0.5,
    min_clear_face_area_ratio: float = 0.002,
    detect_padding_ratio: float = 0.0,
    detect_padding_mode: str = "reflect",
    detect_padding_color: int = 114,
):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = read_image(str(image_path))
    if img is None:
        print(f"[WARN] Failed to read image: {image_path}")
        return None

    detect_img, pad_x, pad_y = pad_image_for_detection(
        img,
        padding_ratio=detect_padding_ratio,
        mode=detect_padding_mode,
        color=detect_padding_color,
    )
    faces = app.get(detect_img)

    result = {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "face_count": 0,
        "clear_face_count": 0,
        "faces": [],
        "detect_padding": {
            "ratio": detect_padding_ratio,
            "mode": detect_padding_mode,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "applied": bool(pad_x or pad_y),
        },
        "eval_plan_fields": {
            "face_detector": {
                "face_count": 0,
                "clear_face_count": 0,
                "faces": [],
            }
        },
    }

    vis = img.copy()

    valid_faces = []
    for face in faces:
        score = float(face.det_score)
        if score < score_threshold:
            continue
        valid_faces.append(face)

    result["face_count"] = len(valid_faces)

    for face in valid_faces:
        bbox = face.bbox.astype(float).tolist()
        bbox = [bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] - pad_x, bbox[3] - pad_y]
        raw_bbox = [int(round(x)) for x in bbox]
        x1, y1, x2, y2 = bbox

        # 防止坐标越界
        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(img.shape[1] - 1, int(round(x2)))
        y2 = min(img.shape[0] - 1, int(round(y2)))
        if x2 <= x1 or y2 <= y1:
            continue

        idx = len(result["faces"])

        face_info = {
            "face_id": idx,
            "bbox": [x1, y1, x2, y2],
            "bbox_unclipped": raw_bbox,
            "det_score": float(face.det_score),
            "face_visible": True,
        }

        # SCRFD 通常会输出 5 个关键点
        if hasattr(face, "kps") and face.kps is not None:
            keypoints_arr = face.kps.astype(float).copy()
            keypoints_arr[:, 0] -= pad_x
            keypoints_arr[:, 1] -= pad_y
            keypoints = keypoints_arr.tolist()
            face_info["kps"] = keypoints
            face_info["keypoints"] = keypoints
        else:
            face_info["keypoints"] = []

        face_info["area_ratio"] = face_area_ratio(face_info["bbox"], result["width"], result["height"])
        face_info["clear_face"] = is_clear_face(
            face_info,
            min_score=clear_score_threshold,
            min_area_ratio=min_clear_face_area_ratio,
        )

        # Optional InsightFace attributes. These are not SCRFD outputs.
        gender = getattr(face, "gender", None)
        age = getattr(face, "age", None)
        if gender is not None:
            face_info["insightface_gender"] = insightface_gender_to_label(gender)
        if age is not None:
            face_info["insightface_age"] = int(age)

        result["faces"].append(face_info)

        # 画人脸框
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"face {idx}: {face.det_score:.3f}"
        cv2.putText(
            vis,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # 画关键点
        for point in face_info.get("keypoints", []):
            px, py = int(round(point[0])), int(round(point[1]))
            if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                cv2.circle(vis, (px, py), 2, (0, 0, 255), -1)

    result["clear_face_count"] = sum(1 for face in result["faces"] if face.get("clear_face"))
    result["eval_plan_fields"]["face_detector"] = {
        "face_count": result["face_count"],
        "clear_face_count": result["clear_face_count"],
        "faces": [
            {
                "face_id": face["face_id"],
                "bbox": face["bbox"],
                "bbox_unclipped": face.get("bbox_unclipped"),
                "det_score": face["det_score"],
                "keypoints": face.get("keypoints", []),
                "face_visible": face.get("face_visible", True),
                "clear_face": face.get("clear_face", False),
                "area_ratio": face.get("area_ratio"),
            }
            for face in result["faces"]
        ],
    }

    # 保存可视化图片
    out_img_path = output_dir / f"{image_path.stem}_faces{image_path.suffix}"
    save_image(str(out_img_path), vis)

    # 保存 JSON 结果
    out_json_path = output_dir / f"{image_path.stem}_faces.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[OK] {image_path.name}: {result['face_count']} face(s)")
    print(f"     annotated image: {out_img_path}")
    print(f"     json result:      {out_json_path}")

    return result


def collect_images(
    input_path: str,
    exclude_dir: str | None = None,
    include_face_visualizations: bool = False,
    recursive: bool = True,
):
    input_path = Path(input_path)
    exclude_path = Path(exclude_dir).resolve() if exclude_dir else None

    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTS:
            if not include_face_visualizations and input_path.stem.endswith("_faces"):
                raise ValueError(f"Input file looks like a face-visualization output; pass --include-face-visualizations to use it: {input_path}")
            return [input_path]
        else:
            raise ValueError(f"Input file is not a supported image: {input_path}")

    if input_path.is_dir():
        images = []
        candidates = input_path.rglob("*") if recursive else input_path.glob("*")
        for p in candidates:
            if exclude_path is not None:
                try:
                    p.resolve().relative_to(exclude_path)
                    continue
                except ValueError:
                    pass
            if not include_face_visualizations and p.stem.endswith("_faces"):
                continue
            if p.suffix.lower() in IMAGE_EXTS:
                images.append(p)
        return sorted(images)

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect faces using InsightFace SCRFD.")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image path or directory path.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./scrfd_outputs",
        help="Directory to save annotated images and JSON results.",
    )

    parser.add_argument(
        "--det-size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Detection size, e.g. --det-size 640 640",
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Minimum detection confidence score.",
    )

    parser.add_argument(
        "--clear-score-threshold",
        type=float,
        default=0.5,
        help="Minimum face confidence used to count clear_face_count.",
    )

    parser.add_argument(
        "--min-clear-face-area-ratio",
        type=float,
        default=0.002,
        help="Minimum face bbox area / image area used to count clear_face_count.",
    )

    parser.add_argument(
        "--detect-padding-ratio",
        type=float,
        default=0.0,
        help="Add this relative padding around the whole image before SCRFD detection, then map boxes back to the original image. Useful for faces clipped by image borders.",
    )

    parser.add_argument(
        "--detect-padding-mode",
        choices=["reflect", "replicate", "constant"],
        default="reflect",
        help="Border mode used by --detect-padding-ratio.",
    )

    parser.add_argument(
        "--detect-padding-color",
        type=int,
        default=114,
        help="Gray border color used when --detect-padding-mode constant.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force using CPU instead of GPU.",
    )

    parser.add_argument(
        "--attributes",
        action="store_true",
        help="Load optional InsightFace age/gender models. This is not SCRFD and does not provide race/skin tone.",
    )

    parser.add_argument(
        "--include-face-visualizations",
        action="store_true",
        help="Also process files ending with *_faces.*. Disabled by default to avoid re-processing SCRFD visualization outputs.",
    )

    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only process images directly under --input. Use this for generation folders that also contain scrfd/fairface/dinov2 subdirectories.",
    )

    args = parser.parse_args()

    app = build_face_app(
        det_size=tuple(args.det_size),
        use_gpu=not args.cpu,
        attributes=args.attributes,
    )

    image_paths = collect_images(
        args.input,
        exclude_dir=args.output_dir,
        include_face_visualizations=args.include_face_visualizations,
        recursive=not args.non_recursive,
    )

    if len(image_paths) == 0:
        print(f"[WARN] No images found in: {args.input}")
        return

    all_results = []

    for image_path in image_paths:
        result = detect_one_image(
            app=app,
            image_path=str(image_path),
            output_dir=args.output_dir,
            score_threshold=args.score_threshold,
            clear_score_threshold=args.clear_score_threshold,
            min_clear_face_area_ratio=args.min_clear_face_area_ratio,
            detect_padding_ratio=args.detect_padding_ratio,
            detect_padding_mode=args.detect_padding_mode,
            detect_padding_color=args.detect_padding_color,
        )
        if result is not None:
            all_results.append(result)

    # 保存总结果
    output_dir = Path(args.output_dir)
    all_json_path = output_dir / "all_face_results.json"

    with open(all_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print()
    print(f"[DONE] processed images: {len(all_results)}")
    print(f"[DONE] all results saved to: {all_json_path}")


if __name__ == "__main__":
    main()
