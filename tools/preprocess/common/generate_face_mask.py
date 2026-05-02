import os
import sys
import cv2
import numpy as np
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def yolo_to_pixel_bbox(cx, cy, bw, bh, img_w, img_h):
    """YOLO normalized (cx, cy, w, h) → pixel (left, top, right, bottom)."""
    x_center = cx * img_w
    y_center = cy * img_h
    width = bw * img_w
    height = bh * img_h

    left = int(x_center - width / 2)
    top = int(y_center - height / 2)
    right = int(x_center + width / 2)
    bottom = int(y_center + height / 2)

    return left, top, right, bottom


def apply_padding(left, top, right, bottom, img_w, img_h, padding_ratio=0.0):
    """bbox 주변에 padding_ratio 만큼 여백을 추가한다."""
    if padding_ratio <= 0.0:
        return (
            max(0, left),
            max(0, top),
            min(img_w, right),
            min(img_h, bottom),
        )

    bw = right - left
    bh = bottom - top
    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)

    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(img_w, right + pad_x),
        min(img_h, bottom + pad_y),
    )


def generate_mask(img_shape, yolo_labels, padding_ratio=0.0):
    """YOLO 라벨 리스트로부터 바이너리 마스크(uint8, 0 or 255)를 생성한다."""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for yolo_label in yolo_labels:
        parts = yolo_label.strip().split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = map(float, parts[:5])
        left, top, right, bottom = yolo_to_pixel_bbox(cx, cy, bw, bh, w, h)
        left, top, right, bottom = apply_padding(
            left, top, right, bottom, w, h, padding_ratio
        )
        mask[top:bottom, left:right] = 255

    return mask


def find_image_path(base_name, image_dir):
    """확장자를 자동으로 탐색하여 이미지 경로를 반환한다."""
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(image_dir, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def generate_masks_from_labels(
    image_dir: str,
    label_dir: str,
    mask_dir: str,
    padding_ratio: float = 0.0,
    skip_existing: bool = True,
):
    os.makedirs(mask_dir, exist_ok=True)

    label_files = sorted(
        f for f in os.listdir(label_dir) if f.endswith(".txt")
    )

    success_count = 0
    skip_count = 0

    for label_name in tqdm(label_files, desc=f"Generating masks in {os.path.basename(mask_dir)}"):
        stem = os.path.splitext(label_name)[0]

        image_path = find_image_path(stem, image_dir)
        if image_path is None:
            continue

        image_ext = os.path.splitext(os.path.basename(image_path))[1]
        mask_save_path = os.path.join(mask_dir, stem + image_ext)

        if skip_existing and os.path.exists(mask_save_path):
            skip_count += 1
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"  [WARN] 이미지 로드 실패: {image_path}")
            continue

        label_path = os.path.join(label_dir, label_name)
        with open(label_path, "r") as f:
            yolo_labels = f.readlines()

        mask = generate_mask(img.shape, yolo_labels, padding_ratio)
        cv2.imwrite(mask_save_path, mask)
        success_count += 1

    print(f"  → 생성: {success_count}, 스킵(기존): {skip_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO 라벨 기반 얼굴 마스크 이미지 생성"
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="데이터셋 루트 디렉토리"
    )
    parser.add_argument(
        "--padding", type=float, default=0.0,
        help="bbox 주변 패딩 비율 (0.0 = 없음, 0.1 = 10%% 확장)"
    )
    parser.add_argument(
        "--eval", action="store_true", help="평가 데이터셋 모드 (images/ 하위 split 없음)"
    )
    parser.add_argument(
        "--no_skip", action="store_true", help="이미 존재하는 마스크도 다시 생성"
    )

    option = parser.parse_args()

    dataset_dir = option.dataset_dir
    if option.eval:
        dataset_types = [""]
    else:
        dataset_types = sorted(os.listdir(os.path.join(dataset_dir, "images")))

    for dataset_type in dataset_types:
        image_dir = os.path.join(dataset_dir, "images", dataset_type)
        label_dir = os.path.join(dataset_dir, "labels", dataset_type)
        mask_dir = os.path.join(dataset_dir, "masks", dataset_type)

        if not os.path.isdir(label_dir):
            print(f"  [SKIP] 라벨 디렉토리 없음: {label_dir}")
            continue

        print(f"\n[{dataset_type or 'eval'}] 처리 중: {label_dir} → {mask_dir}")
        generate_masks_from_labels(
            image_dir,
            label_dir,
            mask_dir,
            padding_ratio=option.padding,
            skip_existing=not option.no_skip,
        )

    print("\n완료.")
