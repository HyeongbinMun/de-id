import os
import sys
import cv2
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utility.image.file import save_face_txt, save_bbox_image_yolo

try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("insightface 패키지가 필요합니다: pip install insightface onnxruntime-gpu")
    sys.exit(1)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_detector(gpu_id: int = 0, det_size: int = 640):
    app = FaceAnalysis(
        allowed_modules=["detection"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=gpu_id, det_size=(det_size, det_size))
    return app


def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """(x1,y1,x2,y2) pixel bbox → YOLO normalized (cx, cy, w, h)."""
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))
    return cx, cy, bw, bh


def detect_and_save_labels(
    app,
    image_dir: str,
    label_dir: str,
    vis_dir: str = None,
    skip_existing: bool = True,
    remove_no_face: bool = False,
    conf_threshold: float = 0.5,
):
    os.makedirs(label_dir, exist_ok=True)
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    image_files = sorted(
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )

    no_face_count = 0
    detect_count = 0

    for image_name in tqdm(image_files, desc=f"Detecting faces in {os.path.basename(image_dir)}"):
        stem = os.path.splitext(image_name)[0]
        label_path = os.path.join(label_dir, stem + ".txt")

        if skip_existing and os.path.exists(label_path):
            continue

        image_path = os.path.join(image_dir, image_name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"  [WARN] 이미지 로드 실패: {image_path}")
            continue

        h, w = img.shape[:2]
        faces = app.get(img)

        # confidence threshold 미만 제거
        faces = [f for f in faces if f.det_score >= conf_threshold]

        if len(faces) == 0:
            no_face_count += 1
            if remove_no_face:
                os.remove(image_path)
            continue

        yolo_labels = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            cx, cy, bw, bh = bbox_to_yolo(x1, y1, x2, y2, w, h)
            yolo_labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        save_face_txt(label_path, yolo_labels)
        detect_count += 1

        if vis_dir:
            save_bbox_image_yolo(
                os.path.join(vis_dir, image_name), img.copy(), yolo_labels
            )

    print(f"  → 검출 성공: {detect_count}, 얼굴 미검출: {no_face_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="InsightFace(SCRFD) 기반 얼굴 검출 → YOLO 포맷 라벨 생성"
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="데이터셋 루트 디렉토리"
    )
    parser.add_argument(
        "--det_size", type=int, default=640, help="검출기 입력 크기 (default: 640)"
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=0.5, help="검출 confidence 임계값"
    )
    parser.add_argument(
        "--eval", action="store_true", help="평가 데이터셋 모드 (images/ 하위 split 없음)"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="검출 결과 시각화 이미지 저장"
    )
    parser.add_argument(
        "--no_skip", action="store_true", help="이미 존재하는 라벨도 다시 생성"
    )
    parser.add_argument(
        "--remove_no_face", action="store_true",
        help="얼굴이 검출되지 않은 이미지 삭제",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="GPU ID (default: 0)"
    )

    option = parser.parse_args()

    app = build_detector(gpu_id=option.gpu_id, det_size=option.det_size)

    dataset_dir = option.dataset_dir
    if option.eval:
        dataset_types = [""]
    else:
        dataset_types = sorted(os.listdir(os.path.join(dataset_dir, "images")))

    for dataset_type in dataset_types:
        image_dir = os.path.join(dataset_dir, "images", dataset_type)
        label_dir = os.path.join(dataset_dir, "labels", dataset_type)
        vis_dir = (
            os.path.join(dataset_dir, "yolo", dataset_type)
            if option.visualize
            else None
        )

        print(f"\n[{dataset_type or 'eval'}] 처리 중: {image_dir}")
        detect_and_save_labels(
            app,
            image_dir,
            label_dir,
            vis_dir=vis_dir,
            skip_existing=not option.no_skip,
            remove_no_face=option.remove_no_face,
            conf_threshold=option.conf_threshold,
        )

    print("\n완료.")
