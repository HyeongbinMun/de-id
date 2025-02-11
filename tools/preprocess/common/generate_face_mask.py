import os
import sys
import cv2
import shutil
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utility.image.file import save_mask_from_yolo

def get_image_path(label_path, image_dir):
    # 레이블 파일 이름에서 확장자를 제거합니다.
    base_name = os.path.splitext(os.path.basename(label_path))[0]
    # 해당하는 이미지 파일 경로를 반환합니다.
    return os.path.join(image_dir, base_name + ".jpg")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="mask images save.")
    parser.add_argument("--dataset_dir", type=str, default="/dataset/face/vggface2hq_refine/", help="source image directory")

    option = parser.parse_known_args()[0]

    dataset_dir = option.dataset_dir
    dataset_types = os.listdir(os.path.join(dataset_dir, "images"))

    for dataset_type in dataset_types:
        image_dir = os.path.join(dataset_dir, "images", dataset_type)
        label_dir = os.path.join(dataset_dir, "labels", dataset_type)
        mask_dir = os.path.join(dataset_dir, "masks", dataset_type)

        os.makedirs(mask_dir, exist_ok=True)

        label_paths = [os.path.join(label_dir, label_name) for label_name in sorted(os.listdir(label_dir))]

        progress_bar = tqdm(label_paths, total=len(label_paths))
        for label_path in progress_bar:
            image_path = get_image_path(label_path, image_dir) # 이미지 경로를 레이블 파일 이름을 기반으로 얻습니다.
            filename = os.path.basename(image_path)

            # 이미지 로드
            img = cv2.imread(image_path)

            # YOLO 레이블 로드
            with open(label_path, 'r') as f:
                yolo_labels = f.readlines()

            # 마스크 저장 경로
            mask_save_path = os.path.join(mask_dir, filename)

            # 해당 mask_path에 라벨 파일이 이미 있다면, 처리를 스킵합니다.
            if os.path.exists(mask_save_path):
                continue

            # 마스크 생성 및 저장
            save_mask_from_yolo(img, yolo_labels, mask_save_path)
