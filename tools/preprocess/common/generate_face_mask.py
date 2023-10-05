import os
import sys
import cv2
import shutil
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utility.image.file import save_mask_from_yolo

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

        image_paths = [os.path.join(image_dir, image_name) for image_name in sorted(os.listdir(image_dir))]
        label_paths = [os.path.join(label_dir, image_name) for image_name in sorted(os.listdir(label_dir))]

        images = []
        image_names = []
        src_image_paths = []
        progress_bar = tqdm(zip(label_paths, image_paths), total=len(image_paths))
        for label_path, image_path in progress_bar:
            filename = os.path.basename(image_path)

            # 이미지 로드
            img = cv2.imread(image_path)

            # YOLO 레이블 로드
            with open(label_path, 'r') as f:
                yolo_labels = f.readlines()

            # 마스크 저장 경로
            mask_save_path = os.path.join(mask_dir, filename)

            # 마스크 생성 및 저장
            save_mask_from_yolo(img, yolo_labels, mask_save_path)

