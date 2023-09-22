import os
import sys
import cv2
import shutil
import argparse
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utility.params import load_params_yml
from utility.image.file import save_face_txt, save_bbox_image_xywh, save_bbox_image_yolo
from model.det.face.yolov7.yolov7_face import YOLOv7Face
from utility.image.coordinates import convert_coordinates_to_yolo_format

def generate_face_mask(image, faces):
    mask = np.zeros_like(image)
    for face in faces:
        x, y, w, h = map(int, face[3:7])  # Convert x, y, w, h values to integers
        mask[y:y+h, x:x+w] = [255, 255, 255]
    return mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detects faces in images under the given directory in the format yolo dataset and generates labels under the directory labels in the format yolo by using the yolov7face model.")
    parser.add_argument("--params_path", type=str, default="/workspace/config/params_yolov7face.yml", help="parameter file path")
    parser.add_argument("--dataset_dir", type=str, default="/data/face/instance/", help="source image directory")

    option = parser.parse_known_args()[0]

    dataset_dir = option.dataset_dir
    dataset_types = os.listdir(os.path.join(dataset_dir, "images"))
    params_yml_path = option.params_path
    params = load_params_yml(params_yml_path)["infer"]
    model = YOLOv7Face(params)

    for dataset_type in dataset_types:
        image_dir = os.path.join(dataset_dir, "images", dataset_type)
        label_dir = os.path.join(dataset_dir, "labels", dataset_type)
        bbox_image_dir_xywh = os.path.join(dataset_dir, "xywh", dataset_type)
        bbox_image_dir_yolo = os.path.join(dataset_dir, "yolo", dataset_type)

        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(bbox_image_dir_xywh, exist_ok=True)
        os.makedirs(bbox_image_dir_yolo, exist_ok=True)

        image_paths = [os.path.join(image_dir, image_name) for image_name in sorted(os.listdir(image_dir))]
        image_count = len(image_paths)
        copy_count = 0

        images = []
        image_names = []
        src_image_paths = []
        progress_bar = tqdm(enumerate(image_paths), total=len(image_paths))
        for i, image_path in progress_bar:
            if len(images) < params["batch_size"]:
                image_size = params["image_size"]
                resized_image = cv2.resize(cv2.imread(image_path), (image_size, image_size))
                images.append(resized_image)
                image_name = image_path.split("/")[-1]
                image_names.append(image_name)
                src_image_paths.append(image_path)
            else:
                image_results = model.detect_batch(images)

                for j, faces in enumerate(image_results):
                    if len(faces) > 0:
                        # 마스크 이미지 생성
                        mask_image = generate_face_mask(images[j], faces)
                        mask_image_path = os.path.join(dataset_dir, "masks", dataset_type, image_names[j])
                        os.makedirs(os.path.dirname(mask_image_path), exist_ok=True)
                        cv2.imwrite(mask_image_path, mask_image)
                        copy_count += 1
                    else:
                        image_path = os.path.join(image_dir, image_names[j])
                        os.remove(image_path)
                progress_bar.set_description(f"Processing({dataset_type})")
                images = []
                image_names = []
                src_image_paths = []

    if len(images) > 0:
        image_results = model.detect_batch(images)

        for j, faces in enumerate(image_results):
            if len(faces) > 0:
                # 마스크 이미지 생성
                mask_image = generate_face_mask(images[j], faces)
                mask_image_path = os.path.join(dataset_dir, "masks", dataset_type, image_names[j])
                os.makedirs(os.path.dirname(mask_image_path), exist_ok=True)
                cv2.imwrite(mask_image_path, mask_image)
                copy_count += 1
            else:
                image_path = os.path.join(image_dir, image_names[j])
                os.remove(image_path)