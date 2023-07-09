import os
import sys
import cv2
import shutil
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utility.params import load_params_yml
from utility.image.file import save_face_txt, save_bbox_image_xywh, save_bbox_image_yolo
from model.det.face.yolov7.yolov7_face import YOLOv7Face
from utility.image.coordinates import convert_coordinates_to_yolo_format

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="object detection model batch inference test script")
    parser.add_argument("--params_path", type=str, default="/workspace/config/infer_yolov7face.yml", help="parameter file path")
    parser.add_argument("--dataset_dir", type=str, default="/dataset/images_1080_0.1/", help="source image directory")

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
                images.append(cv2.imread(image_path))
                image_name = image_path.split("/")[-1]
                image_names.append(image_name)
                src_image_paths.append(image_path)
            else:
                image_results = model.detect_batch(images)

                for i, faces in enumerate(image_results):
                    if len(faces) > 0:
                        label_path = os.path.join(label_dir, image_names[i].replace(".jpg", ".txt"))
                        yolo_labels = convert_coordinates_to_yolo_format(images[i].shape[1], images[i].shape[0], faces)
                        save_bbox_image_xywh(os.path.join(bbox_image_dir_xywh, image_names[i]), images[i], faces)
                        save_face_txt(label_path, yolo_labels)
                        copy_count += 1
                progress_bar.set_description(f"Processing({dataset_type})")
                images = []
                image_names = []
                src_image_paths = []
