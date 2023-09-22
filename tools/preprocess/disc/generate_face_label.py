import os
import sys
import cv2
import shutil
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utility.params import load_params_yml
from utility.image.file import save_face_txt, save_bbox_image_xywh, save_bbox_image_yolo
from model.det.face.yolov7.yolov7_face import YOLOv7Face
from utility.image.coordinates import convert_coordinates_to_yolo_format

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detects faces in images under the given directory in the format yolo dataset and generates labels under the directory labels in the format yolo by using the yolov7face model.")
    parser.add_argument("--params_path", type=str, default="/workspace/config/params_yolov7face.yml", help="parameter file path")
    parser.add_argument("--dataset_dir", type=str, default="/dataset/disc21", help="source image directory")

    option = parser.parse_known_args()[0]

    dataset_dir = option.dataset_dir
    params_yml_path = option.params_path
    params = load_params_yml(params_yml_path)["infer"]
    model = YOLOv7Face(params)

    image_dirs = [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    for i, image_dir in enumerate(image_dirs):
        image_dir_name = image_dir.split("/")[-1]
        label_dir = image_dir + "_label"
        image_bbox_dir = image_dir + "_bbox"

        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(image_bbox_dir, exist_ok=True)

        image_paths = [os.path.join(image_dir, image_name) for image_name in sorted(os.listdir(image_dir))]
        image_count = len(image_paths)

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
            elif len(images) == (i + 1) and len(images) > 0:
                image_results = model.detect_batch(images)
                for j, faces in enumerate(image_results):
                    if len(faces) > 0:
                        label_path = os.path.join(label_dir, image_names[j].replace(".jpg", ".txt"))
                        yolo_labels = convert_coordinates_to_yolo_format(images[j].shape[1], images[j].shape[0], faces)
                        save_bbox_image_xywh(os.path.join(image_bbox_dir, image_names[j]), images[j], faces)
                        save_face_txt(label_path, yolo_labels)
                progress_bar.set_description(f"Processing({image_dir_name})")
            else:
                image_results = model.detect_batch(images)
                for j, faces in enumerate(image_results):
                    if len(faces) > 0:
                        label_path = os.path.join(label_dir, image_names[j].replace(".jpg", ".txt"))
                        yolo_labels = convert_coordinates_to_yolo_format(images[j].shape[1], images[j].shape[0], faces)
                        save_bbox_image_xywh(os.path.join(image_bbox_dir, image_names[j]), images[j], faces)
                        save_face_txt(label_path, yolo_labels)
                progress_bar.set_description(f"Processing({image_dir_name})")
                images = []
                image_names = []
                src_image_paths = []