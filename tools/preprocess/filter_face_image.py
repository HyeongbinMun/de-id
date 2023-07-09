import os
import shutil
import sys
import cv2
import argparse

import numpy
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utility.params import load_params_yml
from model.det.face.yolov7.yolov7_face import YOLOv7Face

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="object detection model batch inference test script")
    parser.add_argument("--source_dir", type=str, default="/hdd/dna_db_frame", help="source image directory")
    parser.add_argument("--target_dir", type=str, default="/hdd/dna_db_frame_face", help="target image directory")
    parser.add_argument("--params_path", type=str, default="/workspace/config/model_params.yml", help="parameter file path")

    option = parser.parse_known_args()[0]

    input_directory = option.source_dir
    output_directory = option.target_dir
    params_yml_path = option.params_path

    os.makedirs(output_directory, exist_ok=True)

    params = load_params_yml(params_yml_path)["det"]["face"]["yolov7"]
    model = YOLOv7Face(params)

    image_paths = [os.path.join(input_directory, image_name) for image_name in os.listdir(input_directory)]

    image_count = len(image_paths)
    copy_count = 0

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        image_name = image_path.split("/")[-1]

        faces = model.detect_batch([image])

        if len(faces) > 0:
            output_path = os.path.join(output_directory, image_name)
            shutil.copy(image_path, output_path)
            copy_count += 1
            print(f"\r[{i: 6} / {image_count: 6}]\t Saved image with faces: {output_path} - ({copy_count: 6})", end="")

    print("Ended.")