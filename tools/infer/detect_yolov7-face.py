import os
import sys
import cv2
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utility.params import load_params_yml
from model.det.face.yolov7.yolov7_face import YOLOv7Face

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="object detection model batch inference test script")
    parser.add_argument("--image_path", type=str, default="/workspace/data/image/test.jpg", help="image path")
    parser.add_argument("--config", type=str, default="config/params_yolov7face.yml", help="parameter file path")
    option = parser.parse_known_args()[0]

    image_path = option.image_path
    params_yml_path = option.config
    images = None
    if os.path.isdir(image_path):
        image_paths = [os.path.join(image_path, image_name) for image_name in os.listdir(image_path)]
        images = [cv2.imread(image) for image in image_paths]
    else:
        images = [cv2.imread(image_path)]

    params = load_params_yml(params_yml_path)["infer"]
    model = YOLOv7Face(params)
    result = model.detect_batch(images)
    print(result)