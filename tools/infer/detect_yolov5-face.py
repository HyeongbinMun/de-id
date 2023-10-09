import os
import sys
import cv2
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utility.params import load_params_yml
from model.det.face.yolov5.yolov5_face import YOLOv5Face

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="object detection model batch inference test script")
    parser.add_argument("--image_path", type=str, default="/workspace/data/image/test.jpg", help="image path")
    parser.add_argument("--config", type=str, default="/workspace/config/params_yolov5face.yml", help="parameter file path")
    option = parser.parse_known_args()[0]

    image_path = option.image_path
    params_yml_path = option.config

    image = cv2.imread(image_path)
    params = load_params_yml(params_yml_path)["infer"]

    model = YOLOv5Face(params)
    result = model.detect(image)
    print(result)