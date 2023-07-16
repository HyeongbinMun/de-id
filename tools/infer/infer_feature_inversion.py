import os
import sys
import cv2
import argparse
import PIL.Image as Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.models import model_classes
from utility.params import load_params_yml
from utility.image.file import save_tensort2image
from utility.image.coordinates import convert_coordinates_to_yolo_float_list
from model.det.face.yolov5.yolov5_face import YOLOv5Face
from model.deid.feature_inversion.feature_inversion import FeatureInversion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="object detection model batch inference test script")
    parser.add_argument("--face_params_path", type=str, default="/workspace/config/params_yolov5face.yml", help="parameter file path")
    parser.add_argument("--inverter_params_path", type=str, default="/workspace/config/params_inversion_mobileunet.yml", help="parameter file path")
    parser.add_argument("--image_path", type=str, default="/workspace/data/image/test.jpg", help="image path")
    option = parser.parse_known_args()[0]

    image_path = option.image_path
    face_params_path = option.face_params_path
    inverter_params_path = option.inverter_params_path

    face_params = load_params_yml(face_params_path)["infer"]
    inverter_params = load_params_yml(inverter_params_path)["infer"]

    face_model = YOLOv5Face(face_params)
    inversion_model = FeatureInversion(inverter_params)

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    boxes = convert_coordinates_to_yolo_float_list(image.shape[1], image.shape[0], face_model.detect_batch([image])[0])

    inverted_image = inversion_model.inference(image, boxes)
    inverted_image = cv2.cvtColor(inverted_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("inverted_image.jpg", inverted_image)
