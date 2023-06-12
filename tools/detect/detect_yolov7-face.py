import os
import sys
import cv2
import torch
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.det.face.yolov7.yolov7_face import YOLOv7Face

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="object detection model batch inference test script")
    parser.add_argument("--image_path", type=str, default="/workspace/data/image/test.jpg", help="image path")
    parser.add_argument("--params", type=str, default="/workspace/config/test_params.yml", help="parameter file path")
    option = parser.parse_known_args()[0]

    image_path = option.image_path
    image = cv2.imread(image_path)

    params = {
        "model_path": "/workspace/model/weights/yolov7-w6-face.pt",
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "image_size": 640,
        "conf_thresh": 0.3,
        "iou_thresh": 0.5
    }
    model = YOLOv7Face(params)
    result = model.detect(image)
    print(result)