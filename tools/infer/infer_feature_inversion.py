import os
import shutil
import sys
import cv2
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.deid.dataset.dataset import FaceDetDataset
from model.det.face.yolov7.yolov7_face import YOLOv7Face

from config.models import model_classes
from utility.params import load_params_yml
from utility.model.region import overlay_faces_on_image, crop_face
from utility.image.coordinates import convert_coordinates_to_yolo_format
from utility.image.file import save_face_txt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="object detection model batch inference test script")
    parser.add_argument("--face_config", type=str, default="/workspace/config/config_yolov7face.yml", help="parameter file path")
    parser.add_argument("--inverter_config", type=str, default="/workspace/config/config_inversion_resnet50unet_dna.yml", help="parameter file path")
    parser.add_argument("--batch_size", type=int, default=16, help="inference batch size")
    parser.add_argument("--image_dir", type=str, default="/workspace/output", help="image path")
    parser.add_argument("--output_dir", type=str, default="/workspace/output_", help="image path")
    option = parser.parse_known_args()[0]

    image_dir = option.image_dir
    output_dir = option.output_dir
    image_paths = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir) if image_name.lower().endswith('.jpg')]
    face_config = option.face_config
    inverter_config = option.inverter_config
    batch_size = option.batch_size
    tmp_dir = os.path.join(output_dir, "tmp")

    tmp_images_dir = os.path.join(tmp_dir, "images")
    tmp_labels_dir = os.path.join(tmp_dir, "labels")
    os.makedirs(tmp_images_dir, exist_ok=True)
    os.makedirs(tmp_labels_dir, exist_ok=True)

    model_config = load_params_yml(inverter_config)["infer"]
    face_params = model_config["model"]["face"]
    inverteion_model_config = model_config["model"]["inverter"]

    face_model = YOLOv7Face(face_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(model_config["device"])

    checkpoint = torch.load(inverteion_model_config["model_path"], map_location=device)
    inversion_model = model_classes["deid"][inverteion_model_config["model_name"]]()
    inversion_model.load_state_dict(checkpoint['model_state_dict'])
    inversion_model = inversion_model.to(device)
    transform = transforms.Compose([transforms.ToTensor()])
    inversion_input_size = inverteion_model_config['input_size']
    inversion_model.eval()

    # print(image_paths)
    images = []
    image_names = []
    src_image_paths = []
    progress_bar = tqdm(enumerate(image_paths), total=len(image_paths))
    for i, image_path in progress_bar:
        image_size = face_params["image_size"]
        resized_image = cv2.resize(cv2.imread(image_path), (image_size, image_size))
        images.append(resized_image)
        image_name = image_path.split("/")[-1]
        image_names.append(image_name)
        src_image_paths.append(image_path)
        if len(images) == face_params["batch_size"] or (len(image_paths) % face_params["batch_size"] == len(image_paths)):
            image_results = face_model.detect_batch(images)

            for j, faces in enumerate(image_results):
                if len(faces) > 0:
                    tmp_image_path = os.path.join(tmp_images_dir, image_names[j])
                    label_path = os.path.join(tmp_labels_dir, image_names[j].replace(".jpg", ".txt"))
                    yolo_labels = convert_coordinates_to_yolo_format(images[j].shape[1], images[j].shape[0], faces)
                    save_face_txt(label_path, yolo_labels)
                    shutil.copy(image_path, tmp_image_path)
                else:
                    image_path = os.path.join(image_dir, image_names[j])
                    os.remove(image_path)
            progress_bar.set_description(f"Face Detection Processing..")
            images = []
            image_names = []
            src_image_paths = []


    test_dataset = FaceDetDataset(tmp_images_dir, tmp_labels_dir)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=1,
                             collate_fn=FaceDetDataset.collate_fn)

    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    for batch_idx, (image_path, label_paths, images, boxes_list, image_size) in progress_bar:
        images = images.clone().to(device)

        batch_orin_faces = []
        batch_images_indices = []
        batch_faces_boxes = []

        for i, boxes in enumerate(boxes_list):
            real_orin_faces, faces_index, face_boxes = crop_face(images, boxes, i)
            batch_orin_faces.extend(real_orin_faces)
            batch_images_indices.extend(faces_index)
            batch_faces_boxes.extend(face_boxes)

        if len(batch_orin_faces) > 0:
            batch_orin_faces = [F.interpolate(face.unsqueeze(0), size=(inversion_input_size, inversion_input_size), mode='bilinear', align_corners=False).squeeze(0) for face in batch_orin_faces]
            batch_orin_faces = torch.stack(batch_orin_faces)

            batch_deid_faces = inversion_model(batch_orin_faces)
            deid_full_images = overlay_faces_on_image(images, batch_deid_faces, batch_faces_boxes, batch_images_indices)

            for d, deid_full_image in enumerate(deid_full_images):
                deid_image_path = os.path.join(output_dir, image_path[d].split("/")[-1])
                pil_image = Image.fromarray((deid_full_image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil_image = pil_image.resize(image_size[d])
                pil_image.save(deid_image_path, "JPEG")
            progress_bar.set_description(f"Deid Full Image Generating..")

    # shutil.rmtree(tmp_dir)
