import os
import sys
import cv2
import piqa
import numpy
import shutil
import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.models import model_classes
from utility.params import load_params_yml
from utility.image.file import save_tensort2image
from utility.model.region import crop_face, overlay_faces_on_image
from utility.image.dataset import compute_total_area, classify_percentage
from model.deid.dataset.dataset import FaceDetDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, default="/workspace/config/config_d2gan_dna.yml",
                        help="d2gan parameter file path")
    parser.add_argument("--dataset_dir", type=str, default="/dataset/dna/eval/", help="test image directory path")
    parser.add_argument("--output_dir", type=str, default="/mldisk2/deid/dna/d2gan/generated_images", help="image path")
    option = parser.parse_known_args()[0]

    config = load_params_yml(option.config)['infer']
    model_config = config["model"]
    dataset_dir = option.dataset_dir
    output_dir = option.output_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(config["device"])

    image_dir = os.path.join(dataset_dir, "images")
    label_dir = os.path.join(dataset_dir, "labels")
    test_image_dir, test_label_dir = os.path.join(image_dir), os.path.join(label_dir)
    test_dataset = FaceDetDataset(test_image_dir, test_label_dir)
    test_loader = DataLoader(test_dataset,
                             batch_size=16,
                             shuffle=False,
                             num_workers=1,
                             collate_fn=FaceDetDataset.collate_fn)

    # Face Detection Model
    # face_config = model_config["face"]
    # face_model = model_classes["det"]["face"][face_config["model_name"]](face_config)
    # print(f"Face Detection model is successfully loaded.({face_config['model_name']})")

    # D2GAN Generator Model
    generator_config = model_config["generator"]
    generator_input_size = generator_config['input_size']
    channels = generator_config['channels']
    image_shape = (channels, generator_input_size, generator_input_size)
    residual_block_number = generator_config['residual_block_number']
    checkpoint = torch.load(generator_config["model_path"], map_location=device)
    generator_model = model_classes["deid"][generator_config["model_name"]](image_shape, residual_block_number)
    generator_model.load_state_dict(checkpoint['model_state_dict'])
    generator_model = generator_model.to(device)
    transform = transforms.Compose([transforms.ToTensor()])
    generator_model.eval()
    print(f"Face Generator model is successfully loaded.({generator_config['model_name']})")

    # Feature Extraction Model
    feature_config = config["model"]["feature"]
    if feature_config["model_name"] == "ResNet50":
        feature_model = model_classes["feature"][feature_config["model_name"]](backbone="TV_RESNET50", dims=512, pool_param=3).to(device)
        feature_model.load_state_dict(torch.load(feature_config["model_path"]))
    else:
        feature_model = model_classes["feature"][feature_config["model_name"]]().to(device)
        state_dict = torch.load(feature_config["model_path"], map_location=device)
        state_dict = {k.replace("base.", ""): v for k, v in state_dict.items()}
        feature_model.load_state_dict(state_dict, strict=False)
    feature_model.eval()
    feature_model_input_size = feature_config["input_size"]
    print(f"Feature Extraction model is successfully loaded.({feature_config['model_name']})")

    # Face Similarity
    ssim = piqa.SSIM(n_channels=3).cuda()
    psnr = piqa.PSNR().cuda()

    # Feature Similarity
    cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    for batch_idx, (image_paths, label_paths, images, boxes_list) in progress_bar:
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
            batch_orin_faces = [F.interpolate(face.unsqueeze(0), size=(generator_input_size, generator_input_size), mode='bilinear', align_corners=False).squeeze(0) for face in batch_orin_faces]
            batch_orin_faces = torch.stack(batch_orin_faces)

            batch_deid_faces = generator_model(batch_orin_faces)
            deid_full_images = overlay_faces_on_image(images, batch_deid_faces, batch_faces_boxes, batch_images_indices)

            resized_origin_images = F.interpolate(images.clone(), size=(feature_model_input_size, feature_model_input_size))
            resized_deid_images = F.interpolate(deid_full_images.clone(), size=(feature_model_input_size, feature_model_input_size))

            for i, (image_path, deid_full_image) in enumerate(zip(image_paths, deid_full_images)):
                generated_image_path = os.path.join(option.output_dir, image_path.split("/")[-1])
                if not os.path.exists(option.output_dir):
                    os.makedirs(option.output_dir)
                save_tensort2image(deid_full_image, generated_image_path)