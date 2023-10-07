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
from utility.model.region import crop_face, overlay_faces_on_image
from model.deid.dataset.dataset import FaceDetDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="object detection model batch inference test script")
    parser.add_argument("--config", type=str, default="/workspace/config/config_inversion_mobileunet_icd.yml", help="invertsion parameter file path")
    parser.add_argument("--dataset_dir", type=str, default="/dataset/widerface/inversion/", help="test image directory path")
    parser.add_argument("--output_dir", type=str, default="/workspace/output/", help="image path")
    option = parser.parse_known_args()[0]

    config = load_params_yml(option.config)['infer']
    model_config = config["model"]
    dataset_dir = option.dataset_dir
    output_dir = option.output_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(config["device"])

    image_dir = os.path.join(dataset_dir, "images")
    label_dir = os.path.join(dataset_dir, "labels")
    test_image_dir, test_label_dir = os.path.join(image_dir, "test"), os.path.join(label_dir, "test")
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

    # Feature Inversion Model
    inversion_config = model_config["inverter"]
    checkpoint = torch.load(inversion_config["model_path"])
    inversion_model = model_classes["deid"][inversion_config["model_name"]]().to(device)
    inversion_model.load_state_dict(checkpoint['model_state_dict'])
    transform = transforms.Compose([transforms.ToTensor()])
    inversion_input_size = inversion_config['input_size']
    inversion_model.eval()
    print(f"Face Generator model is successfully loaded.({inversion_config['model_name']})")

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

    total_ssim = 0.0
    total_psnr = 0.0
    total_cossim = 0.0

    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    for batch_idx, (image_path, label_path, images, boxes_list) in progress_bar:
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

            resized_origin_images = F.interpolate(images.clone(), size=(feature_model_input_size, feature_model_input_size))
            resized_deid_images = F.interpolate(deid_full_images.clone(), size=(feature_model_input_size, feature_model_input_size))
            orin_full_feature = feature_model(resized_origin_images)
            deid_full_feature = feature_model(resized_deid_images)

            batch_orin_faces = torch.clamp(batch_orin_faces, 0, 1)
            batch_deid_faces = torch.clamp(batch_deid_faces, 0, 1)
            orin_deid_psnr = psnr(batch_orin_faces, batch_deid_faces)
            orin_deid_ssim = ssim(batch_orin_faces, batch_deid_faces)
            feature_cossim = cosine_similarity(orin_full_feature, deid_full_feature).mean()

            total_ssim += orin_deid_ssim.item()
            total_psnr += orin_deid_psnr.item()
            total_cossim += feature_cossim.item()

    avg_ssim = total_ssim / len(test_loader)
    avg_psnr = total_psnr / len(test_loader)
    avg_cossim = total_cossim / len(test_loader)

    print(f"Average Face Image SSIM: {avg_ssim:.4f}")
    print(f"Average Face Image PSNR: {avg_psnr:.4f}")
    print(f"Average Full Image Cosine Similarity: {avg_cossim:.4f}")