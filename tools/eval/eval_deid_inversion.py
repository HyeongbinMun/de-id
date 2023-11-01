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
    parser.add_argument("--config", type=str, default="/workspace/config/config_inversion_mobileunet_dna.yml", help="invertsion parameter file path")
    parser.add_argument("--batch_size", type=int, default=16, help="inference batch size")
    parser.add_argument("--dataset_dir", type=str, default="/dataset/dna_frame/eval/", help="test image directory path")
    parser.add_argument("--output_dir", type=str, default="/dataset/result/mobileunet", help="image path")
    parser.add_argument('--save', action='store_true', help='If you want to save result image, please give this argument.')

    option = parser.parse_known_args()[0]

    config = load_params_yml(option.config)['infer']
    model_config = config["model"]
    dataset_dir = option.dataset_dir
    output_dir = option.output_dir
    is_result_save = option.save
    batch_size = option.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(config["device"])

    image_dir = os.path.join(dataset_dir, "images")
    label_dir = os.path.join(dataset_dir, "labels")
    test_image_dir, test_label_dir = os.path.join(image_dir), os.path.join(label_dir)
    test_dataset = FaceDetDataset(test_image_dir, test_label_dir)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=1,
                             collate_fn=FaceDetDataset.collate_fn)

    # Face Detection Model
    # face_config = model_config["face"]
    # face_model = model_classes["det"]["face"][face_config["model_name"]](face_config)
    # print(f"Face Detection model is successfully loaded.({face_config['model_name']})")

    # Feature Inversion Model
    inversion_config = model_config["inverter"]
    checkpoint = torch.load(inversion_config["model_path"], map_location=device)
    inversion_model = model_classes["deid"][inversion_config["model_name"]]()
    inversion_model.load_state_dict(checkpoint['model_state_dict'])
    inversion_model = inversion_model.to(device)
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

    metrics_by_bbox_ratio = {
        "10": {"ssim": 0, "psnr": 0, "cossim": 0, "count": 0},
        "30": {"ssim": 0, "psnr": 0, "cossim": 0, "count": 0},
        "50": {"ssim": 0, "psnr": 0, "cossim": 0, "count": 0},
        "70": {"ssim": 0, "psnr": 0, "cossim": 0, "count": 0},
        "70above": {"ssim": 0, "psnr": 0, "cossim": 0, "count": 0},
    }
    for ratio in metrics_by_bbox_ratio.keys():
        os.makedirs(os.path.join(output_dir, ratio), exist_ok=True)

    image_count = 0
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    for batch_idx, (image_paths, label_paths, images, boxes_list, image_size) in progress_bar:
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

            for b, boxes in enumerate(boxes_list):
                try:
                    image_w, image_h = images[b].shape[2], images[b].shape[1]
                    box_list = []
                    for box in boxes:
                        x_center, y_center, width, height = box[1], box[2], box[3], box[4]
                        x1 = int((x_center - width / 2) * image_w)
                        y1 = int((y_center - height / 2) * image_h)
                        x2 = int((x_center + width / 2) * image_w)
                        y2 = int((y_center + height / 2) * image_h)
                        box_list.append((x1, y1, x2, y2))
                    bbox_area_ratio = compute_total_area(box_list, image_w, image_h)

                    category = classify_percentage(bbox_area_ratio)

                    orin_faces = batch_orin_faces[b].unsqueeze(0)
                    deid_faces = batch_deid_faces[b].unsqueeze(0)

                    orin_deid_psnr = psnr(orin_faces, deid_faces)
                    orin_deid_ssim = ssim(orin_faces, deid_faces)

                    feature_cossim = cosine_similarity(orin_full_feature[b].unsqueeze(0), deid_full_feature[b].unsqueeze(0)).mean()

                    total_ssim += orin_deid_ssim.item()
                    total_psnr += orin_deid_psnr.item()
                    total_cossim += feature_cossim.item()

                    single_orin_deid_ssim = orin_deid_ssim.item()
                    single_orin_deid_psnr = orin_deid_psnr.item()
                    single_orin_deid_cossim = feature_cossim.item()

                    metrics_by_bbox_ratio[category]["ssim"] += single_orin_deid_ssim
                    metrics_by_bbox_ratio[category]["psnr"] += single_orin_deid_psnr
                    metrics_by_bbox_ratio[category]["cossim"] += single_orin_deid_cossim
                    metrics_by_bbox_ratio[category]["count"] += 1

                    image_path = os.path.join(output_dir, category, image_paths[b].split("/")[-1])
                    if is_result_save:
                        save_tensort2image(deid_full_images[b], image_path)
                    image_count += 1
                except:
                    pass

    avg_ssim = total_ssim / image_count
    avg_psnr = total_psnr / image_count
    avg_cossim = total_cossim / image_count

    categories = ["10%", "30%", "50%", "70%", "70above", "average"]
    print("         \t".join([""] + categories))
    ssim_values = [
        metrics_by_bbox_ratio[cat]["ssim"] / metrics_by_bbox_ratio[cat]["count"] if metrics_by_bbox_ratio[cat]["count"] > 0 else "#" for cat in
        ["10", "30", "50", "70", "70above"]
    ]
    print("-----------------------------------------------------------")
    ssim_values.append(avg_ssim)
    print("\t".join(["ssim     "] + [f"{val:.4f}" if isinstance(val, float) else "#" for val in ssim_values]))

    psnr_values = [
        metrics_by_bbox_ratio[cat]["psnr"] / metrics_by_bbox_ratio[cat]["count"] if metrics_by_bbox_ratio[cat]["count"] > 0 else "#" for cat in
        ["10", "30", "50", "70", "70above"]
    ]
    psnr_values.append(avg_psnr)
    print("\t".join(["psnr     "] + [f"{val:.4f}" if isinstance(val, float) else "#" for val in psnr_values]))

    full_image_similarity_values = [
        metrics_by_bbox_ratio[cat]["cossim"] / metrics_by_bbox_ratio[cat]["count"] if metrics_by_bbox_ratio[cat]["count"] > 0 else "#" for cat in ["10", "30", "50", "70", "70above"]
    ]
    full_image_similarity_values.append(avg_cossim)
    print("\t".join(["image sim"] + [f"{val:.4f}" if isinstance(val, float) else "#" for val in full_image_similarity_values]))