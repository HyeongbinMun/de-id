import os
import sys
import cv2
import piqa
import torch
import argparse
import torchvision.transforms as transforms

import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.models import model_classes
from utility.params import load_params_yml
from model.det.face.yolov7.yolov7_face import YOLOv7Face

# 리사이즈 함수 추가
def resize_image_to_match(target, source):
    return F.interpolate(target, size=(source.shape[2], source.shape[3]), mode='bilinear', align_corners=False)

def get_face_region(image, detections):
    """Get the face region from the image using detections."""
    for detection in detections:
        if detection[0] == 'face':
            x, y, w, h = map(int, detection[3:7])

            # Convert x, y, w, h to top-left and bottom-right corner coordinates
            x1, y1 = x, y
            x2, y2 = x + w, y + h

            face_region = (image[y1:y2, x1:x2] * 255).astype('uint8')
            return face_region
    return None

def load_single_image(image_path_or_array, device):
    try:
        transform = transforms.Compose([transforms.ToTensor()])
        if type(image_path_or_array) == str:
            image = Image.open(image_path_or_array).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            return image
        else:
            image = Image.fromarray(image_path_or_array).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            return image
    except ValueError as e:
        print(f"Error occurred while processing the image: {image_path_or_array}")
        print(f"Error message: {e}")
        return None
    except AttributeError as e:
        print(f"AttributeError occurred with input '{image_path_or_array}': {e}")
        print(f"Input type: {type(image_path_or_array)}")
        return None

def compute_face_area_ratio(face, original_image):
    face_area = face.shape[0] * face.shape[1]
    total_area = original_image.shape[0] * original_image.shape[1]
    return (face_area / total_area) * 100


def evaluate_image_pair(feature_model, feature_model_input_size, origin_image_path, deid_image_path):
    # Load images
    origin_img = load_single_image(origin_image_path, device)
    deid_img = load_single_image(deid_image_path, device)

    # Check if resolutions are different
    if origin_img.shape[2:] != deid_img.shape[2:]:
        deid_img = F.interpolate(deid_img, size=(origin_img.shape[2], origin_img.shape[3]), mode='bilinear', align_corners=False)

    # Metrics Initialization
    ssim = piqa.SSIM(n_channels=3).to(device)
    psnr = piqa.PSNR().to(device)
    cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    # Compute metrics
    orin_deid_psnr = psnr(origin_img, deid_img)
    orin_deid_ssim = ssim(origin_img, deid_img)
    #orin_deid_mse = F.mse_loss(origin_img, deid_img)

    # Feature Similarity
    resized_origin_img = F.interpolate(origin_img, size=(feature_model_input_size, feature_model_input_size))
    resized_deid_img = F.interpolate(deid_img, size=(feature_model_input_size, feature_model_input_size))
    orin_features = feature_model(resized_origin_img)
    deid_features = feature_model(resized_deid_img)

    feature_cossim = cosine_similarity(orin_features, deid_features).mean()

    origin_images = [cv2.imread(origin_image_path)]
    deid_images = [cv2.imread(deid_image_path)]
    # Check if resolutions are different
    if origin_images[0].shape[:2] != deid_images[0].shape[:2]:
        deid_images[0] = cv2.resize(deid_images[0], (origin_images[0].shape[1], origin_images[0].shape[0]))

    origin_results = detect_model.detect_batch(origin_images)

    # Assuming you are comparing two images only
    origin_face = get_face_region(origin_images[0], origin_results[0])
    deid_face = get_face_region(deid_images[0], origin_results[0])

    # Calculate the face area ratio
    # origin_face_ratio = compute_face_area_ratio(origin_face, origin_images[0])
    # deid_face_ratio = compute_face_area_ratio(deid_face, deid_images[0])
    # print(f"Original Image Face Area Ratio: {origin_face_ratio:.2f}%")
    # print(f"De-identified Image Face Area Ratio: {deid_face_ratio:.2f}%")

    orgin_face_img = load_single_image(origin_face, device)
    deid_face_img = load_single_image(deid_face, device)

    # 이미지 로드에 실패한 경우
    if orgin_face_img is None or deid_face_img is None:
        mse_face_score = 0
        print(f"Image path : {origin_image_path}")
    else:
        orgin_face_img = resize_image_to_match(orgin_face_img, deid_face_img)

        # Feature Similarity
        # resized_origin_face_img = F.interpolate(orgin_face_img, size=(feature_model_input_size, feature_model_input_size))
        # resized_deid_face_img = F.interpolate(deid_face_img, size=(feature_model_input_size, feature_model_input_size))
        # orin_face_features = feature_model(resized_origin_face_img)
        # deid_face_features = feature_model(resized_deid_face_img)

        # face score
        mse_face_score = F.mse_loss(orgin_face_img, deid_face_img)
        # face_psnr_score = psnr(orgin_face_img, deid_face_img)
        # face_ssim_score = ssim(orgin_face_img, deid_face_img)
        # face_cosine_score = cosine_similarity(orin_face_features, deid_face_features).mean()

    # 결과 반환
    return {
        "psnr": orin_deid_psnr.item(),
        "ssim": orin_deid_ssim.item(),
        "mse_face": mse_face_score,
        "cosine_similarity": feature_cossim.item()
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate metrics for single image pair.")
    parser.add_argument("--config", type=str, default="/workspace/config/config_yolov7face.yml", help="parameter file path")
    parser.add_argument("--model", type=str, default="ResNet50", help="feature model ResNet50 or MobileNet_AVG")
    parser.add_argument("--all_print", type=str, default=True, help="all print option True or False")
    parser.add_argument("--model_path", type=str, default="/workspace/model/weights/sscd_disc_mixup.torchvision.pt", help="feature model path")
    parser.add_argument("--origin_directory", type=str, required=True, help="Directory containing original images.")
    parser.add_argument("--deid_directory", type=str, required=True, help="Directory containing de-identified images.")
    option = parser.parse_known_args()[0]

    params_yml_path = option.config
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = load_params_yml(params_yml_path)["infer"]
    detect_model = YOLOv7Face(params)

    # Feature Extraction Model
    if option.model == "ResNet50":
        feature_model = model_classes["feature"][option.model](backbone="TV_RESNET50", dims=512, pool_param=3).to(
            device)
        feature_model.load_state_dict(torch.load(option.model_path))
    else:
        feature_model = model_classes["feature"][option.model]().to(device)
        state_dict = torch.load(option.model_path, map_location=device)
        state_dict = {k.replace("base.", ""): v for k, v in state_dict.items()}
        feature_model.load_state_dict(state_dict, strict=False)
    feature_model.eval()
    feature_model_input_size = 224
    print(f"Feature Extraction model is successfully loaded.({option.model})")

    # 이미지 파일 이름 목록 가져오기
    origin_files = [f for f in os.listdir(option.origin_directory) if f.endswith('.jpg')]
    deid_files = [f for f in os.listdir(option.deid_directory) if f.endswith('.jpg')]

    # 두 디렉토리에 모두 존재하는 파일 이름만 가져오기
    common_files = set(origin_files).intersection(set(deid_files))

    # 각 스코어의 누적 값을 저장할 딕셔너리
    total_scores = {"psnr": 0, "ssim": 0, "mse_face": 0, "cosine_similarity": 0}

    # 각 이미지 쌍의 스코어를 저장할 리스트
    all_scores = []

    # 각 쌍에 대한 평가 수행
    for image_file in tqdm(common_files, desc="Evaluating"):
        origin_image_path = os.path.join(option.origin_directory, image_file)
        deid_image_path = os.path.join(option.deid_directory, image_file)

        scores = evaluate_image_pair(feature_model, feature_model_input_size, origin_image_path, deid_image_path)
        all_scores.append((image_file, scores))

        # 각 스코어 누적
        for key in scores:
            total_scores[key] += scores[key]

    # 모든 스코어 출력
    if option.all_print:
        for image_file, scores in all_scores:
            print(f"Image: {image_file}")
            print(
                f"PSNR: {scores['psnr']:.4f}, \n"
                f"SSIM: {scores['ssim']:.4f}, \n"
                f"MSE Face: {scores['mse_face']:.4f}, \n"
                f"Cosine Similarity: {scores['cosine_similarity']:.4f}")
            print("---------------------------------------------------")

    num_images = len(common_files)

    # 평균 스코어 출력
    print("\n----------------- Average Scores -----------------")
    for key in total_scores:
        avg_score = total_scores[key] / num_images
        print(f"Average {key}: {avg_score:.4f}")
    print("---------------------------------------------------")
