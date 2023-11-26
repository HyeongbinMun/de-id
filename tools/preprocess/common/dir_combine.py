import os
import shutil
import argparse
import random
from tqdm import tqdm

def copy_all_images_from_directory(src_dir, dst_dir, num_samples=None):
    """
    src_dir에서 원하는 개수만큼 이미지를 무작위로 선택하여 dst_dir에 저장합니다.

    Parameters:
    - src_dir: 이미지가 저장된 원본 디렉토리 경로
    - dst_dir: 이미지를 복사할 대상 디렉토리 경로
    - num_samples: 복사할 이미지의 개수 (None이면 모든 이미지를 복사)
    """

    os.makedirs(dst_dir, exist_ok=True)

    all_images = [os.path.join(subdir, file)
                  for subdir, _, files in os.walk(src_dir)
                  for file in files
                  if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # num_samples가 주어진 경우, 해당 개수만큼 이미지를 무작위로 선택
    if num_samples is not None:
        all_images = random.sample(all_images, num_samples)

    for src_file_path in tqdm(all_images, desc="Copying images", unit="file"):
        file_name = os.path.basename(src_file_path)
        dst_file_path = os.path.join(dst_dir, file_name)
        shutil.copy2(src_file_path, dst_file_path)


def copy_image_and_mask(src_image_dir, dst_image_dir, dst_mask_dir, keyword=None, num_samples=None):
    """
    Args:
    src_image_dir (str): 소스 이미지 디렉토리의 경로.
    dst_image_dir (str): 대상 이미지 디렉토리의 경로.
    dst_mask_dir (str): 대상 마스크 디렉토리의 경로.
    keyword (str): 파일 이름에 추가할 키워드.
    num_samples (int, optional): 복사할 이미지의 수. None이면 모든 이미지를 복사합니다.
    """
    
    # 모든 이미지 파일의 경로를 리스트로 저장합니다.
    all_images = [os.path.join(subdir, file)
                  for subdir, _, files in os.walk(src_image_dir)
                  for file in files
                  if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if num_samples:
        all_images = random.sample(all_images, num_samples)

    for src_image_path in tqdm(all_images, desc="Copying images and masks", unit="file"):
        # 이미지와 마스크 파일의 이름은 동일하므로 basename을 사용
        file_name = keyword + "_" + os.path.basename(src_image_path)

        src_mask_path = src_image_path.replace("images", "masks")

        dst_image_path = os.path.join(dst_image_dir, file_name)
        dst_mask_path = os.path.join(dst_mask_dir, file_name)

        # 마스크 파일이 존재하지 않을 경우 에러를 무시하고 계속 진행
        try:
            shutil.copy2(src_mask_path, dst_mask_path)
            shutil.copy2(src_image_path, dst_image_path)
        except FileNotFoundError:
            print(f"Mask file not found for {src_mask_path}, skipping.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy a specified number of images and their masks from one directory to another.")
    parser.add_argument("--type", type=str, default="double", help="single or double")
    parser.add_argument("--src_image_dir", type=str, default='/dataset/face/vggface2hq_refine/images', help="Source directory images path.")
    parser.add_argument("--dst_image_dir", type=str, default='/dataset/face/class/vggface2_pre_1000/images', help="save directory images path.")
    parser.add_argument("--dst_mask_dir", type=str, default='/dataset/face/class/vggface2_pre_1000/masks', help="save directory masks path.")
    parser.add_argument("--keyword", type=str, help="image keyword.")
    parser.add_argument("--num_samples", type=int, help="Number of images and masks to copy.")

    args = parser.parse_args()

    if args.type == 'single':
        copy_all_images_from_directory(args.src_image_dir, args.dst_image_dir)
    else:
        copy_image_and_mask(args.src_image_dir, args.dst_image_dir, args.dst_mask_dir, args.keyword, args.num_samples)