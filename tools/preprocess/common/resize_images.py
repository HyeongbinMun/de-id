import os
import sys
import argparse
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

def resize_images(source_dir, target_dir, resolution):
    file_list = os.listdir(source_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    original_resolutions = defaultdict(int)
    for file_name in tqdm(file_list, desc="Resizing images"):
        source_path = os.path.join(source_dir, file_name)
        if os.path.isfile(source_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image = Image.open(source_path)
                original_resolutions[image.size] += 1
                resized_image = image.resize(resolution, Image.LANCZOS)
                target_path = os.path.join(target_dir, file_name)
                resized_image.save(target_path)
            except Exception as e:
                print(f"Error occurred while processing {file_name}: {str(e)}")
    return original_resolutions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Resizes an image in the source image directory to the given resolution.")
    parser.add_argument("--source_dir", type=str, default="/mldisk_shared_/hbmun/vcdb/vcdb_frame_origin/images", help="source image directory")
    parser.add_argument("--target_dir", type=str, default="/mldisk_shared_/hbmun/vcdb/vcdb_frame_1080p/images", help="target image directory")
    parser.add_argument("--resolution", type=str, default="1920x1080", help="resize resolution 'wxh'(e.g. 1920x1080)")

    args = parser.parse_args()

    width, height = map(int, args.resolution.split("x"))
    resolution = (width, height)

    original_resolutions = []
    dataset_types = os.listdir(args.source_dir)

    for dataset_type in dataset_types:
        original_resolutions.append(resize_images(os.path.join(args.source_dir, dataset_type), os.path.join(args.target_dir, dataset_type), resolution))
    for i, dataset_type in enumerate(dataset_types):
        print(f"Original Resolutions - ({dataset_type}):")
        for res, count in original_resolutions[i].items():
            print(f"- {res[0]}x{res[1]}: {count} images")
