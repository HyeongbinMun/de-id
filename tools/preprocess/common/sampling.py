import os
import sys
import cv2
import random
import shutil
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creates a subset of the original dataset by sampling the dataset at a given fraction.")
    parser.add_argument("--fraction", type=float, default=0.1, help="small dataset fraction")
    parser.add_argument("--source_dir", type=str, default="/hdd/dna_db/images_1080/", help="source dataset directory")
    parser.add_argument("--target_dir", type=str, default="/dataset/images_1080_0.1/", help="target dataset directory")

    option = parser.parse_known_args()[0]

    origin_image_dir = os.path.join(option.source_dir, "images")
    origin_label_dir = os.path.join(option.source_dir, "labels")
    target_image_dir = os.path.join(option.target_dir, "images")
    target_label_dir = os.path.join(option.target_dir, "labels")

    fraction = option.fraction

    dataset_types = os.listdir(origin_image_dir)

    for dataset_type in dataset_types:
        origin_images = [os.path.join(origin_image_dir, dataset_type, f) for f in os.listdir(os.path.join(origin_image_dir, dataset_type))]
        origin_labels = [os.path.join(origin_label_dir, dataset_type, f) for f in os.listdir(os.path.join(origin_label_dir, dataset_type))]
        small_image_dir = os.path.join(target_image_dir, dataset_type)
        small_label_dir = os.path.join(target_label_dir, dataset_type)
        pairs = list(zip(origin_images, origin_labels))

        random.shuffle(pairs)

        num_samples = int(fraction * len(pairs))
        selected_pairs = pairs[:num_samples]

        os.makedirs(small_image_dir, exist_ok=True)
        os.makedirs(small_label_dir, exist_ok=True)

        progress_bar = tqdm(enumerate(selected_pairs), total=len(selected_pairs))
        for i, (img_path, lbl_path) in progress_bar:
            shutil.copy2(img_path, small_image_dir)
            shutil.copy2(lbl_path, small_label_dir)
            progress_bar.set_description(f"Copying({dataset_type})... {i + 1}/{len(selected_pairs)}")