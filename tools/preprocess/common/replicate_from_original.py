import os
import sys
import shutil
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utility import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create a dataset by copying images and labels from the original dataset that have matching names in the deepfake dataset.")
    parser.add_argument("--origin_dir", type=str, default="/mldisk_shared/deid/ETRI/data/image/images_filtered", help="train image directory")
    parser.add_argument("--deepfake_dir", type=str, default="/mldisk_shared/deid/ETRI/data/image/gan/deepfake", help="train image directory")
    parser.add_argument("--target_dir", type=str, default="/mldisk_shared/deid/ETRI/data/image/gan/origin", help="train image directory")
    option = parser.parse_known_args()[0]

    dataset1_base = option.origin_dir
    dataset2_base = option.deepfake_dir
    dataset3_base = option.target_dir

    sub_dirs = ["valid", "test", "train"]

    for sub_dir in sub_dirs:
        dataset1_images = os.path.join(dataset1_base, 'images', sub_dir)
        dataset1_labels = os.path.join(dataset1_base, 'labels', sub_dir)

        dataset2_images = os.path.join(dataset2_base, 'images', sub_dir)
        dataset2_labels = os.path.join(dataset2_base, 'labels', sub_dir)

        dataset3_images = os.path.join(dataset3_base, 'images', sub_dir)
        dataset3_labels = os.path.join(dataset3_base, 'labels', sub_dir)

        for dataset_image in [dataset2_labels, dataset3_images, dataset3_labels]:
            if not os.path.exists(dataset_image):
                os.makedirs(dataset_image, exist_ok=True)

        filtered_images = os.listdir(dataset2_images)
        filtered_labels = [image.replace('.jpg', '.txt') for image in filtered_images]

        for image, label in tqdm(zip(filtered_images, filtered_labels), total=len(filtered_images), desc=f"Copying {sub_dir}"):
            shutil.copy(os.path.join(dataset1_images, image), os.path.join(dataset3_images, image))
            shutil.copy(os.path.join(dataset1_labels, label), os.path.join(dataset2_labels, label))
            shutil.copy(os.path.join(dataset1_labels, label), os.path.join(dataset3_labels, label))
