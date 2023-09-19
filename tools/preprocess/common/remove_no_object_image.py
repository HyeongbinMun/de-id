import os
from tqdm import tqdm
import argparse

def remove_missing_labels(images_dir, labels_dir):
    image_files = [os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith('.jpg')]
    label_files = [os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith('.txt')]

    # 집합 변환 (빠른 검색을 위해)
    image_files_set = set(image_files)
    label_files_set = set(label_files)

    images_without_labels = image_files_set - label_files_set

    for image_name in tqdm(images_without_labels, desc="Deleting images"):
        os.remove(os.path.join(images_dir, image_name + '.jpg'))
        print(f"Deleted image: {os.path.join(images_dir, image_name + '.jpg')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Remove images without corresponding labels")
    parser.add_argument("--images_dir", type=str, default="/dataset/dnadb/images", help="images directory")
    parser.add_argument("--labels_dir", type=str, default="/dataset/dnadb/labels", help="labels directory")

    args = parser.parse_args()

    dataset_types = os.listdir(args.images_dir)

    for i, dataset_type in enumerate(dataset_types):
        images_dir = os.path.join(args.images_dir, dataset_type)
        labels_dir = os.path.join(args.labels_dir, dataset_type)
        remove_missing_labels(images_dir, labels_dir)