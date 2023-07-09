import os
import PIL
import argparse
from tqdm import tqdm
from PIL import Image

def check_and_delete_images(dataset_dir):
    dataset_types = os.listdir(dataset_dir)
    for dataset_type in dataset_types:
        images_folder = os.path.join(dataset_dir, 'images', dataset_type)
        labels_folder = os.path.join(dataset_dir, 'labels', dataset_type)

        image_files = os.listdir(images_folder)
        progress_bar = tqdm(total=len(image_files), desc="Checking images", unit="image")
        for image_file in image_files:
            if image_file.endswith('.jpg'):
                image_path = os.path.join(images_folder, image_file)
                try:
                    Image.open(image_path).convert('RGB')
                except (OSError, PIL.UnidentifiedImageError):
                    os.remove(image_path)
                    label_file = os.path.splitext(image_file)[0] + '.txt'
                    label_path = os.path.join(labels_folder, label_file)
                    if os.path.exists(label_path):
                        os.remove(label_path)
                    progress_bar.write(f"Deleted: {image_file}")
                progress_bar.update(1)
        progress_bar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="object detection model batch inference test script")
    parser.add_argument("--dataset_dir", type=str, default="/dataset/images_1080_0.1", help="dataset directory path")

    option = parser.parse_known_args()[0]

    dataset_type = option.dataset_type
    dataset_dir = option.dataset_dir

    check_and_delete_images(dataset_dir)