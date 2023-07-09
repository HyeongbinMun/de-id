import os
import argparse

def remove_missing_labels(images_dir, labels_dir):
    images_files = os.listdir(images_dir)
    labels_files = os.listdir(labels_dir)

    labels_images = set([file.split(".")[0] for file in labels_files])

    for image_file in images_files:
        image_name = image_file.split(".")[0]
        if image_name not in labels_images:
            image_path = os.path.join(images_dir, image_file)
            os.remove(image_path)
            print(f"Deleted image: {image_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Remove images without corresponding labels")
    parser.add_argument("--images_dir", type=str, default="/dataset/images_1080_0.1/labels/train", help="images directory")
    parser.add_argument("--labels_dir", type=str, default="/dataset/images_1080_0.1/images/train", help="labels directory")

    args = parser.parse_args()

    remove_missing_labels(args.images_dir, args.labels_dir)