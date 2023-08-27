import os
import PIL
import argparse
from tqdm import tqdm
from PIL import Image

def check_and_delete_images(dataset_dir):
    root_images_folder = os.path.join(dataset_dir, 'images')
    root_labels_folder = os.path.join(dataset_dir, 'labels')
    dataset_types = os.listdir(root_images_folder)
    for dataset_type in dataset_types:
        images_folder = os.path.join(root_images_folder, dataset_type)
        labels_folder = os.path.join(root_labels_folder, dataset_type)
        image_files = os.listdir(images_folder)
        progress_bar = tqdm(total=len(image_files), desc="Checking images", unit="image")
        for image_file in image_files:
            if image_file.endswith('.jpg'):
                image_path = os.path.join(images_folder, image_file)
                try:
                    Image.open(image_path).convert('RGB')
                except (OSError, PIL.UnidentifiedImageError):
                    os.remove(image_path)
                    label_name = os.path.splitext(image_file)[0] + '.txt'
                    label_path = os.path.join(labels_folder, label_name)
                    if os.path.exists(label_path):
                        os.remove(label_path)
                    progress_bar.write(f"Deleted: {image_file}")
                label_name = os.path.splitext(image_file)[0] + '.txt'
                label_path = os.path.join(labels_folder, label_name)
                with open(label_path, "r") as label_file:
                    labels = label_file.readlines()
                    if len(labels) == 0:
                        os.remove(label_path)
                        os.remove(os.path.join(images_folder, image_file))
                        progress_bar.write(f"Deleted: {image_file}, {label_name}")
                    elif len(labels) > 0:
                        flag = False
                        for label in labels:
                            if len(label.split(" ")) != 5:
                                print()
                                flag = True
                        if flag == True:
                            print([len(label.split(" ")) for label in labels])
                            os.remove(label_path)
                            os.remove(os.path.join(images_folder, image_file))
                            progress_bar.write(f"Deleted: {image_file}, {label_name}")
                    label_file.close()

                progress_bar.update(1)
        progress_bar.close()

    for dataset_type in dataset_types:
        images_folder = os.path.join(root_images_folder, dataset_type)
        labels_folder = os.path.join(root_labels_folder, dataset_type)
        image_files = os.listdir(images_folder)
        label_files = os.listdir(labels_folder)
        print(f"{dataset_type}: {len(image_files)}/{len(label_files)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Delete any images in the folder that don't open with PIL.")
    parser.add_argument("--dataset_dir", type=str, default="/dataset/images_1080", help="dataset directory path")

    option = parser.parse_known_args()[0]

    dataset_dir = option.dataset_dir

    check_and_delete_images(dataset_dir)