import os
import sys
import shutil
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Distribute the images under the given directory according to the given fold to fit the yolo dataset directory structure.(train, test, valid)")
    parser.add_argument("--source_dir", type=str, default="/mldisk_shared_/hbmun/widerface/origin", help="train image directory")
    parser.add_argument("--target_dir", type=str, default="/mldisk_shared_/hbmun/widerface/refine", help="train image directory")

    option = parser.parse_known_args()[0]

    source_dir = option.source_dir
    target_dir = option.target_dir

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    images_dir = os.path.join(target_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    dataset_types = os.listdir(source_dir)

    for i, dataset_type in enumerate(dataset_types):
        dataset_dir = os.path.join(source_dir, dataset_type)
        target_image_dir = os.path.join(target_dir, "images", dataset_type)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        if not os.path.exists(target_image_dir):
            os.makedirs(target_image_dir)
        sub_dirs = [os.path.join(dataset_dir, cls) for cls in os.listdir(dataset_dir)]

        for j, sub_dir in enumerate(sub_dirs):
            source_image_classes = os.listdir(sub_dir)
            progress_bar = tqdm(enumerate(source_image_classes), total=len(source_image_classes))
            for k, source_image_class in progress_bar:
                source_image_names = os.listdir(os.path.join(sub_dir, source_image_class))
                for source_image_name in source_image_names:
                    source_image_path = os.path.join(sub_dir, source_image_class, source_image_name)
                    target_image_path = os.path.join(target_image_dir, source_image_name)
                    shutil.copy(source_image_path, target_image_path)
                progress_bar.set_description(f"Copying({dataset_type} - {source_image_class})... {k + 1}/{len(source_image_classes)}")