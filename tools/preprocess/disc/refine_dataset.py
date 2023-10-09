import os
import sys
import shutil
import argparse
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Distribute the images under the given directory according to the given fold to fit the yolo dataset directory structure.(train, test, valid)")
    parser.add_argument("--source_dir", type=str, default="/dataset/disc21/disc21", help="train image directory")
    parser.add_argument("--target_dir", type=str, default="/hdd/disc21/disc21_all", help="train image directory")

    option = parser.parse_known_args()[0]

    source_dir = option.source_dir
    target_dir = option.target_dir

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    src_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    total_files = sum([len(files) for src in src_dirs for _, _, files in os.walk(os.path.join(source_dir, src)) if
                       any(file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')) for file in files)])

    with tqdm(total=total_files, desc="Copying files", unit="file") as pbar:
        for src_dir in src_dirs:
            for root, dirs, files in os.walk(os.path.join(source_dir, src_dir)):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        shutil.copy2(os.path.join(root, file), os.path.join(target_dir, file))
                        pbar.update(1)