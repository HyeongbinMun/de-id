import os
import sys
import shutil
import argparse
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Distribute the images under the given directory according to the given fold to fit the yolo dataset directory structure.(train, test, valid)")
    parser.add_argument("--source_dir", type=str, default="/dataset/vggface2/origin", help="VGGface2-HQ original dataset directory path")
    parser.add_argument("--target_dir", type=str, default="/dataset/vggface2/refine", help="refine dataset directory path")

    option = parser.parse_known_args()[0]

    source_dir = option.source_dir
    target_dir = option.target_dir

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    dataset_types = os.listdir(source_dir)

    resolutions = []

    source_image_classes = os.listdir(source_dir)
    progress_bar = tqdm(enumerate(source_image_classes), total=len(source_image_classes))
    for k, source_image_class in progress_bar:
        source_image_names = os.listdir(os.path.join(source_dir, source_image_class))
        for source_image_name in source_image_names:
            source_image_path = os.path.join(source_dir, source_image_class, source_image_name)
            target_image_path = os.path.join(target_dir, f"{source_image_class}_{source_image_name}")

            with Image.open(source_image_path) as img:
                resolutions.append(img.size)

            shutil.copy(source_image_path, target_image_path)
        progress_bar.set_description(f"Copying({source_image_class})... {k + 1}/{len(source_image_classes)}")

    min_resolution = min(resolutions, key=lambda x: x[0] * x[1])
    max_resolution = max(resolutions, key=lambda x: x[0] * x[1])
    avg_resolution = tuple(sum(x[i] for x in resolutions) / len(resolutions) for i in range(2))

    print(f"Min Resolution: {min_resolution}")
    print(f"Max Resolution: {max_resolution}")
    print(f"Avg Resolution: {avg_resolution}")
