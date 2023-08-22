import os
import sys
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utility import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Distribute the images under the given directory according to the given fold to fit the yolo dataset directory structure.(train, test, valid)")
    parser.add_argument("--source_dir", type=str, default="/mldisk_shared_/hbmun/vcdb/frames", help="train image directory")
    parser.add_argument("--target_dir", type=str, default="/mldisk_shared_/hbmun/vcdb/frame_dataset", help="train image directory")
    parser.add_argument("--fold", type=str, default="1,1,4", help="fraction of valid,test,train")
    option = parser.parse_known_args()[0]

    source_dir = option.source_dir
    target_dir = option.target_dir

    image_paths = [os.path.join(source_dir, image_name) for image_name in os.listdir(source_dir)]

    dataset_types = ["valid", "test", "train"]
    dataset_ratios = [int(ratio) for ratio in str(option.fold).split(",")]
    dataset_fold_count = sum(dataset_ratios)
    dataset_count = 0

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for i, dataset_type in enumerate(dataset_types):
        dataset_dir = os.path.join(target_dir, "images", dataset_type)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        if i != len(dataset_types):
            dataset_ratio = dataset_ratios[i]
            image_count = int(dataset_ratio/dataset_fold_count * len(image_paths))

            for idx in range(dataset_count, (image_count + dataset_count)):
                source = image_paths[idx]
                target = os.path.join(dataset_dir, source.split("/")[-1])
                shutil.copy(source, target)
                print(logging.ir(f"[{idx: 6d} /{dataset_count: 6d}] {source} -> {target}"), end="")

            dataset_count += (image_count - 1)
        else:
            for idx in range(dataset_count, len(image_paths)):
                source = image_paths[idx]
                target = os.path.join(dataset_dir, source.split("/")[-1])
                shutil.copy(source, target)
                print(logging.ir(f"[{idx: 6d} /{len(image_paths): 6d}] {source} -> {target}"), end="")
