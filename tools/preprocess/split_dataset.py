import os
import sys
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utility import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="data")
    parser.add_argument("--origin_dir", type=str, default="/hdd/dna_db/origin/dna_db_frame_face", help="train image directory")
    parser.add_argument("--output_dir", type=str, default="/hdd/dna_db/dataset", help="train image directory")
    parser.add_argument("--ratio", type=str, default="1,1,4", help="val,test,train")
    option = parser.parse_known_args()[0]

    origin_dir = option.origin_dir
    output_dir = option.output_dir

    image_paths = [os.path.join(origin_dir, image_name) for image_name in os.listdir(origin_dir)]

    dataset_types = ["val", "test", "train"]
    dataset_ratios = [int(ratio) for ratio in str(option.ratio).split(",")]
    dataset_fold_count = sum(dataset_ratios)
    dataset_count = 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i, dataset_type in enumerate(dataset_types):
        dataset_dir = os.path.join(output_dir, dataset_type)
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)
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
