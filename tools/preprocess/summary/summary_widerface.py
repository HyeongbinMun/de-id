import os
import sys
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utility.image.dataset import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--origin_dataset_dir", type=str, default="/mldisk2/deid/widerface/split", help="parameter file path")
    parser.add_argument("--face_ratio_dir", type=str, default="/mldisk2/deid/widerface/split/eval", help="parameter file path")
    option = parser.parse_known_args()[0]

    origin_dataset_dir = option.origin_dataset_dir
    face_ratio_dir = option.face_ratio_dir
    terminal_size = shutil.get_terminal_size().columns

    print("-" * terminal_size)
    print("widerface train dataset")
    print("origin dataset information")
    print("-" * terminal_size)
    summary = summary_yolo_dataset(origin_dataset_dir)
    print(f"total # of data        : {len(summary['total_images'])}")
    print(f"total # of bbox        : {summary['total_boxes']}")
    print(f"avg # of bbox in images: {summary['avg_boxes_per_image']:.2f}")
    print(f"max ratio of the whole image occupied by the bounding box: {summary['box_area_ratios']['max']['ratio'] * 100:.2f}% - {summary['box_area_ratios']['max']['image']}")
    print(f"min ratio of the whole image occupied by the bounding box: {summary['box_area_ratios']['min']['ratio'] * 100:.2f}% - {summary['box_area_ratios']['min']['image']}")
    print(f"avg ratio of the whole image occupied by the bounding box: {summary['box_area_ratios']['average'] * 100:.2f}%")
    summery_face_ratio(origin_dataset_dir, face_ratio_dir)
    print_face_ratio(face_ratio_dir)