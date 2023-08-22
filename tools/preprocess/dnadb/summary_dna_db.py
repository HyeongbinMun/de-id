import os
import sys
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utility.image.dataset import summary_video_dir, summary_image_dir, summary_yolo_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--orin_video_dir", type=str, default="/hdd/dna_db/DNA_DB/", help="parameter file path")
    parser.add_argument("--orin_image_dir", type=str, default="/hdd/dna_db/dna_db_frame_origin/", help="parameter file path")
    parser.add_argument("--dataset_dir", type=str, default="/dataset/dna_db/", help="source image directory")
    option = parser.parse_known_args()[0]

    orin_video_dir = option.orin_video_dir
    orin_image_dir = option.orin_image_dir
    dataset_dir = option.dataset_dir
    terminal_size = shutil.get_terminal_size().columns

    video_list, resolution_dict = summary_video_dir(orin_video_dir)
    print("-" * terminal_size)
    print("Origin DNA DB Video dataset Information")
    print("=" * terminal_size)
    print(f"Total video    : {len(video_list)}")
    print(f"Resolution Info:")
    for resolution, data in resolution_dict.items():
        print(f"{resolution}: {data['count']}개 - {', '.join(data['videos'])}")
    print("-" * terminal_size)

    image_list, video_frames_dict = summary_image_dir(orin_image_dir)
    frame_counts = list(video_frames_dict.values())
    max_length = max(frame_counts)
    min_length = min(frame_counts)
    avg_length = sum(frame_counts) / len(frame_counts)
    print("Origin DNA DB Frame data Information")
    print("-" * terminal_size)
    print(f"Total image     : {len(image_list)}")
    print(f"max video length: {max_length} sec")
    print(f"min video length: {min_length} sec")
    print(f"avg video length: {avg_length:.2f} sec")

    print("-" * terminal_size)
    print("DNA DB Face Dataset")
    print("-" * terminal_size)
    summary = summary_yolo_dataset(dataset_dir)
    print(f"total # of data        : {len(summary['total_images'])}")
    print(f"total # of bbox        : {summary['total_boxes']}")
    print(f"avg # of bbox in images: {summary['avg_boxes_per_image']:.2f}")
    print(f"max ratio of the whole image occupied by the bounding box: {summary['box_area_ratios']['max']['ratio'] * 100:.2f}% - {summary['box_area_ratios']['max']['image']}")
    print(f"min ratio of the whole image occupied by the bounding box: {summary['box_area_ratios']['min']['ratio'] * 100:.2f}% - {summary['box_area_ratios']['min']['image']}")
    print(f"avg ratio of the whole image occupied by the bounding box: {summary['box_area_ratios']['average'] * 100:.2f}%")
    print("-" * terminal_size)
