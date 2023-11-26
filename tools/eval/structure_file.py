import os
import shutil
import argparse
from tqdm import tqdm

def copy_images_to_structure(source_dir, target_dir, save_dir, keyword):
    source_images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    keyword_rename = keyword + '_'
    target_images = {}
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            target_images[file] = root

    for image in tqdm(source_images, desc="Copying images", unit="file"):
        modified_name = image.replace(keyword_rename, '')

        if modified_name in target_images:
            source_path = os.path.join(source_dir, image)
            target_subdir = os.path.relpath(target_images[modified_name], target_dir)
            save_subdir = os.path.join(save_dir, target_subdir)

            if not os.path.exists(save_subdir):
                os.makedirs(save_subdir)

            destination_path = os.path.join(save_subdir, modified_name)
            shutil.copy(source_path, destination_path)
            # print(f'Copied: {source_path} -> {destination_path}')  # Optional: Uncomment to see detailed copy log

def main():
    parser = argparse.ArgumentParser(description="Copy images from source to save directory with target directory structure.")
    parser.add_argument("--source_dir", help="Source directory containing images.")
    parser.add_argument("--target_dir", help="Target directory to mimic the structure.")
    parser.add_argument("--save_dir", help="Save directory where images will be copied.")
    parser.add_argument("--keyword", help="Keyword to remove from the source image filenames.")

    args = parser.parse_args()

    copy_images_to_structure(args.source_dir, args.target_dir, args.save_dir, args.keyword)

if __name__ == "__main__":
    main()
