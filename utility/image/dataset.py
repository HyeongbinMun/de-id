import os
import cv2
from tqdm import tqdm
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

def summary_video_dir(video_dir):
    video_extensions = ['.mp4', '.avi', '.mkv', '.flv']
    video_list = os.listdir(video_dir)
    video_files = [file for file in video_list if os.path.splitext(file)[-1].lower() in video_extensions]
    resolution_dict = defaultdict(lambda: {"count": 0, "videos": []})
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution = f"{width}x{height}"

        resolution_dict[resolution]["count"] += 1
        resolution_dict[resolution]["videos"].append(video_file)

        cap.release()

    return video_list, resolution_dict


def summary_image_dir(image_dir):
    image_extensions = ['.jpg', '.png', '.bmp', 'jpeg']
    image_list = os.listdir(image_dir)
    image_files = sorted([file for file in image_list if os.path.splitext(file)[-1].lower() in image_extensions])

    video_frames_dict = defaultdict(int)

    for image_file in image_files:
        video_number = image_file.split('_')[0]
        video_frames_dict[video_number] += 1

    return image_list, video_frames_dict


def gather_files_from_directory(directory_path, extension):
    all_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(extension):
                all_files.append(os.path.join(root, file))
    return all_files


def summary_yolo_dataset(dataset_dir):
    all_image_files = sorted(gather_files_from_directory(os.path.join(dataset_dir, 'images'), '.jpg'))
    all_label_files = sorted([os.path.splitext(img_file)[0].replace('images', 'labels') + '.txt' for img_file in all_image_files])

    total_boxes = 0
    total_area_ratio = 0.0
    max_area_ratio = 0.0
    min_area_ratio = float('inf')
    max_ratio_image = ""
    min_ratio_image = ""
    progress_bar = tqdm(enumerate(all_label_files), total=len(all_label_files))
    for i, label_file in progress_bar:
        if not os.path.exists(label_file):
            print(label_file)
            continue

        with open(label_file, 'r') as f:
            lines = f.readlines()
            try:
                boxes = []
                image = Image.open(all_image_files[i])
                image_w, image_h = image.size
                for line in lines:
                    _, x_center, y_center, width, height = map(float, line.strip().split())

                    x1 = int((x_center - width / 2) * image_w)
                    y1 = int((y_center - height / 2) * image_h)
                    x2 = int((x_center + width / 2) * image_w)
                    y2 = int((y_center + height / 2) * image_h)
                    boxes.append((x1, y1, x2, y2))

                    total_boxes += 1
                area_ratio = compute_total_area(boxes, image_w, image_h)
                total_area_ratio += area_ratio
                if area_ratio > max_area_ratio:
                    max_area_ratio = area_ratio
                    max_ratio_image = os.path.basename(label_file).replace('.txt', '.jpg')
                if area_ratio < min_area_ratio:
                    min_area_ratio = area_ratio
                    min_ratio_image = os.path.basename(label_file).replace('.txt', '.jpg')
                f.close()
            except:
                pass

    avg_boxes_per_image = total_boxes / len(all_image_files)
    avg_area_ratio = total_area_ratio / total_boxes
    summary = {
        "total_images": all_image_files,
        "total_labels": all_label_files,
        "total_boxes": total_boxes,
        "avg_boxes_per_image": avg_boxes_per_image,
        "box_area_ratios": {
            "average": avg_area_ratio,
            "max": {
                "ratio": max_area_ratio,
                "image": max_ratio_image
            },
            "min": {
                "ratio": min_area_ratio,
                "image": min_ratio_image
            }
        }
    }

    return summary

def compute_total_area(boxes, img_width, img_height):
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    total_area = sum(areas)
    return total_area / (img_width * img_height)

def classify_percentage(total_area):
    if total_area <= 0.10:
        return "10"
    elif total_area <= 0.30:
        return "30"
    elif total_area <= 0.50:
        return "50"
    elif total_area <= 0.70:
        return "70"
    else:
        return "70above"


def count_images_in_directory(directory_path):
    image_extensions = ['.jpg', '.png', '.jpeg']
    return sum([len(files) for subdir, dirs, files in os.walk(directory_path) if
                any(file.lower().endswith(ext) for file in files for ext in image_extensions)])


def summery_face_ratio(dataset_path, output_path):
    if not os.path.exists(dataset_path):
        print(f"'{dataset_path}' does not exist!")
        return

    if os.path.exists(output_path):
        dataset_image_count = count_images_in_directory(os.path.join(dataset_path, 'images'))
        output_image_count = count_images_in_directory(output_path)

        if dataset_image_count == output_image_count:
            print("Image counts in dataset_path and output_path are same!")
            return

    categories = os.listdir(os.path.join(dataset_path, "images"))

    for percentage in ["10", "30", "50", "70", "70above"]:
        percentage_path = os.path.join(output_path, percentage)
        if not os.path.exists(percentage_path):
            os.makedirs(percentage_path)

    for category in categories:
        label_path = os.path.join(dataset_path, "labels", category)
        image_path = os.path.join(dataset_path, "images", category)

        for label_file in tqdm(os.listdir(label_path), desc=f"Processing {category}"):
            with open(os.path.join(label_path, label_file), 'r') as f:
                lines = f.readlines()
                boxes = []

                try:
                    for line in lines:
                        parts = line.strip().split()
                        class_id, x_center, y_center, width, height = map(float, parts)
                        img_file_path = os.path.join(image_path, label_file.replace(".txt", ".jpg"))
                        img = Image.open(img_file_path)
                        image_w, image_h = img.size

                        x1 = int((x_center - width / 2) * image_w)
                        y1 = int((y_center - height / 2) * image_h)
                        x2 = int((x_center + width / 2) * image_w)
                        y2 = int((y_center + height / 2) * image_h)

                        boxes.append((x1, y1, x2, y2))

                    total_area = compute_total_area(boxes, image_w, image_h)
                    percentage_category = classify_percentage(total_area)

                    draw = ImageDraw.Draw(img)
                    for box in boxes:
                        draw.rectangle(box, outline="red", width=2)

                    total_bbox_area = sum([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes])
                    total_image_area = image_w * image_h
                    area_percentage = total_bbox_area / total_image_area

                    text = f"{total_bbox_area}({'+'.join([f'bbox{i + 1}: {(box[2] - box[0])}x{(box[3] - box[1])}' for i, box in enumerate(boxes)])})/ {total_image_area}({image_w}x{image_h}) = {area_percentage:.2f}"

                    font = ImageFont.truetype("utility/font/malgun.ttf", 15)
                    draw.text((10, 10), text, font=font, fill="red")

                    img.save(os.path.join(output_path, percentage_category, label_file.replace(".txt", ".jpg")))
                except:
                    img.save(os.path.join(output_path, "10", label_file.replace(".txt", ".jpg")))

def print_face_ratio(output_path):
    for percentage in ["10", "30", "50", "70", "70above"]:
        percentage_path = os.path.join(output_path, percentage)
        count = count_images_in_directory(percentage_path)
        print(f"{percentage}: {count} images")