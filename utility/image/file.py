import os
import cv2
import numpy as np
import torchvision.utils as vutils

from utility.image.visualize import draw_boxed_text


def save_cropped_faces(image_name:str, cropped_faces, save_dir):
    image_name.replace(".jpg", "")
    for idx, cropped_face in enumerate(cropped_faces):
        cv2.imwrite(os.path.join(save_dir, f"{image_name}_{idx}.jpg"), cropped_face)

def save_face_txt(path, face_regions):
    with open(path, "w") as file:
        for face_region in face_regions:
            file.write(face_region + "\n")
    return True

def save_tensort2image(image, path):
    """
    Save a image using PyTorch's save_image function.

    Args:
        image (torch.Tensor): An image tensor to save. Shape: [C, H, W]
        path (str): The path where the image will be saved.

    """
    image = image.cpu()
    image = (image * 0.5) + 0.5

    vutils.save_image(image.unsqueeze(0), path)


def save_bbox_image_xywh(save_path, image, labels):
    for i, label in enumerate(labels):
        _, cls, conf, x, y, width, height = label

        left = int(x)
        top = int(y)
        right = int(x + width)
        bottom = int(y + height)

        txt_loc = (max(left + 2, 0), max(top + 2, 0))
        txt = f'{i}: {conf:.2f}'
        image = draw_boxed_text(image, txt, txt_loc, (0, 255, 0))

        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.imwrite(save_path, image)


def save_bbox_image_yolo(save_path, image, labels):
    for label in labels:
        _, x, y, width, height = map(float, label.split(" "))

        left = int((x - width / 2) * image.shape[1])
        top = int((y - height / 2) * image.shape[0])
        right = int((x + width / 2) * image.shape[1])
        bottom = int((y + height / 2) * image.shape[0])

        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.imwrite(save_path, image)

def save_mask_from_yolo(img, yolo_labels, mask_save_path):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for yolo_label in yolo_labels:
        _, x_center, y_center, width, height = map(float, yolo_label.strip().split())
        x_center *= w
        y_center *= h
        width *= w
        height *= h

        left = int(x_center - width / 2)
        right = int(x_center + width / 2)
        top = int(y_center - height / 2)
        bottom = int(y_center + height / 2)

        mask[top:bottom, left:right] = 255

    cv2.imwrite(mask_save_path, mask)
