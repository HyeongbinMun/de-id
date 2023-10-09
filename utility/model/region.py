import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F


def crop_face(images, boxes, index):
    max_face = 16
    faces = []
    face_sizes = []
    image_indices = []
    valid_boxes = []

    for b, box in enumerate(boxes):
        image_w, image_h = images.shape[3], images.shape[2]
        _, x_center, y_center, w, h = box
        x1 = int((x_center - w / 2) * image_w)
        y1 = int((y_center - h / 2) * image_h)
        x2 = int((x_center + w / 2) * image_w)
        y2 = int((y_center + h / 2) * image_h)

        if (x2 - x1) > 10 and (y2 - y1) > 10 and x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            face = images[index, :, y1:y2, x1:x2].clone()
            faces.append(face)
            face_sizes.append((y2 - y1) * (x2 - x1))
            image_indices.append(index)
            valid_boxes.append(box)

    if len(faces) > max_face:
        sorted_indices = sorted(range(len(face_sizes)), key=lambda k: face_sizes[k], reverse=True)[:max_face]

        faces = [faces[i] for i in sorted_indices]
        image_indices = [image_indices[i] for i in sorted_indices]
        valid_boxes = [valid_boxes[i] for i in sorted_indices]

    return faces, image_indices, valid_boxes


def overlay_faces_on_image(original_images, fake_faces, valid_boxes, indices):
    overlaid_images = original_images.clone()

    for face, box, idx in zip(fake_faces, valid_boxes, indices):
        image_w, image_h = original_images.shape[3], original_images.shape[2]
        _, x_center, y_center, w, h = box
        x1 = int((x_center - w / 2) * image_w)
        y1 = int((y_center - h / 2) * image_h)
        x2 = int((x_center + w / 2) * image_w)
        y2 = int((y_center + h / 2) * image_h)

        face_resized = F.interpolate(face.unsqueeze(0), size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False).squeeze(0)
        h_face, w_face = face_resized.shape[1], face_resized.shape[2]

        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(image_h, y1 + h_face)
        x2 = min(image_w, x1 + w_face)

        face_resized = face_resized[:, 0:y2 - y1, 0:x2 - x1]
        overlaid_images[idx, :, y1:y2, x1:x2] = face_resized

    return overlaid_images


def save_gan_concat_text_image(tensor_image, captions, image_path):
    image_w, image_h = tensor_image.shape[2]/3, tensor_image.shape[1]/2
    pil_image = Image.fromarray((tensor_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("utility/font/malgun.ttf", 50)
    positions = [(10, 10), (10 + image_w, 10), (10 + (image_w * 2), 10), (10, 10 + image_h), (10 + image_w, 10 + image_h), (10 + (image_w * 2), 10 + image_h)]
    for idx, caption in enumerate(captions):
        draw.text(positions[idx], caption, font=font, fill=(255, 0, 0))  # red color for the text
    pil_image.save(image_path, "JPEG")