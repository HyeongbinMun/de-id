def crop_faces(image, results):
    cropped_faces = []
    for idx, bbox in enumerate(results):
        x = bbox[3]
        y = bbox[4]
        w = bbox[5]
        h = bbox[6]

        cropped_face = image[y:y + h, x:x + w]
        cropped_faces.append(cropped_face)

    return cropped_faces