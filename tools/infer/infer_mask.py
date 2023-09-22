import cv2
import numpy as np


def generate_face_mask(image_path):
    # Load the cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the input image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Create a black mask with same size as the image
    mask = np.zeros_like(gray)

    # Fill in the detected faces in the mask with white color
    for (x, y, w, h) in faces:
        mask[y:y + h, x:x + w] = 255

    return mask


# Example
mask = generate_face_mask('/workspace/test/multi.jpg')
cv2.imwrite('/workspace/test/multi_face.jpg', mask)
