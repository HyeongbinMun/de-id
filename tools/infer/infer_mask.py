import cv2
import numpy as np
import argparse

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

    total_face_area = 0  # Initialize variable to store total face area

    # Fill in the detected faces in the mask with white color and calculate the total face area
    for (x, y, w, h) in faces:
        mask[y:y + h, x:x + w] = 255
        total_face_area += w * h

    # Calculate the percentage of face area in the total image area
    total_image_area = gray.shape[0] * gray.shape[1]
    face_percentage = (total_face_area / total_image_area) * 100

    return mask, face_percentage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate face mask for a given image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the generated mask image.")
    args = parser.parse_args()

    mask, percentage = generate_face_mask(args.image_path)
    print(f"The face covers {percentage:.2f}% of the total image.")
    cv2.imwrite(args.save_path, mask)
