import cv2
import os
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_dir = os.path.join(BASE_DIR, 'images')

face_cascade = cv2.CascadeClassifier('./haar/haarcascade_frontalface_alt2.xml')

print(image_dir)

y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.upper().endswith('PNG') or file.upper().endswith('JPG'):
            image = os.path.join(root, file)
            label = os.path.basename(root).replace(' ', '-').lower()
            pil_image = Image.open(image).convert('L') # Grayscale
            image_array = np.array(pil_image, 'uint8')
            width, height = image_array.shape[1], image_array.shape[0]
            scale = 1000.0 / width
            resized = cv2.resize(image_array, ((int(width * scale)),
                (int(height * scale))), fx=scale, fy=scale,
                interpolation=cv2.INTER_LINEAR)
            faces = face_cascade.detectMultiScale(resized, scaleFactor=1.2, minNeighbors=5)
            print(label, image, width, height, scale, len(faces))
            # y_labels.append(label)
            # x_train.append(image)
