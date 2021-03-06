import cv2
import os
import sys
import numpy as np
from PIL import Image
import pickle

arg = ''
if len(sys.argv) > 1:
    image_dir = sys.argv[1]
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, 'images')

face_cascade = cv2.CascadeClassifier('./haar/haarcascade_frontalface_alt2.xml')

recognizer = cv2.createLBPHFaceRecognizer()
# print(image_dir)

y_labels = []
x_train = []

current_id = 0
label_ids = {}

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.upper().endswith('PNG') or file.upper().endswith('JPG'):
            image = os.path.join(root, file)
            label = os.path.basename(root).replace(' ', '-').lower()
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            #print(label_ids)
            pil_image = Image.open(image).convert('L') # Grayscale
            image_array = np.array(pil_image, 'uint8')
            width, height = image_array.shape[1], image_array.shape[0]
            if width > height:
                scale = 1000.0 / width
            else:
                scale = 800.0 / height
            resized = cv2.resize(image_array, ((int(width * scale)),
                (int(height * scale))), fx=scale, fy=scale,
                interpolation=cv2.INTER_LINEAR)
            faces = face_cascade.detectMultiScale(resized, scaleFactor=1.2, minNeighbors=5)
            roi = None
            for fx, fy, fw, fh in faces:
                roi = resized[fy: fy + fh, fx: fx + fw]
            # print(label, image, width, height, scale, len(faces))
            y_labels.append(id_)
            x_train.append(roi)

# print(y_labels)
# print(x_train)

with open('label.pickle', 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save('trainer.yml')
