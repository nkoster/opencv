import cv2
import sys
import pickle

arg = 0
if len(sys.argv) > 1:
    arg = sys.argv[1]
video = cv2.VideoCapture(arg)
face_cascade = cv2.CascadeClassifier('./haar/haarcascade_frontalface_alt2.xml')
width, height = 0, 0
check, frame = video.read()
gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
width, height = frame.shape[1], frame.shape[0]

if width > height:
    scale = 1000.0 / width
else:
    scale = 800.0 / height

recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load('trainer.yml')

labels = {}
with open('label.pickle', 'rb') as f:
    ori_labels = pickle.load(f)
    labels = {v:k for k,v in ori_labels.items()}

while check:
    resized = cv2.resize(frame, ((int(width * scale)),
        (int(height * scale))), fx=scale, fy=scale,
        interpolation=cv2.INTER_LINEAR)
    gray_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
    for fx, fy, fw, fh in faces:
        cv2.rectangle(resized, (fx, fy), (fx + fw, fy + fh), (100, 255, 100), 3)
        roi_gray = gray_img[fy: fy + fh, fx: fx + fw]
        roi_color = resized[fy: fy + fh, fx: fx + fw]
        id_, conf = recognizer.predict(roi_gray)
        print(labels[id_])
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[id_]
        stroke = 2
        color = (100, 255, 100)
        cv2.putText(resized, name, (fx, fy-10), font, 1, color, stroke, 1)
    cv2.imshow('video', resized)
    key = cv2.waitKey(20)
    if key == ord('q'):
        break
    check, frame = video.read()

video.release()
cv2.destroyAllWindows()
print(width, height, scale)
