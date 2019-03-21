import sys
import cv2

arg = 0
if len(sys.argv) > 1:
    arg = sys.argv[1]

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

img = cv2.imread(arg, 1)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.2, minNeighbors = 5)

# print(img)

for fx, fy, fw, fh in faces:
    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (0, 0, 0), 3)
    roi_gray = gray_img[fy: fy + fh, fx: fx + fw]
    roi_color = img[fy: fy + fh, fx: fx + fw]
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.25, minNeighbors=5)
    for ex, ey, ew, eh in eyes:
        cv2.circle(roi_color, (ex + (ew / 2), ey + (eh / 2)), ew, (255, 255, 255), 3)
# for x, y, w, h in faces:
#     # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 3)
#     cv2.circle(img, (x + w / 2, y + h / 2), (w / 2) + 5, (255, 255, 255), 3)

# print(img)

scale = 0.6

resized = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

cv2.imshow('test', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
