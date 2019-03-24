import cv2
import sys

arg = 0
if len(sys.argv) > 1:
    arg = sys.argv[1]
video = cv2.VideoCapture(arg)
face_cascade = cv2.CascadeClassifier('./haar/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('./haar/haarcascade_eye.xml')
width, height = 0, 0
check, frame = video.read()
gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
width, height = frame.shape[1], frame.shape[0]
scale = 1000.0 / width

while check:
    resized = cv2.resize(frame, ((int(width * scale)),
        (int(height * scale))), fx=scale, fy=scale,
        interpolation=cv2.INTER_LINEAR)
    gray_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
    for fx, fy, fw, fh in faces:
        cv2.rectangle(resized, (fx, fy), (fx + fw, fy + fh), (0, 0, 0), 3)
        roi_gray = gray_img[fy: fy + fh, fx: fx + fw]
        roi_color = resized[fy: fy + fh, fx: fx + fw]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.25, minNeighbors=5)
        for ex, ey, ew, eh in eyes:
            cv2.circle(roi_color, (ex + (ew / 2), ey + (eh / 2)), ew, (255, 255, 255), 3)
    cv2.imshow('video', resized)
    key = cv2.waitKey(20)
    if key == ord('q'):
        break
    check, frame = video.read()

video.release()
cv2.destroyAllWindows()
print(width, height, scale)
