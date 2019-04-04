import sys
import cv2

arg = 0
if len(sys.argv) > 1:
    arg = sys.argv[1]

face_cascade = cv2.CascadeClassifier('./haar/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haar/haarcascade_eye.xml')

img = cv2.imread(arg, 1)
width, height = img.shape[1], img.shape[0]

if width > height:
    scale = 1000.0 / width
else:
    scale = 800.0 / height

resized = cv2.resize(img, ((int(width * scale)),
    (int(height * scale))), fx=scale, fy=scale,
    interpolation=cv2.INTER_LINEAR)

gray_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.2, minNeighbors = 5)

for fx, fy, fw, fh in faces:
    cv2.rectangle(resized, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 3)
    roi_gray = gray_img[fy: fy + fh, fx: fx + fw]
    roi_color = resized[fy: fy + fh, fx: fx + fw]
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.25, minNeighbors=5)
    for ex, ey, ew, eh in eyes:
        cv2.circle(roi_color, (ex + (ew / 2), ey + (eh / 2)), ew, (255, 255, 255), 3)
    cv2.imwrite('./roi.png', roi_color)

cv2.imshow(arg, resized)
while cv2.waitKey(0) != ord('q'):
    continue
cv2.destroyAllWindows()
