import cv2

video = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

a = 1
check = True

while check:
    a += 1
    check, frame = video.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.2, minNeighbors = 5)
    for fx, fy, fw, fh in faces:
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 0), 3)
        roi_gray = gray_img[fy : fy + fh, fx : fx + fw]
        roi_color = frame[fy : fy + fh, fx : fx + fw]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor = 1.25, minNeighbors = 5)
        for ex, ey, ew, eh in eyes:
            cv2.circle(roi_color, (ex + (ew / 2), ey + (eh / 2)), ew, (255, 255, 255), 3)
    cv2.imshow('video', frame)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

print(a)
