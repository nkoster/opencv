import cv2

video = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

a = 1
check = True

while check:
    a += 1
    check, frame = video.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.2, minNeighbors = 5)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 3)
        cv2.circle(frame, (x + w / 2, y + h / 2), (w / 2) - 1, (255, 255, 255), 1)
    cv2.imshow('video', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

print(a)
