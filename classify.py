import sys
import cv2
import random
import string

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

arg = ''
if len(sys.argv) > 1:
    arg = sys.argv[1]
else:
    print('No image provided.')
    exit(1)

face_cascade = cv2.CascadeClassifier('./haar/haarcascade_frontalface_default.xml')

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

def classify(arg, c):
    print(arg, c)

if len(faces) > 0:
    for fx, fy, fw, fh in faces:
        cv2.rectangle(resized, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 3)
        roi_gray = gray_img[fy: fy + fh, fx: fx + fw]
        roi_color = resized[fy: fy + fh, fx: fx + fw]
    cv2.imshow(arg, resized)
    person = 'unknown'
    key = cv2.waitKey(0)
    if key == ord('l'):
        person = 'laury'
    if key == ord('n'):
        person = 'niels'
    cv2.destroyAllWindows()
    if person != 'unknown':
        path = './images/' + person + '/' + randomword(6) + '.png'
        print(str(len(faces)) + ' ' + path)
        cv2.imwrite(path , resized)
