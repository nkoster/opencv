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

img = cv2.imread(arg, 1)
width, height = img.shape[1], img.shape[0]

if width > height:
    scale = 1000.0 / width
else:
    scale = 800.0 / height

resized = cv2.resize(img, ((int(width * scale)),
    (int(height * scale))), fx=scale, fy=scale,
    interpolation=cv2.INTER_LINEAR)

def classify(arg, c):
    print(arg, c)

for i in range(len(sys.argv)):
    if i > 1:
        print(sys.argv[i].split(':')[0] + ' ' + sys.argv[i].split(':')[1])

cv2.imshow(arg, resized)
person = 'unknown'
key = cv2.waitKey(0)
if key == ord('l'):
    person = 'laury'
if key == ord('n'):
    person = 'niels'
if key == ord('k'):
    person = 'kees'
if key == ord('m'):
    person = 'max'
if key == ord('d'):
    person = 'danny'
if key == ord('w'):
    person = 'kriz'
if key == ord('r'):
    person = 'ray'

cv2.destroyAllWindows()
if person != 'unknown':
    path = './images/' + person + '/' + randomword(6) + '.png'
    print(path)
    cv2.imwrite(path , resized)
else:
    print(arg)
