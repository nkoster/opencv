import sys
import cv2
import random
import string
import os

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

cv2.imshow(arg, resized)
person = 'unknown'
classified = False
key = cv2.waitKey(0)

for i in range(len(sys.argv)):
    if i > 1:
        classifyKey = sys.argv[i].split(':')[0]
        classifyName = sys.argv[i].split(':')[1]
        path = './images/'
        if not os.path.exists(path):
            print('-- create ' + path)
            os.mkdir(path)
        if (key == ord(classifyKey)):
            classified = True
            path = path + classifyName + '/'
            if not os.path.exists(path):
                print('-- create ' + path)
                os.mkdir(path)
            cv2.destroyAllWindows()
            path = path + randomword(6) + '.png'
            cv2.imwrite(path , resized)
            print('-- save ' + path)

if not classified:
    print('-- ' + arg)
