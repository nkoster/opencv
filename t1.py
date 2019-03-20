import cv2

img = cv2.imread('adri.jpg', 1)

# print(img)
# print(type(img))
# print(img.shape)

scale = 3
resized = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
cv2.imshow('adri', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
