import cv2

image = cv2.imread("./test.jpg", cv2.IMREAD_UNCHANGED)
dsize = (640, 480)
image = cv2.resize(image, dsize)
cv2.imwrite("test_resized.png",image)