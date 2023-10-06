
import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("./preprocess/data/img4.png", cv2.IMREAD_UNCHANGED)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
sat = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 2]

hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
lower_blue= np.array([78,158,124])
upper_blue = np.array([138,255,255])

mask = cv2.inRange(image,lower_blue,upper_blue)


_, thresh = cv2.threshold(sat, 90, 255, 0)

# contours, hierarchy = cv2.findContours(
#     cv2.bitwise_not(morph), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
# )

# cv2.drawContours(image, contours, -1, (0, 255, 0), 5, cv2.LINE_AA)


edges = cv2.Canny(image, 100, 255)


kernel = np.zeros((10, 10) , np.uint8)
img_erosion = cv2.erode (edges , kernel , iterations=1)

OPEN_KERNEL, CLOSE_KERNEL = np.ones((25, 25), np.uint8), np.ones((10, 10), np.uint8)

morph = cv2.morphologyEx(cv2.bitwise_not(img_erosion), cv2.MORPH_OPEN, OPEN_KERNEL)
morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, CLOSE_KERNEL)

morph = cv2.bitwise_not(morph)

titles = ["raw", "sat", "morph", "mask", "canny"]

images = [image, sat, morph, mask, img_erosion]

plt.figure(figsize=(15, 10))
for i in range(len(images)):
    
    plt.subplot(2, 3, i + 1)
    plt.title(titles[i])
    plt.imshow(images[i], cmap="gray")
    plt.axis("off")

plt.show()
