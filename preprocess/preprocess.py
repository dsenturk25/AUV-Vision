import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("./preprocess/data/img4.png", cv2.IMREAD_UNCHANGED)


def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    sat = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 1]

    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # lower_blue = np.array([78, 158, 124])
    # upper_blue = np.array([138, 255, 255])

    # mask = cv2.inRange(image, lower_blue, upper_blue)

    # _, thresh = cv2.threshold(sat, 90, 255, 0)

    edges = cv2.Canny(image, 100, 255)

    kernel = np.zeros((10, 10), np.uint8)
    img_erosion = cv2.erode(edges, kernel, iterations=1)

    OPEN_KERNEL, CLOSE_KERNEL = np.ones((5, 5), np.uint8), np.ones((15, 15), np.uint8)

    morph = cv2.morphologyEx(cv2.bitwise_not(img_erosion), cv2.MORPH_OPEN, OPEN_KERNEL)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, CLOSE_KERNEL)

    morph = cv2.bitwise_not(morph)

    contours, hierarchy = cv2.findContours(
        morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    rectangle_images = []
    min_contour_area = 10000
    try:
        hierarchy = hierarchy[0]
    except:
        hierarchy = []
    height, width, _ = image.shape
    min_x, min_y = width, height
    max_x = max_y = 0
    for contour, hier in zip(contours, hierarchy):
        if cv2.contourArea(contour) > min_contour_area:
            (x, y, w, h) = cv2.boundingRect(contour)
            min_x, max_x = min(x, min_x), max(x + w, max_x)
            min_y, max_y = min(y, min_y), max(y + h, max_y)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 10)
            extracted_rectangle = image[y : y + h, x : x + w]
            rectangle_images.append(list([x, y, w, h]))
    if max_x - min_x > 0 and max_y - min_y > 0:
        pass

    titles = ["image", "morph"]

    images = [image, morph]

    # plt.figure(figsize=(10, 7))
    # for i in range(len(images)):
    #     plt.subplot(1, 2, i + 1)
    #     plt.title(titles[i])
    #     plt.imshow(images[i], cmap="gray")
    #     plt.axis("off")

    # plt.show()

    return rectangle_images, morph
