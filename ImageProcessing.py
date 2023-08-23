import cv2


def grayImageConverter(image):
    return cv2.cvtcolor(image, cv2.COLOR_BGR2GRAY)


def blurImageConverter(image):
    return cv2.GaussianBlur(image, (5, 5), 1)


def edgeImageConverter(image):
    return cv2.Canny(image, 40, 255)
