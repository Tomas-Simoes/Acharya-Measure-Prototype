import cv2

def thresholdImage(img, output_image = False):
    img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

    T_, thresholded = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if output_image: cv2.imshow('Thresholded Image', thresholded)


    return thresholded

def detectEdges(img, output_image = False) :
    edges = cv2.Canny(img, 50, 150, apertureSize = 3)
    if output_image: cv2.imshow('Edged Image', edges)

    return edges