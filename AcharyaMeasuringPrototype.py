import os
import cv2
import math
import numpy as np
import Utlis as utlis
from ultralytics import YOLO
import ImageProcessing as imageProcessing

# ? Path to use for predictions
testImagePath = "Test Images/"
testImageDataPath = "Test Images/measurements.txt"

# ? Path to use for saving the predictions
savingPath = "Predicted Images"

# ? Model to use
#modelName = "Models/best.pt"
modelName = "Models/bestWallWindow.pt"
#model = None

# ? Configs
runTraining = True
runPrediction = False
findVanishingPoints = True
saveImagesAfterPrediction = True
resizeImage = False
epochsNumber = 31

# ? Camera Settings
focalLength = 28
# focalLength = 55

sensorHeight = 3.60
sensorWidth = 4.80

imageWidth = 960
imageHeight = 560

# ? Global Variables
allImages = []
testImagesData = []


def init():
    # utlis.changeImageID("datasets/train/labels", "0", "1", "part 2")
    # utlis.changeImageID("datasets/val/labels", "0", "1", "part 2")
    startPrototype()


def startPrototype():
    global testImagesData, allImages, imageWidth, imageHeight, model

    model = YOLO(modelName)

    allImages = utlis.readPath(
        testImagePath, resizeImage, imageWidth, imageHeight)

    with open(testImageDataPath, "r") as configFile:
        testImagesData = configFile.readlines()

    imageNumber = 0

    if runTraining:
        trainModel(model)

    for image in allImages:
        imageNumber = imageNumber + 1

        if not resizeImage:
            imageHeight, imageWidth, _ = image.shape

        if (findVanishingPoints):
            recognizeVanishingPoints(image, imageNumber)

        if runPrediction:
            recognizeObjects(image, imageNumber)

        if saveImagesAfterPrediction:
            utlis.saveImage(image, f'{savingPath}',
                            f"PredictedImage_{imageNumber}.jpg")

        cv2.imshow(f'Image {imageNumber}', image)

    cv2.waitKey(0)


def recognizeObjects(image, imageNumber):
    model = YOLO(modelName)
    result = model.predict(image)[0]

    for box in result.boxes:
        className, conf, x1, y1, x2, y2 = utlis.getObjectInformation(
            box, result)

        if (conf <= 0.5):
            print(
                f'Recognized one {className} with less than 50% confidence rate.')
            continue

        windowDistance = ""
        if (className == "window"):
            windowDistance = recognizeWindowDistance(imageNumber, y1, y2)

        utlis.drawRectangle(image, className, conf, x1,
                            y1, x2, y2, windowDistance)


def recognizeWindowDistance(imageNumber, y1, y2):
    _, windowWidth, windowHeight = utlis.getWindowInformation(
        imageNumber, testImagesData)

    windowWidth = windowWidth * 10
    pixelSize = 0
    if y1 > y2:
        pixelSize = y1 - y2
    else:
        pixelSize = y2 - y1

    windowDistance = (focalLength * float(windowHeight) *
                      imageHeight) / (pixelSize * sensorHeight)

    print(
        f'Calculating formula with height: {focalLength} * {float(windowHeight) * 10} * {imageHeight} / {pixelSize} * {sensorHeight} = {windowDistance}')

    windowDistance = round(windowDistance)

    return windowDistance


def recognizeVanishingPoints(image, imageNumber):
    preProcessedImage = imageProcessing.edgeImageConverter(imageProcessing.blurImageConverter(
        imageProcessing.grayImageConverter(image)))

    imageLines = cv2.HoughLinesP(preProcessedImage, 1, np.pi / 180, 50, 10, 15)

    if imageLines is None:
        print("There are no lines in the image.")
        return

    i = 0
    for imageLine in imageLines:
        thisLine = imageLine[0]

        cv2.line(image, (thisLine[0], thisLine[1]),
                 (thisLine[2], thisLine[3]), (0, 255, 0), 3, 10)

        i += 1

    # cv2.imshow(f'Pre-processed Image {imageNumber}', preProcessedImage)


def trainModel(model):
    print("Started the training phase.")
    print(os.getcwd())
    model.train(data="data.yaml", epochs=epochsNumber)


init()

# recognizedObjects = recognizeObjects(image)
# recognizedObjects.show()
