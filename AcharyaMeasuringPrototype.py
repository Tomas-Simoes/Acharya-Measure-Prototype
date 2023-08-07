import cv2
import glob
import Utlis as utlis
from ultralytics import YOLO
from PIL import Image

# ? Path to use for predictions
testImagePath = "Test Images/"
testImageDataPath = "Test Images/measurements.txt"

# ? Path to use for saving the predictions
savingPath = "Predicted Images"

# ? Model to use
modelName = "Models/best.pt"

# ? Configs
runTraining = False
saveImagesAfterPrediction = False
epochsNumber = 10

# ? Camera Settings
focalLength = 25

sensorHeight = 14.9
sensorWidth = 22.3

imageWidth = 960
imageHeight = 560

# ? Global Variables
allImages = []
testImagesData = []


def init():
    global testImagesData
    global allImages

    allImages = utlis.readPath(testImagePath, imageWidth, imageHeight)

    with open(testImageDataPath, "r") as configFile:
        testImagesData = configFile.readlines()

    imageNumber = 0

    for image in allImages:
        imageNumber = imageNumber + 1

        recognizeObjects(image, imageNumber)

        if saveImagesAfterPrediction:
            utlis.saveImage(image, f'{savingPath}',
                            f"PredictedImage_{imageNumber}.jpg")

        cv2.imshow(f'Image {imageNumber}', image)

    cv2.waitKey(0)


def recognizeObjects(image, imageNumber):
    model = YOLO(modelName)

    if runTraining:
        model.train(data="data.yaml", epochs=epochsNumber)

    result = model.predict(image)[0]

    for box in result.boxes:
        className, conf, x1, x2, y1, y2 = utlis.getObjectInformation(
            box, result)

        windowDistance = ""
        if (className == "window"):
            windowDistance = recognizeWindowDistance(imageNumber, y1, y2)

        utlis.drawRectangle(image, className, conf, x1,
                            x2, y1, y2, windowDistance)


def recognizeWindowDistance(imageNumber, y1, y2):
    _, windowWidth, windowHeight = utlis.getWindowInformation(
        imageNumber, testImagesData)

    windowDistance = (focalLength * (float(windowHeight) * 10)
                      * imageHeight) / ((y1 - y2) * sensorHeight)
    print(
        f'Calculating formula with height: {focalLength} * {float(windowHeight) * 10} * {imageHeight} / {y1 - y2} * {sensorHeight} = {windowDistance}')

    windowDistance = round(windowDistance) / 10

    return windowDistance


init()

# recognizedObjects = recognizeObjects(image)
# recognizedObjects.show()
