import cv2
import Utlis as utlis
from ultralytics import YOLO

# ? Path to use for predictions
testImagePath = "Test Images/"
testImageDataPath = "Test Images/measurements.txt"

# ? Path to use for saving the predictions
savingPath = "Predicted Images"

# ? Model to use
modelName = "Models/best.pt"
modelName = "runs/detect/train5/weights/best.pt"
model = None

# ? Configs
runTraining = True
runPrediction = True
saveImagesAfterPrediction = False
resizeImage = False
epochsNumber = 50

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

        if not resizeImage:
            imageHeight, imageWidth, _ = image.shape

        imageNumber = imageNumber + 1

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
            print("Recognized one window with less than 50% confidence rate.")
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


def trainModel(model):
    print("Started the training phase.")
    model.train(data="data.yaml", epochs=epochsNumber)


init()

# recognizedObjects = recognizeObjects(image)
# recognizedObjects.show()
