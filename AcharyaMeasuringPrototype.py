import os
import cv2
import cloudinary.api
import requests
import numpy as np
import json
from flask import Flask
from ultralytics import YOLO

import Utlis as utlis
import ImageProcessing as imageProcessing

# ? Path to use for predictions
testImagePath = "Test Images/"
testImageDataPath = "Test Images/measurements.txt"

# ? Path to use for saving the predictions
savingPath = "Predicted Images"

# ? Model to use
modelName = "Models/best.pt"
model = None

# ? Configs
serverless = True
runTraining = False
runPrediction = True
findVanishingPoints = False
saveImagesAfterPrediction = False
resizeImage = False
chooseWallBounderies = True

epochsNumber = 30
minimumWindowConfRate = 0.6

# ? Camera Settings
focalLength = 28

sensorHeight = 3.60
sensorWidth = 4.80

imageWidth = 960
imageHeight = 560

# ? Cloudinary Config Variables
cloudinaryFolder = "acharya-prototype-test"
cloudinaryFileName = "4_ykz60d"

# ? Global Variables

allImages = []
testImagesData = []
cachedMousePositionX = -1
cachedMousePositionY = -1
recognitionResults = None

app = Flask(__name__)

cloudinary.config(
    cloud_name="acharya",
    api_key="832265632616173",
    api_secret="pXd5M0xmCS0eXtTqtx058x4W1Oc"
)


@app.route("/")
def init(event, context):
    if serverless:
        print("Running prototype in serverless mode.")

    with app.app_context():
        try:
            imageToPredict = fetchDatabase()

            if imageToPredict is None:
                response = {
                    'message': 'Image was not successfully processed.',
                    'result': None
                }

                return {
                    "statusCode": 200,
                    "headers": {
                        'Content-Type': "application/json"
                    },
                    "body": json.dumps(response)
                }

            imageAfterPrediction = startServerlessPrototype(imageToPredict)

            if imageToPredict is not None and imageAfterPrediction.shape[0] > 0 and imageAfterPrediction.shape[1] > 0 and imageAfterPrediction.shape[2] > 0:
                print("This image is valid for base64 conversion.")
            else:
                print("This image is not valid for base64 conversion.")

            imageAfterPrediction_base64 = utlis.convertCV2ToBase64(
                imageAfterPrediction)

            response = {
                'message': 'Image processed successfully.',
                'result': imageAfterPrediction_base64
            }

            return {
                "statusCode": 200,
                "headers": {
                    'Content-Type': "application/json"
                },
                "body": json.dumps(response)
            }
        except Exception as e:
            return {
                'statusCode': 500,
                'headers': {
                    'Content-Type': 'application/json',
                },
                'body': json.dumps(e)
            }


@app.route("/health")
def health(event, context):
    print("The server is healthy.")

    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'The server is healthy.'})
    }


def fetchDatabase():
    publicID = f"{cloudinaryFolder}/{cloudinaryFileName}"

    imageInfo = cloudinary.api.resource(publicID)
    imageURL = imageInfo.get("url")

    response = requests.get(imageURL)

    if response.status_code == 200:
        image_bytes = np.frombuffer(response.content, np.uint8)
        imageCV2 = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if imageCV2 is not None:
            print("Fetched and decoded the image successfully.")
            return imageCV2
        else:
            print("Failed to decode the image.")
            return None
    else:
        print("Failed to fetch the image.")
        return None

    # TODO fetch cloudinary database and fetch the image to predict
    # TODO return then that image


def startServerlessPrototype(image):
    global model, imageName, imageHeight, imageWidth, recognitionResults

    model = YOLO(modelName)
    imageName = 'Image 1'

    if not resizeImage:
        imageHeight, imageWidth, _ = image.shape

    recognitionResults = recognizeObjects(
        image, 1)

    return image


def startPrototype():
    global testImagesData, allImages, imageWidth, imageHeight, model, recognitionResults

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
        imageName = f'Image {imageNumber}'

        if not resizeImage:
            imageHeight, imageWidth, _ = image.shape

        if (findVanishingPoints):
            recognizeVanishingPoints(image, imageNumber)

        if runPrediction:
            recognitionResults = recognizeObjects(
                image, imageNumber)

        if saveImagesAfterPrediction:
            utlis.saveImage(image, f'{savingPath}',
                            f"PredictedImage_{imageNumber}.jpg")

        cv2.imshow(imageName, image)

        if (chooseWallBounderies):
            cv2.setMouseCallback(
                imageName, chooseWallBounderies, (image, imageName, recognitionResults))

    cv2.waitKey(0)


def trainModel(model):
    print("Started the training phase.")
    model.train(data="data.yaml", epochs=epochsNumber)


def recognizeObjects(image, imageNumber):
    model = YOLO(modelName)
    result = model.predict(image)[0]

    for box in result.boxes:
        className, conf, x1, y1, x2, y2 = utlis.getObjectInformation(
            box, result)

        if (conf <= minimumWindowConfRate):
            print(
                f'Recognized one {className} with less than 50% confidence rate.')
            continue

        windowDistance = ""

        if (className == "window"):
            windowDistance = recognizeWindowDistance(imageNumber, y1, y2)

        utlis.drawClassRectangle(image, className, conf, x1,
                                 y1, x2, y2, windowDistance)

    return result.boxes


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


def recognizeWallHeight(image, imageName, windowHeightPixels, windowHeightCM, point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    if y1 > y2:
        wallHeightPixels = y1 - y2
    else:
        wallHeightPixels = y2 - y1

    print(f'Window height in CM: {windowHeightCM}')
    print(f'Window height in Pixels: {windowHeightPixels}')
    print(f'Wall height in Pixels: {wallHeightPixels}')

    wallHeightCM = round((int(windowHeightCM) * wallHeightPixels) /
                         int(windowHeightPixels))

    # ? Draw information rectangle
    print(f'Wall height in CM: {wallHeightCM}')

    cv2.imshow(imageName, image)


def chooseWallBounderies(event, x, y, flags, param):
    global cachedMousePositionX, cachedMousePositionY

    if event == cv2.EVENT_LBUTTONDOWN:
        (image, imageName, recognitionResults) = param

        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow(imageName, image)

        if cachedMousePositionX == -1 or cachedMousePositionY == -1:
            print("First point for the wall choosen. Choose another one.")
            cachedMousePositionX, cachedMousePositionY = x, y
        else:
            print("Two points choosen.")

            point1 = (cachedMousePositionX, cachedMousePositionY)
            point2 = (x, y)

            cv2.line(image, point1, point2, (0, 0, 255), 5)
            cv2.imshow(imageName, image)

            for box in recognitionResults:
                conf = round(box.conf[0].item(), 2)

                if (conf >= minimumWindowConfRate):
                    cords = box.xyxy[0].tolist()
                    cords = [round(x) for x in cords]
                    x1, y1, x2, y2 = cords

                    print(
                        f'Choosing window with coordinates (({x1}, {y1}), ({x2, y2}))')

                    if y1 > y2:
                        windowHeightPixels = y1 - y2
                    else:
                        windowHeightPixels = y2 - y1

                    _, _, windowHeightCM = utlis.getWindowInformation(
                        imageName.split()[-1], testImagesData)

                    recognizeWallHeight(
                        image, imageName, windowHeightPixels, windowHeightCM, point1, point2)

                    return

            print(
                "There is no available windows in the image. Skipping wall height recognition.")


if not serverless:
    print("Running prototype in non-serverless mode.")
    startPrototype()
else:
    print("Running prototype in serverless mode.")

# recognizedObjects = recognizeObjects(image)
# recognizedObjects.show()
