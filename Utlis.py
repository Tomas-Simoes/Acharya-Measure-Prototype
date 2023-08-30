import cv2
import os
# from AcharyaMeasuringPrototype import imageHeight
# from AcharyaMeasuringPrototype import imageWidth

cachedPredictionPath = ""


def drawClassRectangle(img, className, conf, x1, y1, x2, y2, windowDistance):
    label = className + " " + str(conf) + "%"

    print(x1, y1)
    boundingBox = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    (text_w, text_h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    boundingBox = cv2.rectangle(
        boundingBox, (x1, y1 - 20), (x1 + text_w, y1), (255, 0, 0), -1)
    boundingBox = cv2.putText(boundingBox, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if className == "window":
        boundingBox = cv2.putText(
            boundingBox, f'Window Distance: {str(windowDistance)} mm', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def getObjectInformation(box, result):
    class_name = result.names[box.cls[0].item()]
    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    x1, y1, x2, y2 = cords
    conf = round(box.conf[0].item(), 2)

    return class_name, conf, x1, y1, x2, y2


def getWindowInformation(imageNumber, data):
    # print(
    #    f'Reading distance between camera and window for image number {imageNumber}')

    thisWindowData = ""

    for imageData in data:
        if imageData[0] == str(imageNumber):
            thisWindowData = imageData.rstrip('\n')

    # if thisWindowData != "":
    #    print(f'Found windows data for this image "{thisWindowData}".')
    # else:
    #    print(
    #        "Didn't found windows data for this image. Skiping window distance recognition.")
    #    return 0, 0, 0

    newInformation = ""
    allExtractedInformation = []

    for character in thisWindowData:
        readCharacters = character != " "

        if (readCharacters):
            newInformation = newInformation + character
        else:
            allExtractedInformation.append(newInformation)
            newInformation = ""

    allExtractedInformation.append(newInformation)
    return allExtractedInformation


def saveImage(image, path, fileName):
    try:
        global cachedPredictionPath

        rootPath = os.path.dirname(os.path.abspath(__file__))
        outputPath = os.path.join(rootPath, path)

        predictionsFolders = [item for item in os.listdir(
            outputPath) if os.path.isdir(os.path.join(outputPath, item))]

        if not cachedPredictionPath:
            for predictionFolder in predictionsFolders:
                lastPredictionFolderNumber = ''.join(
                    filter(str.isdigit, predictionFolder[-2:]))

            if len(predictionsFolders) == 0:
                lastPredictionFolderNumber = "0"

            outputPath = os.path.join(
                outputPath, f'Prediction {int(lastPredictionFolderNumber) + 1}')
            cachedPredictionPath = outputPath

        os.makedirs(cachedPredictionPath, exist_ok=True)

        outputPath = os.path.join(cachedPredictionPath, fileName)

        saved = cv2.imwrite(outputPath, image)

        if saved:
            print(f'The image was sucessfully saved in path "{outputPath}"')
        else:
            print(
                f'The image was not sucessfully saved in path "{outputPath}"')

    except Exception as e:
        print(f'Error occurred while saving the image: {e}')


def readPath(path, resizeImage, imageNewWidth, imageNewHeight):
    _allImagesInPath = []
    _imagesExtensions = [".jpg", ".jpeg", ".png"]

    for _fileName in os.listdir(path):
        if any(_fileName.endswith(extension) or _fileName.endswith(extension.upper()) for extension in _imagesExtensions):
            _thisImagePath = os.path.join(path, _fileName)

            _thisImage = cv2.imread(_thisImagePath)

            if resizeImage:
                _thisImage = cv2.resize(
                    _thisImage, (imageNewWidth, imageNewHeight))

            _allImagesInPath.append(_thisImage)

    return _allImagesInPath


def readImage(path, resizeImage, imageNewWidth, imageNewHeight):
    _img = cv2.imread(path)

    if resizeImage:
        _img = cv2.resize(_img, (imageNewWidth, imageNewHeight))


def changeImageID(path, currentId, futureID, initialString):
    _imagesExtensions = [".txt"]

    for _fileName in os.listdir(path):
        if any(_fileName.endswith(extension) or _fileName.endswith(extension.upper()) for extension in _imagesExtensions):
            _thisTextPath = os.path.join(path, _fileName)

            print(initialString)
            print(_fileName)
            print(_fileName.startswith(initialString))

            if _fileName.startswith(initialString):
                with open(_thisTextPath, 'r') as file:
                    first_character = file.read(1)
                    print(currentId)
                    print(first_character)
                    print(first_character == currentId)
                    if str(first_character) == currentId:
                        print("ya")
                        rest_of_content = file.read()
                        rest_of_content[1:]
                        print(rest_of_content)

                        new_content = futureID + rest_of_content
                        print(new_content)
                        new_file_path = os.path.join(
                            path, _fileName)  # New file path
                        with open(new_file_path, 'w') as new_file:
                            new_file.write(new_content)

                        # Optionally, you can delete the old file if needed
                        # os.remove(_thisTextPath)
