import cv2
import os
import numpy as np
from PIL import Image

cachedPredictionPath = ""

def drawRectangle(img, label, x1, y1, x2, y2):
  boundingBox = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

  (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
  boundingBox = cv2.rectangle(boundingBox, (x1, y1 - 20), (x1 + text_w, y1), (255, 0, 0), -1)
  boundingBox = cv2.putText(boundingBox, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def getObjectInformation(box, result):
  class_name = result.names[box.cls[0].item()]
  cords = box.xyxy[0].tolist()
  cords = [round(x) for x in cords]
  x1, y1, x2, y2 = cords
  conf = round(box.conf[0].item(), 2)

  boundingBox_label = class_name + " " + str(conf) + "%"
  return boundingBox_label, x1, y1, x2, y2

def readImage(path):
  _img = cv2.imread(path)
  _img = cv2.resize(_img, (960, 540))
  return _img

def saveImage(image, path, fileName):
  try:
    global cachedPredictionPath

    rootPath = os.path.dirname(os.path.abspath(__file__))
    outputPath = os.path.join(rootPath, path)

    predictionsFolders = [item for item in os.listdir(outputPath) if os.path.isdir(os.path.join(outputPath, item))]
    
    if not cachedPredictionPath:
      for predictionFolder in predictionsFolders:
        lastPredictionFolderNumber = ''.join(filter(str.isdigit, predictionFolder[-2:]))
      
      if len(predictionsFolders) == 0:
          lastPredictionFolderNumber = "0"

      outputPath = os.path.join(outputPath, f'Prediction {int(lastPredictionFolderNumber) + 1}')
      cachedPredictionPath = outputPath

      print(outputPath)
      print(lastPredictionFolderNumber)
    

    os.makedirs(cachedPredictionPath, exist_ok=True)
    
    outputPath = os.path.join(cachedPredictionPath, fileName)

    saved = cv2.imwrite(outputPath, image)
    
    if saved : print(f'The image was sucessfully saved in path "{outputPath}"')
    else: print(f'The image was not sucessfully saved in path "{outputPath}"')

  except Exception as e:
    print(f'Error occurred while saving the image: {e}')



def readPath(path):
  _allImagesInPath = []
  _imagesExtensions = [".JPG", ".jpeg", ".png", ""]

  for _fileName in os.listdir(path):
    if any(_fileName.endswith(extension) for extension in _imagesExtensions):  
      _thisImagePath = os.path.join(path, _fileName)

      _thisImage = cv2.imread(_thisImagePath)
      _thisImage = cv2.resize(_thisImage, (960, 540))

      _allImagesInPath.append(_thisImage)

  return _allImagesInPath
