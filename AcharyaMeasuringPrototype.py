import cv2
import glob 
import Utlis as utlis
from ultralytics import YOLO
from PIL import Image

#? Path to use for predictions
imagePath = "Test Images/"

#? Path to use for saving the predictions
savingPath = "Predicted Images"

#? Model to use
modelName = "Models/best.pt"

#? Configs
runTraining = False
useReferencePoint = True;
epochsNumber = 10

#? Global Variables
allImages = []

def init():
  allImages = utlis.readPath(imagePath)
   
  imageNumber = 0

  for image in allImages:
    imageNumber = imageNumber + 1

    recognizeObjects(image)
    utlis.saveImage(image, f'{savingPath}', f"PredictedImage_{imageNumber}.jpg")

    cv2.imshow(f'Image {imageNumber}', image)
    
  
  cv2.waitKey(0)


def recognizeObjects(img):
  model = YOLO(modelName)

  if runTraining: model.train(data="data.yaml", epochs=epochsNumber)
  
  result = model.predict(img)[0]

  for box in result.boxes:
    label, x1, x2, y1, y2 = utlis.getObjectInformation(box, result)
    utlis.drawRectangle(img, label, x1, x2, y1, y2)
    

init()

#recognizedObjects = recognizeObjects(image)
#recognizedObjects.show()

