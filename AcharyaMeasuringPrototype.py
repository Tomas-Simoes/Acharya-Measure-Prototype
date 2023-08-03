import cv2
import glob 
import Utlis as utlis
from ultralytics import YOLO


#? Path to use for predictions
imagePath = "Test Images/"

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
    cv2.imshow(f'Image {imageNumber}', image)
    
  
  cv2.waitKey(0)


def recognizeObjects(img):
  model = YOLO(modelName)

  model.train(data="data.yaml", epochs=epochsNumber)
  
  result = model.predict(img)[0]

  for box in result.boxes:
    class_name = result.names[box.cls[0].item()]
    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    x1, y1, x2, y2 = cords
    conf = round(box.conf[0].item(), 2)

    boundingBox_label = class_name + " " + str(conf) + "%"
    utlis.drawRectangle(img, utlis.getObjectInformation)
    


init()

#recognizedObjects = recognizeObjects(image)
#recognizedObjects.show()

