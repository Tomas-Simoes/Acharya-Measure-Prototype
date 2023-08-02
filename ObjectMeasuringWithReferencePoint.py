import cv2
import DetectPaper
from ultralytics import YOLO
from object_detector import *


useReferencePoint = True;
imagePath = "images/wallWithMarker.jpeg" if useReferencePoint else "images/wall.jpeg" 

def init():
  image = getImage()
  detectPaperImage = image.copy()

  recognizeObjects(image)
  
  detectPaperImage = DetectPaper.thresholdImage(detectPaperImage)
  detectPaperImage = DetectPaper.detectEdges(detectPaperImage, True)

  cv2.imshow("Wall Image", image)
  cv2.waitKey(0)


def getImage():
  _img = cv2.imread(imagePath)
  _img = cv2.resize(_img, (960, 540))
  return _img


def recognizeObjectsInHomogeneousBackground(img):
  detector = HomogeneousBgDetector()
  objects_coordinates = detector.detect_objects(img)

  for coordinate in objects_coordinates:
    cv2.polylines(img, [coordinate], True, (255, 0, 255))
    (x, y), (w, h), angle = cv2.minAreaRect(coordinate)
    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

def recognizeObjects(img):
  model = YOLO("yolov8m-seg.pt")
  #model.train(data="data.yaml", epochs=30)
  result = model.predict(img)[0]
  box = result.boxes[0]
  
  for box in result.boxes:
    class_name = result.names[box.cls[0].item()]
    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    x1, y1, x2, y2 = cords
    conf = round(box.conf[0].item(), 2)

    boundingBox_label = class_name + " " + str(conf) + "%"
    drawRectangle(img, boundingBox_label, x1, y1, x2, y2)
    

def drawRectangle(img, label, x1, y1, x2, y2):
  boundingBox = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

  (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
  boundingBox = cv2.rectangle(boundingBox, (x1, y1 - 20), (x1 + text_w, y1), (255, 0, 0), -1)
  boundingBox = cv2.putText(boundingBox, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

init()

#recognizedObjects = recognizeObjects(image)
#recognizedObjects.show()

