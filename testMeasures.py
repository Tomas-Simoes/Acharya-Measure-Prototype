import cv2
import numpy as np

# Load pre-trained wall detection model
# Replace 'model_path' with the path to your pre-trained model
model_path = 'path/to/your/pretrained/model'
model = cv2.dnn.readNetFromDarknet(model_path)

# Load image
image_path = 'path/to/your/image.jpg'
image = cv2.imread(image_path)

# Preprocess image
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
model.setInput(blob)

# Perform wall detection
layer_names = model.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]
outputs = model.forward(output_layers)

# Process outputs to extract bounding boxes
# (Replace this with your specific processing based on model output format)
boxes = []
confidences = []
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Adjust confidence threshold as needed
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            w = int(detection[2] * image.shape[1])
            h = int(detection[3] * image.shape[0])
            x = center_x - w // 2
            y = center_y - h // 2
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))

# Draw bounding boxes on the image
for box in boxes:
    x, y, w, h = box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the results
cv2.imshow('Wall Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()