# Acharya Measuring Prototype

This prototype intends to discover the measurements of a house from images taken inside it through recognition of reference points with machine learning and with the aid of calculations and geometric techniques.

## Getting Started

To get the project running locally you'll have to run:

```sh
python3 AcharyaMeasuringPrototype.py
```

## Prerequisites

* Ultralytics Library

```sh
  pip install ultralytics
```

## How to use

You can use this project to recognize objects in images and discover the distances between the camera and those objects.

#### Folder Structure

* Models

Where the Machine Learning models used are.

* Predicted Images

Where the output of the predicted images is.

* Raw Images

Some example images that can be used to test the model

* Test Images

The actual images that the model will predict.

#### Configurations

In the code you have some variables that you can change depending on your use case.

``` python
testImagePath = "Test Images/" # Path to use for predictions
testImageDataPath = "Test Images/measurements.txt" # Path where is the images data

savingPath = "Predicted Images" # Path to use for saving the predictions
saveImagesAfterPrediction = False # If it's going to save the images after the prediction

modelName = "Models/best.pt"# Model to use
runTraining = False # If it's going to train the model
epochsNumber = 10 # Number of training cycles

resizeImage = False # If it's going to resize the image or use the original size
# Image size
imageWidth = 960
imageHeight = 560

# Camera settings used to predict the distance between camera and object
focalLength = 28
sensorHeight = 3.60
sensorWidth = 4.80
```