from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import os
import cv2
import numpy as np

images_directory = "images/"

def init():
    all_images_path = get_images_path()
    all_images_features = np.empty((0, 0))

    for image_path in all_images_path:
        print("Reading features for: " + image_path)

        new_image_features = extract_image_features(image_path)
        print("Image features: " + str(new_image_features))
        
        all_images_features = np.append(all_images_features, new_image_features)
        #new_image_features = new_image_features.reshape(1, -1)

    all_images_features = all_images_features.reshape(-1, len(new_image_features))  # Convert to 2D array



def train_model(feature_data, target_variable):
    model = LinearRegression()
    model.fit(feature_data, target_variable)

def extract_image_features(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    aspect_ratio = width / height
    total_area = height * width

    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

    # Calculate the color histograms for the Hue and Saturation channels
    hue_histogram = cv2.calcHist([hsv_image], [0], None, [8], [0, 180])
    saturation_histogram = cv2.calcHist([hsv_image], [1], None, [8], [0, 256])
  
    # Normalize the histograms
    hue_histogram = cv2.normalize(hue_histogram, hue_histogram).flatten()
    saturation_histogram = cv2.normalize(saturation_histogram, saturation_histogram).flatten()

    # Combine all the extracted features into a single feature vector
    features = np.concatenate([hue_histogram, saturation_histogram, [aspect_ratio, total_area]])

    return features 
   
    