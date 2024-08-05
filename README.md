# Face Recognition with SVM

This project implements a face recognition system using Support Vector Machines (SVM). The system detects faces in images, extracts features, and classifies them using SVM.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [License](#license)

## Introduction

This repository contains a Python script for face recognition using SVM. The script uses OpenCV for face detection and `scikit-learn` for training and evaluating the SVM classifier. 

## Prerequisites

Ensure you have the following Python libraries installed:

- `numpy`
- `opencv-python`
- `scikit-learn`
- `pandas`

You can install them using pip:

```bash
pip install numpy opencv-python scikit-learn pandas

Installation
Clone this repository:

bash
Copy code
git clone https://github.com/your-username/face-recognition-svm.git
Navigate into the project directory:

bash
Copy code
cd face-recognition-svm
Install the required dependencies (if not already installed):

bash
Copy code
pip install -r requirements.txt
Usage
Prepare your dataset: Place your images in the project directory or provide the paths to them. Ensure each image has a corresponding label.

Edit the script: Update the image_paths and labels variables in face_recognition_svm.py with the paths to your images and the corresponding labels.

Run the script:

bash
Copy code
python face_recognition_svm.py
This will train the SVM model on the provided dataset and evaluate its accuracy.

Code Explanation
Face Detection: Uses OpenCV's Haar cascades to detect faces and resize them to a uniform size.

Feature Extraction: Extracts and flattens face features for SVM training.

Training and Evaluation: Splits the data into training and test sets, trains the SVM classifier, and evaluates its accuracy.

Sample Code
Here’s a snippet of the core code:

python
Copy code
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face_features(image, size=(100, 100)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    features = []
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, size).flatten()
        features.append(face)
    return features

# Further code...
