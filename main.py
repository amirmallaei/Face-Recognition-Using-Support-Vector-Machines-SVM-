import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the face detector from OpenCV
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

def prepare_data(image_paths, labels, size=(100, 100)):
    features_list = []
    labels_list = []
    for img_path, label in zip(image_paths, labels):
        image = cv2.imread(img_path)
        features = extract_face_features(image, size)
        for feature in features:
            features_list.append(feature)
            labels_list.append(label)
    return np.array(features_list), np.array(labels_list)

def main():
    # Example usage
    image_paths = [
        'path_to_image1.jpg',
        'path_to_image2.jpg',
        # Add more paths
    ]
    labels = [
        'person1',
        'person2',
        # Corresponding labels
    ]

    # Prepare the dataset
    X, y = prepare_data(image_paths, labels)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the SVM classifier
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)
    
    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()