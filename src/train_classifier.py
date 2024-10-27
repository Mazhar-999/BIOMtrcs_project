import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Import Support Vector Classifier
from sklearn.metrics import accuracy_score

def load_features_labels(authorized_dir, unauthorized_dir):
    # Load features and labels
    features = joblib.load(os.path.join(authorized_dir, 'features', 'features.pkl'))
    labels = joblib.load(os.path.join(unauthorized_dir, 'features', 'labels.pkl'))
    return features, labels

def train_model(features, labels):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Initialize the classifier (Support Vector Machine)
    clf = SVC(kernel='linear')  # You can change the kernel as needed
    clf.fit(X_train, y_train)

    # Make predictions
    predictions = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    return clf

def main():
    # Set paths for authorized and unauthorized data
    authorized_dir = "data/authorized/"
    unauthorized_dir = "data/unauthorized/"

    # Load features and labels
    features, labels = load_features_labels(authorized_dir, unauthorized_dir)

    # Train the model
    model = train_model(features, labels)

    # Save the trained model
    joblib.dump(model, 'voice_authentication_model.pkl')

if __name__ == "__main__":
    main()
