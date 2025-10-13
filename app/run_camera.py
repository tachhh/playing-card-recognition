import cv2
import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.image_preprocessing import preprocess_image
from src.feature_extraction.feature_extractor import extract_features
from src.classification.card_classifier import CardClassifier

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    # Load the classifier model
    classifier = CardClassifier()
    classifier.load_model('models/card_classifier.pth')
    
    print("Press 'q' to quit")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Preprocess the image
        processed_img = preprocess_image(frame)
        
        # Extract features using linear algebra techniques
        features = extract_features(processed_img)
        
        # Classify the card
        if features is not None:
            prediction, confidence = classifier.predict(features)
            
            # Display the result on the frame
            if confidence > 0.7:  # Only show high confidence predictions
                cv2.putText(frame, f"Card: {prediction} ({confidence:.2f})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Card Recognition', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
