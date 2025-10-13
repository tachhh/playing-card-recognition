"""
Real-time Card Recognition with CNN Model
Run card detection using webcam with trained CNN
"""

import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os
import sys

class CardCNN(nn.Module):
    """CNN model for card classification"""
    
    def __init__(self, num_classes=53):
        super(CardCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class CNNCardClassifier:
    """Wrapper for CNN model inference"""
    
    def __init__(self, model_path='models/card_classifier_cnn.pth', device='auto'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        
        model_path = os.path.join(parent_dir, model_path)
        class_mapping_path = os.path.join(parent_dir, 'models', 'class_mapping_cnn.json')
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Loading CNN model on {self.device}...")
        
        # Load class mapping
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
        
        self.classes = class_mapping['classes']
        num_classes = class_mapping['num_classes']
        
        # Load model
        self.model = CardCNN(num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ Model loaded successfully!")
    
    def predict(self, image):
        """
        Predict card from OpenCV image
        
        Args:
            image: OpenCV BGR image (numpy array)
            
        Returns:
            (predicted_class, confidence)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL
        pil_image = Image.fromarray(image_rgb)
        
        # Transform
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = self.classes[predicted.item()]
        confidence_score = confidence.item()
        
        return predicted_class, confidence_score

def detect_card_region(frame):
    """
    Detect card region in frame
    Returns cropped card image or None
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Filter by area (should be reasonably large)
    frame_area = frame.shape[0] * frame.shape[1]
    if area < frame_area * 0.05:  # At least 5% of frame
        return None, None
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Extract card region
    card_img = frame[y:y+h, x:x+w]
    
    return card_img, (x, y, w, h)

def main():
    """Main function for real-time card recognition"""
    
    print("=" * 70)
    print("🎴 Real-time Card Recognition with CNN")
    print("=" * 70)
    
    # Load classifier
    try:
        classifier = CNNCardClassifier()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please train the CNN model first with: python train_cnn_model.py")
        return
    
    # Initialize webcam
    print("\n📷 Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print("✅ Webcam ready!")
    print("\n" + "=" * 70)
    print("Instructions:")
    print("  - Show a playing card to the camera")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("=" * 70 + "\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        
        # Detect card region
        card_img, bbox = detect_card_region(frame)
        
        # Draw detection box
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Predict every 5 frames to reduce lag
        if card_img is not None and frame_count % 5 == 0:
            try:
                prediction, confidence = classifier.predict(card_img)
                
                # Display prediction if confidence is high enough
                if confidence > 0.5:
                    # Display on frame
                    text = f"Card: {prediction}"
                    conf_text = f"Confidence: {confidence:.1%}"
                    
                    cv2.putText(frame, text, (10, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.putText(frame, conf_text, (10, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Color code by confidence
                    if confidence > 0.9:
                        color = (0, 255, 0)  # Green
                    elif confidence > 0.7:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 165, 255)  # Orange
                    
                    if bbox is not None:
                        x, y, w, h = bbox
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                        
            except Exception as e:
                print(f"Prediction error: {e}")
        
        # Add instructions on frame
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('CNN Card Recognition', frame)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save frame
            filename = f'captured_frame_{frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"📸 Frame saved as {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Application closed successfully")

if __name__ == "__main__":
    main()
