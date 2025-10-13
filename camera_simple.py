"""
Playing Card Recognition - Real-time Camera (Python Script Version)
รันได้เลยโดยไม่ต้องใช้ Jupyter Notebook
"""

import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import os
import time

# CNN Model Definition
class CardCNN(nn.Module):
    """CNN Model for Card Classification"""
    
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

# Load Model
print("🎴 Playing Card Recognition - Real-time Camera")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")

# Load class mapping
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, 'models', 'class_mapping_cnn.json'), 'r') as f:
    class_mapping = json.load(f)
    # Use class_to_idx from the JSON file
    class_to_idx = class_mapping['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}

# Load model
num_classes = class_mapping['num_classes']
model = CardCNN(num_classes=num_classes)
model.load_state_dict(torch.load(os.path.join(script_dir, 'models', 'card_classifier_cnn.pth'), 
                                 map_location=device))
model.to(device)
model.eval()

print(f"✅ Model loaded successfully!")
print(f"📊 Number of classes: {num_classes}")
print(f"🎯 Model accuracy: 81.89%")
print()

# Transform for preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

def detect_card_region(frame):
    """Detect card region in the frame - Stricter version to avoid false positives"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        frame_area = frame.shape[0] * frame.shape[1]
        
        # Look for rectangular shapes (playing cards are rectangular)
        for contour in contours[:5]:  # Check top 5 largest contours
            area = cv2.contourArea(contour)
            
            # Card should be between 5% and 60% of frame area
            area_ratio = area / frame_area
            
            if 0.05 < area_ratio < 0.6:
                # Approximate contour to polygon
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (cards are typically ~0.7 or ~1.4 when rotated)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Stricter aspect ratio for cards (0.55 to 1.5)
                if 0.55 < aspect_ratio < 1.5:
                    # Check if contour has 4 corners (rectangle-like)
                    if len(approx) >= 4 and len(approx) <= 6:
                        # Check if not too close to edges (likely not background)
                        margin = 30
                        if (x > margin and y > margin and 
                            x + w < frame.shape[1] - margin and 
                            y + h < frame.shape[0] - margin):
                            
                            # Add some padding
                            padding = 5
                            x = max(0, x - padding)
                            y = max(0, y - padding)
                            w = min(frame.shape[1] - x, w + 2 * padding)
                            h = min(frame.shape[0] - y, h + 2 * padding)
                            
                            return x, y, w, h
    
    return None

def predict_card(image_rgb):
    """Predict card from image"""
    pil_image = Image.fromarray(image_rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = idx_to_class[predicted.item()]
        confidence_score = confidence.item() * 100
        
        return predicted_class, confidence_score

# Main Camera Loop
print("📷 Starting camera...")
print("Controls:")
print("  - Press 'q' to quit")
print("  - Press 's' to save current frame")
print("  - Press 'f' to toggle detection mode")
print("=" * 60)
print()

# Try multiple camera indices
cap = None
for camera_index in [0, 1, 2]:
    print(f"🔍 Trying camera index {camera_index}...")
    temp_cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Use DirectShow backend
    
    if temp_cap.isOpened():
        # Test read
        ret, test_frame = temp_cap.read()
        if ret:
            cap = temp_cap
            print(f"✅ Camera {camera_index} opened successfully!")
            break
        else:
            temp_cap.release()
    
if cap is None or not cap.isOpened():
    print("\n" + "=" * 60)
    print("❌ Error: Could not open any camera!")
    print("\n💡 Troubleshooting:")
    print("  1. Check if camera is connected")
    print("  2. Close other apps using camera (Zoom, Teams, Skype)")
    print("  3. Check camera permissions in Windows Settings")
    print("  4. Try restarting your computer")
    print("=" * 60)
    input("\nPress Enter to exit...")
    exit()

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print(f"📹 Camera resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"🎬 FPS: {int(cap.get(cv2.CAP_PROP_FPS))}")
print()

saved_count = 0
prev_time = time.time()
predict_whole_frame = True  # Start with Full Frame mode (safer)

frame_error_count = 0
max_frame_errors = 5

while True:
    ret, frame = cap.read()
    if not ret:
        frame_error_count += 1
        print(f"⚠️  Warning: Could not read frame! (Error {frame_error_count}/{max_frame_errors})")
        
        if frame_error_count >= max_frame_errors:
            print("\n❌ Too many frame read errors. Camera may have disconnected.")
            break
        
        time.sleep(0.1)
        continue
    
    # Reset error count on successful read
    frame_error_count = 0
    
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    predicted_class = None
    confidence = 0
    
    if predict_whole_frame:
        # Full Frame Mode - Use vertical center region (portrait orientation for cards)
        try:
            h, w = frame.shape[:2]
            
            # Create vertical rectangle (portrait) suitable for cards
            # Card aspect ratio is about 2.5:3.5 (width:height) or ~0.71
            center_x = w // 2
            center_y = h // 2
            
            # Box dimensions - vertical rectangle
            box_height = int(h * 0.7)  # 70% of frame height
            box_width = int(box_height * 0.65)  # Maintain card aspect ratio
            
            x1 = center_x - box_width // 2
            y1 = center_y - box_height // 2
            x2 = center_x + box_width // 2
            y2 = center_y + box_height // 2
            
            # Ensure box is within frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            center_region = frame[y1:y2, x1:x2]
            center_rgb = cv2.cvtColor(center_region, cv2.COLOR_BGR2RGB)
            
            predicted_class, confidence = predict_card(center_rgb)
            
            # Draw vertical card-shaped box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 3)
            
            # Draw corner markers for better visibility
            corner_len = 20
            # Top-left
            cv2.line(frame, (x1, y1), (x1+corner_len, y1), (0, 255, 255), 3)
            cv2.line(frame, (x1, y1), (x1, y1+corner_len), (0, 255, 255), 3)
            # Top-right
            cv2.line(frame, (x2, y1), (x2-corner_len, y1), (0, 255, 255), 3)
            cv2.line(frame, (x2, y1), (x2, y1+corner_len), (0, 255, 255), 3)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1+corner_len, y2), (0, 255, 255), 3)
            cv2.line(frame, (x1, y2), (x1, y2-corner_len), (0, 255, 255), 3)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2-corner_len, y2), (0, 255, 255), 3)
            cv2.line(frame, (x2, y2), (x2, y2-corner_len), (0, 255, 255), 3)
            
            # Draw text
            cv2.putText(frame, "Place card here", (x1+10, y1+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "(Vertical)", (x1+10, y1+60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except Exception as e:
            pass
    else:
        # Auto Detect Mode - Try to find card contours
        card_region = detect_card_region(frame)
        
        if card_region:
            x, y, w, h = card_region
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Extract and predict
            card_img = frame[y:y+h, x:x+w]
            card_rgb = cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB)
            
            try:
                predicted_class, confidence = predict_card(card_rgb)
                cv2.putText(frame, "Card detected!", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                pass
        else:
            # No card detected
            cv2.putText(frame, "No card detected - Press 'f' to switch mode", 
                       (10, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Display prediction if available
    if predicted_class:
        # Choose color based on confidence
        color = (0, 255, 0) if confidence > 70 else (0, 165, 255)  # Green or Orange
        
        # Draw prediction text at top
        text = f"{predicted_class.upper()}"
        conf_text = f"Confidence: {confidence:.1f}%"
        
        # Background for text
        cv2.rectangle(frame, (10, 50), (500, 120), (0, 0, 0), -1)
        
        cv2.putText(frame, text, (15, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, conf_text, (15, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Draw mode indicator
    mode_text = "Mode: Full Frame" if predict_whole_frame else "Mode: Auto Detect"
    cv2.putText(frame, mode_text, (frame.shape[1]-250, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw instructions
    cv2.putText(frame, "q=quit | s=save | f=toggle mode", (10, frame.shape[0]-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show frame
    cv2.imshow('Playing Card Recognition', frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\n👋 Exiting...")
        break
    elif key == ord('s'):
        # Create captured_cards folder if it doesn't exist
        capture_dir = os.path.join(script_dir, 'captured_cards')
        os.makedirs(capture_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"captured_{timestamp}.jpg"
        filepath = os.path.join(capture_dir, filename)
        cv2.imwrite(filepath, frame)
        saved_count += 1
        print(f"✅ Saved: {filepath} (Total: {saved_count})")
    elif key == ord('f'):
        predict_whole_frame = not predict_whole_frame
        mode = "Full Frame" if predict_whole_frame else "Auto Detect"
        print(f"🔄 Switched to {mode} mode")

# Cleanup
print("\n🧹 Cleaning up...")
if cap:
    cap.release()
cv2.destroyAllWindows()

# Give time for windows to close
time.sleep(0.5)

print("\n" + "=" * 60)
print(f"✅ Program ended successfully")
print(f"📸 Total images saved: {saved_count}")
if saved_count > 0:
    capture_dir = os.path.join(script_dir, 'captured_cards')
    print(f"📁 Images saved in: {capture_dir}")
print("=" * 60)
