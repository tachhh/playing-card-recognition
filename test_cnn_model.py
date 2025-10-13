"""
Test CNN Model on Sample Images
Visualize predictions and show accuracy
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import json
import numpy as np

class CardCNN(nn.Module):
    """Same CNN architecture as training"""
    
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

def load_cnn_model(model_path='models/card_classifier_cnn.pth', device='auto'):
    """Load the trained CNN model"""
    
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load class mapping
    with open('models/class_mapping_cnn.json', 'r') as f:
        class_mapping = json.load(f)
    
    num_classes = class_mapping['num_classes']
    classes = class_mapping['classes']
    
    # Create and load model
    model = CardCNN(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, classes, device

def predict_image(image_path, model, classes, device):
    """Predict card class for a single image"""
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = classes[predicted.item()]
    confidence_score = confidence.item()
    
    return predicted_class, confidence_score, probabilities[0]

def test_on_dataset():
    """Test model on test dataset"""
    
    print("🧪 Testing CNN Model")
    print("=" * 70)
    
    # Load model
    print("Loading model...")
    model, classes, device = load_cnn_model()
    print(f"✅ Model loaded on {device}")
    print(f"📊 Number of classes: {len(classes)}\n")
    
    # Load dataset path
    ref_file = os.path.join('data', 'dataset_path.txt')
    with open(ref_file, 'r') as f:
        dataset_path = f.read().strip()
    
    test_path = os.path.join(dataset_path, 'test')
    
    # Test on all test images
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    
    print("Testing on test dataset...")
    
    for cls in classes:
        cls_path = os.path.join(test_path, cls)
        if not os.path.exists(cls_path):
            continue
        
        images = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in images:
            img_path = os.path.join(cls_path, img_name)
            
            predicted_class, confidence, _ = predict_image(img_path, model, classes, device)
            
            total += 1
            if cls not in class_total:
                class_total[cls] = 0
                class_correct[cls] = 0
            
            class_total[cls] += 1
            
            if predicted_class == cls:
                correct += 1
                class_correct[cls] += 1
    
    # Print results
    accuracy = 100 * correct / total if total > 0 else 0
    
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS")
    print("=" * 70)
    print(f"Total images tested: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    
    # Show per-class accuracy for first 10 classes
    print("\nPer-class accuracy (first 10 classes):")
    for cls in sorted(classes)[:10]:
        if cls in class_total and class_total[cls] > 0:
            cls_acc = 100 * class_correct[cls] / class_total[cls]
            print(f"  {cls:30s}: {cls_acc:.1f}% ({class_correct[cls]}/{class_total[cls]})")
    
    print("\n" + "=" * 70)
    
    return accuracy

def visualize_predictions(num_samples=12):
    """Visualize predictions on sample images"""
    
    print("\n📸 Visualizing predictions...")
    
    # Load model
    model, classes, device = load_cnn_model()
    
    # Load dataset
    ref_file = os.path.join('data', 'dataset_path.txt')
    with open(ref_file, 'r') as f:
        dataset_path = f.read().strip()
    
    test_path = os.path.join(dataset_path, 'test')
    
    # Collect sample images
    samples = []
    for cls in sorted(classes)[:num_samples]:
        cls_path = os.path.join(test_path, cls)
        if os.path.exists(cls_path):
            images = [f for f in os.listdir(cls_path) if f.endswith('.jpg')]
            if images:
                samples.append((os.path.join(cls_path, images[0]), cls))
    
    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    fig.suptitle('CNN Model Predictions', fontsize=16)
    
    for idx, (img_path, true_class) in enumerate(samples):
        if idx >= 12:
            break
        
        row = idx // 4
        col = idx % 4
        
        # Predict
        predicted_class, confidence, _ = predict_image(img_path, model, classes, device)
        
        # Load image for display
        image = Image.open(img_path)
        
        # Display
        axes[row, col].imshow(image)
        
        # Color code: green if correct, red if wrong
        color = 'green' if predicted_class == true_class else 'red'
        
        title = f'True: {true_class}\nPred: {predicted_class}\nConf: {confidence:.2%}'
        axes[row, col].set_title(title, color=color, fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('models/cnn_predictions_visualization.png', dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: models/cnn_predictions_visualization.png")
    
    # Show if possible
    try:
        plt.show()
    except:
        print("(Display not available in this environment)")

if __name__ == "__main__":
    # Test on dataset
    accuracy = test_on_dataset()
    
    # Visualize predictions
    if accuracy > 0:
        visualize_predictions()
