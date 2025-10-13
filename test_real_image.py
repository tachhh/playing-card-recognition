"""
Test model with REAL training images
ทดสอบว่า model ทำนายรูปจาก dataset ได้ไหม
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os
import random

class CardCNN(nn.Module):
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

print("=" * 80)
print("🧪 TEST WITH REAL TRAINING IMAGES")
print("=" * 80)

# Load model
device = torch.device('cpu')

with open('models/class_mapping_cnn.json', 'r') as f:
    class_mapping = json.load(f)

model = CardCNN(num_classes=class_mapping['num_classes'])
model.load_state_dict(torch.load('models/card_classifier_cnn.pth', map_location=device))
model.to(device)
model.eval()

class_to_idx = class_mapping['class_to_idx']
idx_to_class = {v: k for k, v in class_to_idx.items()}

print(f"✅ Model loaded")

# Load dataset path
with open('data/dataset_path.txt', 'r') as f:
    dataset_path = f.read().strip()

train_path = os.path.join(dataset_path, 'train')
print(f"✅ Dataset: {train_path}")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Test with 10 random images
print("\n" + "=" * 80)
print("🎯 Testing with 10 REAL training images:")
print("=" * 80)

classes = sorted(os.listdir(train_path))
test_classes = random.sample(classes, min(10, len(classes)))

correct = 0
total = 0

for class_name in test_classes:
    class_path = os.path.join(train_path, class_name)
    images = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
    
    if not images:
        continue
    
    # Pick random image
    img_name = random.choice(images)
    img_path = os.path.join(class_path, img_name)
    
    # Load and predict
    pil_img = Image.open(img_path).convert('RGB')
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = idx_to_class[predicted.item()]
        confidence_score = confidence.item() * 100
        
        # Top 3
        top3_conf, top3_idx = torch.topk(probabilities, 3, dim=1)
        
        is_correct = predicted_class == class_name
        if is_correct:
            correct += 1
        total += 1
        
        status = "✅" if is_correct else "❌"
        print(f"\n{status} Image: {img_name[:30]}")
        print(f"   True:      {class_name}")
        print(f"   Predicted: {predicted_class}")
        print(f"   Confidence: {confidence_score:.2f}%")
        print(f"   Top 3:")
        for i in range(3):
            pred_idx = top3_idx[0][i].item()
            pred_class = idx_to_class[pred_idx]
            pred_conf = top3_conf[0][i].item() * 100
            marker = "←" if pred_class == class_name else ""
            print(f"      {i+1}. {pred_class}: {pred_conf:.2f}% {marker}")

print("\n" + "=" * 80)
print(f"📊 RESULTS: {correct}/{total} correct ({100*correct/total:.1f}% accuracy)")
print("=" * 80)

if correct == total:
    print("\n✅✅✅ PERFECT! Model works on training images!")
    print("If camera still gives low confidence, problem is:")
    print("  • Domain gap (camera vs training images)")
    print("  • Camera quality/lighting")
elif correct >= total * 0.6:
    print(f"\n⚠️  Model is learning but not great ({100*correct/total:.0f}%)")
    print("Need more training epochs or better hyperparameters")
else:
    print(f"\n❌ Model is NOT learning properly ({100*correct/total:.0f}%)")
    print("There's a fundamental problem with training")
