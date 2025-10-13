"""
Quick Model Test - ทดสอบว่า Model ทำงานได้จริงหรือไม่
Test model with random input and check if it can make predictions
"""

import torch
import torch.nn as nn
import json
import os

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
print("🧪 QUICK MODEL TEST")
print("=" * 80)

# Check if model file exists
model_path = 'models/card_classifier_cnn.pth'
if not os.path.exists(model_path):
    print(f"❌ ERROR: Model file not found at {model_path}")
    print("\n💡 Solution: Train the model first:")
    print("   python train_cnn_model.py")
    exit(1)

# Check model file size
model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
print(f"\n📁 Model file: {model_path}")
print(f"   Size: {model_size:.2f} MB")

if model_size < 50:
    print(f"   ⚠️  WARNING: Model file is very small!")
    print(f"   Expected: ~100-105 MB")
    print(f"   This may indicate an incomplete or corrupted model.")
elif model_size < 90:
    print(f"   ⚠️  WARNING: Model file seems smaller than expected")
    print(f"   Expected: ~100-105 MB")
else:
    print(f"   ✅ Model file size looks good")

# Load class mapping
print("\n📋 Loading class mapping...")
with open('models/class_mapping_cnn.json', 'r') as f:
    class_mapping = json.load(f)

num_classes = class_mapping['num_classes']
print(f"   Number of classes: {num_classes}")

# Load model
print("\n🧠 Loading model...")
device = torch.device('cpu')
model = CardCNN(num_classes=num_classes)

try:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("   ✅ Model loaded successfully")
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
    exit(1)

model.to(device)
model.eval()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"   Total parameters: {total_params:,}")

# Test with random inputs (5 different random inputs)
print("\n🎲 Testing with random inputs...")
print("   (Simulating 5 different preprocessed images)")
print("-" * 80)

confidence_scores = []

for i in range(5):
    # Create random input (simulating a preprocessed image)
    random_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        # Get model output
        logits = model(random_input)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        conf_percent = confidence.item() * 100
        confidence_scores.append(conf_percent)
        
        # Get top 3
        top3_conf, top3_idx = torch.topk(probabilities, 3, dim=1)
        
        print(f"\nTest {i+1}:")
        print(f"   Max confidence: {conf_percent:.2f}%")
        print(f"   Top 3:")
        for j in range(3):
            idx = top3_idx[0][j].item()
            conf = top3_conf[0][j].item() * 100
            print(f"      {j+1}. Class {idx}: {conf:.2f}%")

# Analyze results
avg_confidence = sum(confidence_scores) / len(confidence_scores)
max_confidence = max(confidence_scores)
min_confidence = min(confidence_scores)

print("\n" + "=" * 80)
print("📊 RESULTS SUMMARY")
print("=" * 80)
print(f"Average max confidence: {avg_confidence:.2f}%")
print(f"Highest confidence: {max_confidence:.2f}%")
print(f"Lowest confidence: {min_confidence:.2f}%")

print("\n💡 INTERPRETATION:")
if avg_confidence > 50:
    print("   ✅ EXCELLENT: Model is confident and likely well-trained")
    print("   → Model should work well with real images")
    print("   → If camera still gives low confidence, problem is likely:")
    print("      • Domain gap (camera images very different from training)")
    print("      • Camera quality/lighting issues")
elif avg_confidence > 20:
    print("   ⚠️  MODERATE: Model shows some confidence")
    print("   → Model may be partially trained")
    print("   → Consider training for more epochs")
    print("   → Test with actual test set to verify accuracy")
elif avg_confidence > 10:
    print("   ⚠️  LOW: Model has weak confidence")
    print("   → Model is undertrained")
    print("   → Strongly recommend retraining with more epochs")
else:
    print("   ❌ CRITICAL: Model is essentially random guessing!")
    print("   → Model is NOT trained or loaded incorrectly")
    print("   → MUST retrain the model:")
    print("      python train_cnn_model.py")

# Additional check: Test logits statistics
print("\n🔬 TECHNICAL ANALYSIS:")
with torch.no_grad():
    test_input = torch.randn(1, 3, 224, 224)
    logits = model(test_input)
    
    logits_std = logits.std().item()
    logits_mean = logits.mean().item()
    logits_range = (logits.max() - logits.min()).item()
    
    print(f"   Logits mean: {logits_mean:.3f}")
    print(f"   Logits std: {logits_std:.3f}")
    print(f"   Logits range: {logits_range:.3f}")
    
    if logits_std < 0.5:
        print("\n   ⚠️  WARNING: Logits have very low standard deviation!")
        print("   This indicates the model is not properly trained.")
        print("   A well-trained model should have logits std > 1.0")
    elif logits_std < 1.0:
        print("\n   ⚠️  CAUTION: Logits standard deviation is lower than ideal")
        print("   Model may benefit from more training.")
    else:
        print("\n   ✅ Logits statistics look healthy")

print("\n" + "=" * 80)
print("🎯 RECOMMENDATION:")
print("=" * 80)

if avg_confidence < 20:
    print("\n🚨 CRITICAL ACTION REQUIRED:")
    print("   1. The model is NOT properly trained")
    print("   2. Run: python train_cnn_model.py")
    print("   3. Wait for 30 epochs to complete (~30-60 minutes)")
    print("   4. You should see accuracy improve to >70%")
    print("   5. Then test camera again with: python camera_simple.py")
elif avg_confidence < 50:
    print("\n⚠️  RECOMMENDED ACTION:")
    print("   1. Model may be undertrained")
    print("   2. Run test_cnn_model.py to check actual accuracy on test set")
    print("   3. If test accuracy < 70%, retrain with: python train_cnn_model.py")
    print("   4. Consider training for more epochs (40-50 instead of 30)")
else:
    print("\n✅ MODEL IS WORKING:")
    print("   1. Model appears to be properly trained")
    print("   2. If camera gives low confidence, the problem is likely:")
    print("      • Lighting conditions in camera different from training")
    print("      • Camera quality/focus issues")
    print("      • Card appearance different (worn cards, different deck)")
    print("   3. Solutions:")
    print("      • Improve lighting (bright, even illumination)")
    print("      • Hold card steady and flat")
    print("      • Use cards similar to training data")
    print("      • Consider retraining with augmented data")

print("\n" + "=" * 80)
