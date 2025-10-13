"""
Comprehensive Diagnostic for Low Confidence Issue
ตรวจสอบทุกขั้นตอนของการอ่านไพ่จากกล้อง
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import json
import os

# Load model architecture
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

def test_preprocessing():
    """Test preprocessing pipeline"""
    print("=" * 80)
    print("🔍 DIAGNOSTIC 1: Preprocessing Pipeline")
    print("=" * 80)
    
    # Training transforms (from train_cnn_model.py)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Inference transforms (from camera_simple.py)
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print("✅ Training transform:")
    print(train_transform)
    print("\n✅ Inference transform:")
    print(inference_transform)
    print("\n✅ Transforms MATCH - OK")
    
    # Test with a sample image
    test_data_path = 'data/Cards/train'
    if os.path.exists(test_data_path):
        # Get first available class and image
        classes = sorted(os.listdir(test_data_path))
        if classes:
            first_class = classes[0]
            class_path = os.path.join(test_data_path, first_class)
            images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png'))]
            if images:
                test_img_path = os.path.join(class_path, images[0])
                print(f"\n📷 Testing with: {test_img_path}")
                
                # Load with PIL (training method)
                pil_img = Image.open(test_img_path).convert('RGB')
                print(f"   PIL Image: size={pil_img.size}, mode={pil_img.mode}")
                
                # Transform
                tensor = inference_transform(pil_img)
                print(f"   Tensor: shape={tensor.shape}, dtype={tensor.dtype}")
                print(f"   Tensor range: min={tensor.min():.3f}, max={tensor.max():.3f}, mean={tensor.mean():.3f}")
                
                return True
    
    print("⚠️  No test images found")
    return False

def test_model_loading():
    """Test model loading"""
    print("\n" + "=" * 80)
    print("🔍 DIAGNOSTIC 2: Model Loading")
    print("=" * 80)
    
    device = torch.device('cpu')
    
    # Load class mapping
    with open('models/class_mapping_cnn.json', 'r') as f:
        class_mapping = json.load(f)
    
    num_classes = class_mapping['num_classes']
    print(f"✅ Loaded class mapping: {num_classes} classes")
    
    # Create model
    model = CardCNN(num_classes=num_classes)
    print(f"✅ Created model with {num_classes} classes")
    
    # Load weights
    try:
        state_dict = torch.load('models/card_classifier_cnn.pth', map_location=device)
        model.load_state_dict(state_dict)
        print("✅ Loaded model weights successfully")
        
        # Check model structure
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Total parameters: {total_params:,}")
        
        model.eval()
        print("✅ Model set to eval mode")
        
        return model, class_mapping
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def test_class_mapping():
    """Test class mapping correctness"""
    print("\n" + "=" * 80)
    print("🔍 DIAGNOSTIC 3: Class Mapping")
    print("=" * 80)
    
    with open('models/class_mapping_cnn.json', 'r') as f:
        class_mapping = json.load(f)
    
    class_to_idx = class_mapping['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    print(f"✅ Number of classes: {len(class_to_idx)}")
    print(f"\n📋 First 10 class mappings:")
    for i in range(min(10, len(idx_to_class))):
        print(f"   {i}: {idx_to_class[i]}")
    
    # Check for missing indices
    max_idx = max(idx_to_class.keys())
    missing = [i for i in range(max_idx + 1) if i not in idx_to_class]
    if missing:
        print(f"\n⚠️  WARNING: Missing indices: {missing}")
    else:
        print(f"\n✅ All indices 0-{max_idx} present")
    
    return class_to_idx, idx_to_class

def test_inference_pipeline(model, idx_to_class):
    """Test full inference pipeline"""
    print("\n" + "=" * 80)
    print("🔍 DIAGNOSTIC 4: Full Inference Pipeline")
    print("=" * 80)
    
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Test with training image
    test_data_path = 'data/Cards/train'
    if not os.path.exists(test_data_path):
        print("⚠️  Training data not found")
        return
    
    classes = sorted(os.listdir(test_data_path))
    
    print(f"\n🎯 Testing with 5 random training images:")
    print("-" * 80)
    
    import random
    test_classes = random.sample(classes, min(5, len(classes)))
    
    correct = 0
    total = 0
    low_confidence = []
    
    for test_class in test_classes:
        class_path = os.path.join(test_data_path, test_class)
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png'))]
        
        if not images:
            continue
        
        # Pick random image from this class
        img_name = random.choice(images)
        img_path = os.path.join(class_path, img_name)
        
        # Load and preprocess
        pil_img = Image.open(img_path).convert('RGB')
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = idx_to_class[predicted.item()]
            confidence_score = confidence.item() * 100
            
            # Get top 3 predictions
            top3_conf, top3_idx = torch.topk(probabilities, 3, dim=1)
            
            is_correct = predicted_class == test_class
            if is_correct:
                correct += 1
            total += 1
            
            status = "✅" if is_correct else "❌"
            print(f"\n{status} Image: {img_name}")
            print(f"   True label:      {test_class}")
            print(f"   Predicted:       {predicted_class}")
            print(f"   Confidence:      {confidence_score:.2f}%")
            print(f"   Top 3 predictions:")
            for i in range(3):
                pred_idx = top3_idx[0][i].item()
                pred_class = idx_to_class[pred_idx]
                pred_conf = top3_conf[0][i].item() * 100
                print(f"      {i+1}. {pred_class}: {pred_conf:.2f}%")
            
            if confidence_score < 50:
                low_confidence.append((test_class, predicted_class, confidence_score))
    
    print("\n" + "-" * 80)
    print(f"📊 Results: {correct}/{total} correct ({100*correct/total:.1f}% accuracy)")
    
    if low_confidence:
        print(f"\n⚠️  Low confidence predictions (<50%):")
        for true_class, pred_class, conf in low_confidence:
            print(f"   {true_class} → {pred_class} ({conf:.1f}%)")
    else:
        print("\n✅ All predictions have high confidence (>50%)")

def test_camera_preprocessing():
    """Test camera frame preprocessing"""
    print("\n" + "=" * 80)
    print("🔍 DIAGNOSTIC 5: Camera Frame Preprocessing")
    print("=" * 80)
    
    print("\n📷 Opening camera...")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print("✅ Camera opened")
    print("\n📸 Capturing frame...")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not capture frame")
        return
    
    print(f"✅ Frame captured: shape={frame.shape}, dtype={frame.dtype}")
    print(f"   Frame range: min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")
    
    # Extract center region (as in camera_simple.py)
    h, w = frame.shape[:2]
    box_height = int(h * 0.7)
    box_width = int(box_height * 0.65)
    center_x, center_y = w // 2, h // 2
    x1 = center_x - box_width // 2
    y1 = center_y - box_height // 2
    x2 = center_x + box_width // 2
    y2 = center_y + box_height // 2
    
    center_region = frame[y1:y2, x1:x2]
    print(f"\n✅ Extracted center region: shape={center_region.shape}")
    
    # Convert BGR to RGB
    center_rgb = cv2.cvtColor(center_region, cv2.COLOR_BGR2RGB)
    print(f"✅ Converted BGR → RGB")
    print(f"   RGB range: min={center_rgb.min()}, max={center_rgb.max()}, mean={center_rgb.mean():.1f}")
    
    # Convert to PIL and transform
    pil_img = Image.fromarray(center_rgb)
    print(f"✅ Converted to PIL: size={pil_img.size}, mode={pil_img.mode}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(pil_img)
    print(f"✅ Transformed to tensor: shape={tensor.shape}")
    print(f"   Tensor range: min={tensor.min():.3f}, max={tensor.max():.3f}, mean={tensor.mean():.3f}")
    
    # Check if frame looks like noise
    std_dev = np.std(center_region)
    mean_val = np.mean(center_region)
    print(f"\n📊 Frame statistics:")
    print(f"   Standard deviation: {std_dev:.2f}")
    print(f"   Mean value: {mean_val:.2f}")
    
    if std_dev > 60 and 100 < mean_val < 150:
        print("   ⚠️  WARNING: Frame appears to be NOISE/STATIC!")
        print("   This is likely the main problem!")
    else:
        print("   ✅ Frame looks normal")

def check_model_outputs():
    """Check raw model output statistics"""
    print("\n" + "=" * 80)
    print("🔍 DIAGNOSTIC 6: Model Output Analysis")
    print("=" * 80)
    
    device = torch.device('cpu')
    
    # Load model
    with open('models/class_mapping_cnn.json', 'r') as f:
        class_mapping = json.load(f)
    
    model = CardCNN(num_classes=class_mapping['num_classes'])
    model.load_state_dict(torch.load('models/card_classifier_cnn.pth', map_location=device))
    model.to(device)
    model.eval()
    
    # Create random input (simulating preprocessed image)
    random_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        # Raw logits
        logits = model(random_input)
        print(f"✅ Raw logits shape: {logits.shape}")
        print(f"   Logits range: min={logits.min():.3f}, max={logits.max():.3f}, mean={logits.mean():.3f}")
        print(f"   Logits std: {logits.std():.3f}")
        
        # Softmax probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)
        print(f"\n✅ Softmax probabilities:")
        print(f"   Probs range: min={probs.min():.6f}, max={probs.max():.6f}, sum={probs.sum():.6f}")
        print(f"   Max confidence: {probs.max().item() * 100:.2f}%")
        
        # Top 5 predictions
        top5_conf, top5_idx = torch.topk(probs, 5, dim=1)
        print(f"\n   Top 5 confidence scores:")
        for i in range(5):
            print(f"      {i+1}. {top5_conf[0][i].item() * 100:.2f}%")
        
        if logits.std() < 0.1:
            print("\n   ⚠️  WARNING: Logits have very low variance!")
            print("   This suggests model may not be properly trained or loaded.")
        
        if probs.max() < 0.1:
            print("\n   ⚠️  WARNING: All probabilities are very low!")
            print("   This suggests model is very uncertain (confidence distributed across many classes).")

def main():
    """Run all diagnostics"""
    print("\n" + "=" * 80)
    print("🔬 COMPREHENSIVE CAMERA INFERENCE DIAGNOSTIC")
    print("   วิเคราะห์ปัญหา confidence ต่ำ (<10%)")
    print("=" * 80)
    
    try:
        # Test 1: Preprocessing
        test_preprocessing()
        
        # Test 2: Model loading
        model, class_mapping = test_model_loading()
        
        if model and class_mapping:
            # Test 3: Class mapping
            class_to_idx, idx_to_class = test_class_mapping()
            
            # Test 4: Inference on training images
            test_inference_pipeline(model, idx_to_class)
        
        # Test 5: Camera preprocessing
        test_camera_preprocessing()
        
        # Test 6: Model output analysis
        check_model_outputs()
        
        print("\n" + "=" * 80)
        print("🏁 DIAGNOSTIC COMPLETE")
        print("=" * 80)
        print("\n💡 ANALYSIS:")
        print("   1. ถ้าทุก diagnostic ผ่าน แต่ camera ยัง confidence ต่ำ")
        print("      → ปัญหาคือ domain gap (ภาพกล้องต่างจาก training data)")
        print("   2. ถ้า camera frame เป็น noise/static")
        print("      → ต้องแก้ไขกล้อง (ดู FIX_CAMERA_NOISE.md)")
        print("   3. ถ้า model inference บน training images ได้ confidence ต่ำ")
        print("      → ปัญหาที่ model หรือ class mapping")
        print("   4. ถ้า logits มี variance ต่ำ")
        print("      → model อาจโหลดไม่ถูกต้องหรือยังไม่ได้ฝึก")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
