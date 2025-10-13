"""
Compare Two Models
เปรียบเทียบ accuracy ระหว่างโมเดลเก่าและโมเดลใหม่
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os
from pathlib import Path

# Import model architecture
from train_cnn_model import CardCNN

def load_model(model_path, device='cpu'):
    """Load a model from file"""
    with open('models/class_mapping_cnn.json', 'r') as f:
        class_mapping = json.load(f)
        num_classes = class_mapping['num_classes']
    
    model = CardCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def test_model(model, test_path, device='cpu'):
    """Test a model on test dataset"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    correct = 0
    total = 0
    
    with open('models/class_mapping_cnn.json', 'r') as f:
        class_mapping = json.load(f)
        class_to_idx = class_mapping['class_to_idx']
    
    for class_name in os.listdir(test_path):
        class_path = os.path.join(test_path, class_name)
        if not os.path.isdir(class_path):
            continue
        
        for img_file in os.listdir(class_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(class_path, img_file)
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
            
            true_label = class_to_idx[class_name]
            if predicted.item() == true_label:
                correct += 1
            total += 1
    
    accuracy = 100 * correct / total
    return accuracy, correct, total

def main():
    print("📊 Model Comparison Tool")
    print("="*70)
    print()
    
    device = torch.device('cpu')
    
    # Get test dataset path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check for original test dataset
    ref_file = os.path.join(script_dir, 'data', 'dataset_path.txt')
    if os.path.exists(ref_file):
        with open(ref_file, 'r') as f:
            dataset_path = f.read().strip()
        test_path = os.path.join(dataset_path, 'test')
    else:
        # Try merged dataset
        test_path = os.path.join(script_dir, 'data', 'merged_dataset', 'valid')
    
    if not os.path.exists(test_path):
        print(f"❌ Test dataset not found at: {test_path}")
        return
    
    print(f"📁 Test dataset: {test_path}")
    print()
    
    # Current model
    current_model_path = os.path.join(script_dir, 'models', 'card_classifier_cnn.pth')
    
    # Backup models
    backup_dir = os.path.join(script_dir, 'models', 'backup')
    
    print("🔍 Available models:")
    print()
    print("1. Current model: models/card_classifier_cnn.pth")
    
    if os.path.exists(backup_dir):
        backups = [f for f in os.listdir(backup_dir) if f.startswith('card_classifier_cnn_') and f.endswith('.pth')]
        for i, backup in enumerate(backups, start=2):
            print(f"{i}. Backup: models/backup/{backup}")
    
    print()
    choice = input("Compare which models? (e.g., '1 2' or 'all'): ").strip()
    
    if choice == 'all':
        # Compare current with all backups
        models_to_compare = [current_model_path]
        if os.path.exists(backup_dir):
            for backup in backups:
                models_to_compare.append(os.path.join(backup_dir, backup))
    else:
        # Parse choices
        indices = [int(x) for x in choice.split()]
        models_to_compare = []
        
        for idx in indices:
            if idx == 1:
                models_to_compare.append(current_model_path)
            else:
                backup_idx = idx - 2
                if backup_idx < len(backups):
                    models_to_compare.append(os.path.join(backup_dir, backups[backup_idx]))
    
    print()
    print("🧪 Testing models...")
    print("="*70)
    
    results = []
    
    for model_path in models_to_compare:
        model_name = os.path.basename(model_path)
        print(f"\n📦 Testing: {model_name}")
        
        try:
            model = load_model(model_path, device)
            accuracy, correct, total = test_model(model, test_path, device)
            
            results.append({
                'name': model_name,
                'path': model_path,
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            })
            
            print(f"   Accuracy: {accuracy:.2f}% ({correct}/{total})")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    print()
    print("="*70)
    print("📊 COMPARISON RESULTS")
    print("="*70)
    print()
    
    # Sort by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    for i, result in enumerate(results, start=1):
        marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{marker} {result['name']}")
        print(f"   Accuracy: {result['accuracy']:.2f}%")
        print(f"   Path: {result['path']}")
        print()
    
    if len(results) >= 2:
        best = results[0]
        worst = results[-1]
        diff = best['accuracy'] - worst['accuracy']
        
        print(f"📈 Best model is {diff:.2f}% better than worst")
        print()
        
        if best['path'] != current_model_path:
            print(f"💡 Recommendation: Consider restoring the better model!")
            print(f"   Run: restore_model.bat")
    
    print("="*70)

if __name__ == "__main__":
    main()
