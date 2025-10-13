"""
Train Card Classifier Model
Train a neural network to classify 53 types of playing cards
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm.auto import tqdm
import json
from datetime import datetime

from src.feature_extraction.feature_extractor import extract_features
from src.preprocessing.image_preprocessing import preprocess_image
from src.classification.card_classifier import CardClassifierNN

class CardDataset(Dataset):
    """Custom dataset for card images"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Directory with all the card class folders
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Build file list
        self.samples = []
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_path):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_path, img_name), 
                                       self.class_to_idx[cls]))
        
        print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Extract features using our feature extractor
        features = extract_features(image)
        
        if features is None:
            # Fallback: return zero features
            features = np.zeros(70, dtype=np.float32)
        
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return features, label

def train_model(num_epochs=50, batch_size=32, learning_rate=0.001):
    """Train the card classifier model"""
    
    print("🎴 Card Classifier Training")
    print("=" * 70)
    
    # Load dataset path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ref_file = os.path.join(script_dir, 'data', 'dataset_path.txt')
    if not os.path.exists(ref_file):
        print("❌ Dataset not found. Please run 'download_dataset.py' first.")
        return
    
    with open(ref_file, 'r') as f:
        dataset_path = f.read().strip()
    
    print(f"📁 Dataset: {dataset_path}\n")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = CardDataset(os.path.join(dataset_path, 'train'))
    valid_dataset = CardDataset(os.path.join(dataset_path, 'valid'))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=0)
    
    # Model setup
    num_classes = len(train_dataset.classes)
    input_size = 70  # Feature vector size
    
    print(f"\n🧠 Model Configuration:")
    print(f"   Input size: {input_size}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Epochs: {num_epochs}\n")
    
    model = CardClassifierNN(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_valid_acc = 0.0
    train_history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}
    
    print("🚀 Starting training...")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for features, labels in train_pbar:
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            valid_pbar = tqdm(valid_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Valid]')
            for features, labels in valid_pbar:
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()
                
                valid_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * valid_correct / valid_total:.2f}%'
                })
        
        valid_loss = valid_loss / len(valid_loader)
        valid_acc = 100 * valid_correct / valid_total
        
        # Save history
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        train_history['valid_loss'].append(valid_loss)
        train_history['valid_acc'].append(valid_acc)
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.2f}%')
        
        # Save best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/card_classifier.pth')
            print(f'  ✅ New best model saved! (Accuracy: {best_valid_acc:.2f}%)')
        
        print("-" * 70)
    
    # Save training history
    history_file = f'models/training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(history_file, 'w') as f:
        json.dump(train_history, f, indent=2)
    
    # Save class mapping
    class_mapping = {
        'classes': train_dataset.classes,
        'class_to_idx': train_dataset.class_to_idx,
        'num_classes': num_classes
    }
    with open('models/class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    print("\n" + "=" * 70)
    print("🎉 Training Complete!")
    print("=" * 70)
    print(f"✅ Best validation accuracy: {best_valid_acc:.2f}%")
    print(f"📁 Model saved to: models/card_classifier.pth")
    print(f"📊 Training history saved to: {history_file}")
    print(f"📋 Class mapping saved to: models/class_mapping.json")
    print("\n📝 Next step: Test the model with 'python app/run_camera.py'")

if __name__ == "__main__":
    train_model(num_epochs=50, batch_size=32, learning_rate=0.001)
