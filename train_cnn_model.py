"""
Train Card Classifier with CNN
Using Convolutional Neural Network for better accuracy
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from tqdm.auto import tqdm
import json
from datetime import datetime
from PIL import Image

class CardCNN(nn.Module):
    """Convolutional Neural Network for card classification"""
    
    def __init__(self, num_classes=53):
        super(CardCNN, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block: 3 -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 224 -> 112
            
            # Second conv block: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112 -> 56
            
            # Third conv block: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56 -> 28
            
            # Fourth conv block: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28 -> 14
        )
        
        # Fully connected layers
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
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

class CardImageDataset(Dataset):
    """Dataset for card images using PIL and transforms"""
    
    def __init__(self, root_dir, transform=None):
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
        
        # Load image with PIL
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_cnn_model(num_epochs=30, batch_size=32, learning_rate=0.001, device='auto'):
    """Train the CNN card classifier model"""
    
    print("🎴 CNN Card Classifier Training")
    print("=" * 70)
    
    # Set device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load dataset path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use ORIGINAL dataset (better quality!)
    ref_file = os.path.join(script_dir, 'data', 'dataset_path.txt')
    if not os.path.exists(ref_file):
        print("❌ Dataset not found. Please run 'download_dataset.py' first.")
        return
    with open(ref_file, 'r') as f:
        dataset_path = f.read().strip()
    
    # Merged dataset (DO NOT USE - has issues with accuracy)
    # dataset_path = os.path.join(script_dir, 'data', 'merged_dataset')
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("💡 Run: python download_dataset.py")
        return
    
    print(f"📁 Dataset: {dataset_path}\n")
    
    # Data transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = CardImageDataset(os.path.join(dataset_path, 'train'), 
                                     transform=train_transform)
    valid_dataset = CardImageDataset(os.path.join(dataset_path, 'valid'), 
                                     transform=valid_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=0, pin_memory=True)
    
    # Model setup
    num_classes = len(train_dataset.classes)
    
    print(f"\n🧠 Model Configuration:")
    print(f"   Architecture: Convolutional Neural Network (CNN)")
    print(f"   Input size: 224x224x3 (RGB images)")
    print(f"   Number of classes: {num_classes}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Epochs: {num_epochs}\n")
    
    model = CardCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                      factor=0.5, patience=3)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}\n")
    
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
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
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
            for images, labels in valid_pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
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
        
        # Update learning rate scheduler
        scheduler.step(valid_acc)
        
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
            
            # Save model state dict
            torch.save(model.state_dict(), 
                      os.path.join(script_dir, 'models', 'card_classifier_cnn.pth'))
            
            # Save entire model for easier loading
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_valid_acc': best_valid_acc,
                'num_classes': num_classes,
            }, os.path.join(script_dir, 'models', 'card_classifier_cnn_full.pth'))
            
            print(f'  ✅ New best model saved! (Accuracy: {best_valid_acc:.2f}%)')
        
        print("-" * 70)
    
    # Save training history
    history_file = os.path.join(script_dir, 'models', 
                                f'training_history_cnn_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(history_file, 'w') as f:
        json.dump(train_history, f, indent=2)
    
    # Save class mapping
    class_mapping = {
        'classes': train_dataset.classes,
        'class_to_idx': train_dataset.class_to_idx,
        'num_classes': num_classes
    }
    with open(os.path.join(script_dir, 'models', 'class_mapping_cnn.json'), 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    print("\n" + "=" * 70)
    print("🎉 Training Complete!")
    print("=" * 70)
    print(f"✅ Best validation accuracy: {best_valid_acc:.2f}%")
    print(f"📁 Model saved to: models/card_classifier_cnn.pth")
    print(f"📊 Training history saved to: {history_file}")
    print(f"📋 Class mapping saved to: models/class_mapping_cnn.json")
    print("\n📝 Next step: Test the model with 'python test_cnn_model.py'")
    
    return model, train_history, best_valid_acc

if __name__ == "__main__":
    # Train with LOWER learning rate for better convergence
    # Increased epochs to 50 for better accuracy
    train_cnn_model(num_epochs=50, batch_size=32, learning_rate=0.0001)
