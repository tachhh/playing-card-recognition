"""
Continue Training CNN Model
Resume training from the best saved checkpoint to improve accuracy
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
import json
from datetime import datetime

class CardCNN(nn.Module):
    """Same CNN architecture as before"""
    
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

class CardImageDataset(Dataset):
    """Dataset for card images"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
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
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def continue_training(additional_epochs=20, batch_size=32, learning_rate=0.0005, device='auto'):
    """Continue training from the last checkpoint"""
    
    print("🎴 Continue Training CNN Model")
    print("=" * 70)
    
    # Set device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load dataset path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ref_file = os.path.join(script_dir, 'data', 'dataset_path.txt')
    
    with open(ref_file, 'r') as f:
        dataset_path = f.read().strip()
    
    print(f"📁 Dataset: {dataset_path}")
    
    # Check if checkpoint exists
    checkpoint_path = os.path.join(script_dir, 'models', 'card_classifier_cnn_full.pth')
    if not os.path.exists(checkpoint_path):
        print("❌ No checkpoint found! Please train the model first.")
        return
    
    # Load checkpoint
    print("\n📦 Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    start_epoch = checkpoint['epoch'] + 1
    best_valid_acc = checkpoint['best_valid_acc']
    num_classes = checkpoint['num_classes']
    
    print(f"✅ Checkpoint loaded!")
    print(f"   Previous best accuracy: {best_valid_acc:.2f}%")
    print(f"   Resuming from epoch: {start_epoch}")
    
    # Enhanced data augmentation for continued training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  # Increased rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # More variation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Added translation
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
    print("\nLoading datasets...")
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
    print(f"\n🧠 Training Configuration:")
    print(f"   Starting epoch: {start_epoch}")
    print(f"   Additional epochs: {additional_epochs}")
    print(f"   Total epochs: {start_epoch + additional_epochs - 1}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate} (reduced from 0.001)")
    print(f"   Previous best: {best_valid_acc:.2f}%\n")
    
    # Load model
    model = CardCNN(num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Update learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                      factor=0.5, patience=3)
    
    # Training history
    train_history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}
    
    print("🚀 Resuming training...")
    print("=" * 70)
    
    end_epoch = start_epoch + additional_epochs
    
    for epoch in range(start_epoch, end_epoch):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{end_epoch-1} [Train]')
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
            valid_pbar = tqdm(valid_loader, desc=f'Epoch {epoch}/{end_epoch-1} [Valid]')
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
        print(f'\nEpoch {epoch}/{end_epoch-1} Summary:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.2f}%')
        
        # Save best model
        if valid_acc > best_valid_acc:
            improvement = valid_acc - best_valid_acc
            best_valid_acc = valid_acc
            
            # Save model state dict
            torch.save(model.state_dict(), 
                      os.path.join(script_dir, 'models', 'card_classifier_cnn.pth'))
            
            # Save full checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_valid_acc': best_valid_acc,
                'num_classes': num_classes,
            }, os.path.join(script_dir, 'models', 'card_classifier_cnn_full.pth'))
            
            print(f'  ✅ New best model saved! (Accuracy: {best_valid_acc:.2f}%, +{improvement:.2f}%)')
        else:
            print(f'  📊 Best so far: {best_valid_acc:.2f}%')
        
        print("-" * 70)
    
    # Save training history
    history_file = os.path.join(script_dir, 'models', 
                                f'training_history_continued_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(history_file, 'w') as f:
        json.dump(train_history, f, indent=2)
    
    print("\n" + "=" * 70)
    print("🎉 Continued Training Complete!")
    print("=" * 70)
    print(f"✅ Best validation accuracy: {best_valid_acc:.2f}%")
    print(f"📁 Model saved to: models/card_classifier_cnn.pth")
    print(f"📊 Training history saved to: {history_file}")
    print("\n📝 Next step: Test the improved model with 'python test_cnn_model.py'")
    
    return model, train_history, best_valid_acc

if __name__ == "__main__":
    # Continue training for 20 more epochs with reduced learning rate
    # This should push accuracy to 90%+
    continue_training(additional_epochs=20, batch_size=32, learning_rate=0.0005)
