"""
Explore the Cards Dataset
Display statistics and sample images from the dataset
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def load_dataset_path():
    """Load the dataset path from the reference file"""
    ref_file = os.path.join(os.path.dirname(__file__), 'data', 'dataset_path.txt')
    if os.path.exists(ref_file):
        with open(ref_file, 'r') as f:
            return f.read().strip()
    return None

def explore_dataset():
    """Explore and display dataset statistics"""
    
    # Load dataset path
    dataset_path = load_dataset_path()
    if not dataset_path or not os.path.exists(dataset_path):
        print("❌ Dataset not found. Please run 'download_dataset.py' first.")
        return
    
    print("🎴 Cards Image Dataset Explorer")
    print("=" * 70)
    print(f"📁 Dataset location: {dataset_path}\n")
    
    # Explore train, test, valid directories
    for split in ['train', 'test', 'valid']:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            continue
            
        print(f"\n📊 {split.upper()} SET:")
        print("-" * 70)
        
        # Get all classes
        classes = sorted([d for d in os.listdir(split_path) 
                         if os.path.isdir(os.path.join(split_path, d))])
        
        print(f"   Number of classes: {len(classes)}")
        
        # Count images per class
        class_counts = {}
        total_images = 0
        
        for cls in classes:
            cls_path = os.path.join(split_path, cls)
            images = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            class_counts[cls] = len(images)
            total_images += len(images)
        
        print(f"   Total images: {total_images}")
        print(f"   Avg images per class: {total_images / len(classes):.1f}")
        print(f"   Min images in class: {min(class_counts.values())} ({min(class_counts, key=class_counts.get)})")
        print(f"   Max images in class: {max(class_counts.values())} ({max(class_counts, key=class_counts.get)})")
        
        # Show first 10 classes
        print(f"\n   First 10 classes:")
        for cls in classes[:10]:
            print(f"      {cls}: {class_counts[cls]} images")
        
        if len(classes) > 10:
            print(f"      ... and {len(classes) - 10} more classes")
    
    # Check image properties
    print("\n" + "=" * 70)
    print("🖼️  Sample Image Properties:")
    print("-" * 70)
    
    train_path = os.path.join(dataset_path, 'train')
    if os.path.exists(train_path):
        first_class = sorted(os.listdir(train_path))[0]
        first_class_path = os.path.join(train_path, first_class)
        first_image = [f for f in os.listdir(first_class_path) if f.endswith('.jpg')][0]
        first_image_path = os.path.join(first_class_path, first_image)
        
        img = cv2.imread(first_image_path)
        if img is not None:
            print(f"   Sample: {first_class}/{first_image}")
            print(f"   Shape: {img.shape} (Height, Width, Channels)")
            print(f"   Data type: {img.dtype}")
            print(f"   Size: {os.path.getsize(first_image_path) / 1024:.1f} KB")
    
    print("\n" + "=" * 70)
    print("✅ Dataset exploration complete!")
    print("\n📝 Next step: Run 'python train_model.py' to train the model")

def show_sample_images(num_samples=12):
    """Display sample images from each card type"""
    dataset_path = load_dataset_path()
    if not dataset_path:
        print("❌ Dataset not found.")
        return
    
    train_path = os.path.join(dataset_path, 'train')
    if not os.path.exists(train_path):
        return
    
    classes = sorted(os.listdir(train_path))[:num_samples]
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle('Sample Cards from Dataset', fontsize=16)
    
    for idx, cls in enumerate(classes):
        row = idx // 4
        col = idx % 4
        
        cls_path = os.path.join(train_path, cls)
        images = [f for f in os.listdir(cls_path) if f.endswith('.jpg')]
        
        if images:
            img_path = os.path.join(cls_path, images[0])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[row, col].imshow(img)
            axes[row, col].set_title(cls)
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    explore_dataset()
    
    # Uncomment to show sample images (requires matplotlib display)
    # print("\n📷 Showing sample images...")
    # show_sample_images()
