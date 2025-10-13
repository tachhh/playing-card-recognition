"""
Dataset Format Converter
แปลง dataset format ต่างๆ ให้เข้ากับโครงสร้างของโปรเจกต์
"""

import os
import shutil
from pathlib import Path
import json

def detect_dataset_format(dataset_path):
    """
    ตรวจสอบ format ของ dataset
    รองรับหลายรูปแบบ:
    1. Flat structure: /images/card_name_001.jpg
    2. Class folders: /class_name/image.jpg
    3. Train/Val/Test split: /train/class/image.jpg
    4. Custom structure
    """
    dataset_path = Path(dataset_path)
    
    # Check if has train/val/test folders
    has_split = (dataset_path / 'train').exists() or \
                (dataset_path / 'training').exists() or \
                (dataset_path / 'Train').exists()
    
    if has_split:
        print("✅ Detected: Train/Val/Test split structure")
        return "split"
    
    # Check for class folders
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    if len(subdirs) > 10:  # Likely class folders
        print("✅ Detected: Class folder structure")
        return "class_folders"
    
    # Check for flat structure with labeled filenames
    images = list(dataset_path.glob("*.jpg")) + \
             list(dataset_path.glob("*.png")) + \
             list(dataset_path.glob("*.jpeg"))
    
    if len(images) > 100:
        print("✅ Detected: Flat structure with images")
        return "flat"
    
    print("❌ Could not detect dataset format")
    return "unknown"

def convert_flat_to_class_folders(source_path, output_path, naming_pattern=None):
    """
    แปลงจาก flat structure เป็น class folders
    naming_pattern: regex หรือ function สำหรับแยก class name จากชื่อไฟล์
    """
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images = list(source_path.glob("*.jpg")) + \
             list(source_path.glob("*.png")) + \
             list(source_path.glob("*.jpeg"))
    
    print(f"📁 Found {len(images)} images")
    
    # Extract class names from filenames
    for img_path in images:
        filename = img_path.name
        
        # Try to extract class name (customize based on your naming)
        # Example: "ace_of_hearts_001.jpg" -> "ace of hearts"
        parts = filename.split('_')
        if len(parts) >= 3:
            class_name = '_'.join(parts[:-1])  # Remove number at end
            class_name = class_name.replace('_', ' ')
        else:
            class_name = "unknown"
        
        # Create class folder
        class_folder = output_path / class_name
        class_folder.mkdir(exist_ok=True)
        
        # Copy image
        dest = class_folder / filename
        shutil.copy2(img_path, dest)
    
    print(f"✅ Converted to class folders in: {output_path}")

def merge_datasets(dataset1_path, dataset2_path, output_path, train_split=0.8):
    """
    รวม 2 datasets เข้าด้วยกัน และแบ่ง train/val
    """
    import random
    from collections import defaultdict
    
    output_path = Path(output_path)
    train_path = output_path / 'train'
    val_path = output_path / 'valid'
    
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all images by class
    images_by_class = defaultdict(list)
    
    for dataset_path in [Path(dataset1_path), Path(dataset2_path)]:
        if not dataset_path.exists():
            continue
            
        print(f"📂 Processing: {dataset_path}")
        
        # Find all class folders
        for class_folder in dataset_path.iterdir():
            if not class_folder.is_dir():
                continue
            
            class_name = class_folder.name
            
            # Find all images in class
            for img_path in class_folder.glob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    images_by_class[class_name].append(img_path)
    
    # Split and copy files
    total_train = 0
    total_val = 0
    
    for class_name, images in images_by_class.items():
        print(f"  {class_name}: {len(images)} images")
        
        # Shuffle
        random.shuffle(images)
        
        # Split
        split_idx = int(len(images) * train_split)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Create class folders
        (train_path / class_name).mkdir(exist_ok=True)
        (val_path / class_name).mkdir(exist_ok=True)
        
        # Copy train images
        for idx, img_path in enumerate(train_images):
            dest = train_path / class_name / f"{class_name.replace(' ', '_')}_{idx:04d}{img_path.suffix}"
            shutil.copy2(img_path, dest)
            total_train += 1
        
        # Copy val images
        for idx, img_path in enumerate(val_images):
            dest = val_path / class_name / f"{class_name.replace(' ', '_')}_{idx:04d}{img_path.suffix}"
            shutil.copy2(img_path, dest)
            total_val += 1
    
    print(f"\n✅ Dataset merged successfully!")
    print(f"   Train: {total_train} images ({len(images_by_class)} classes)")
    print(f"   Valid: {total_val} images ({len(images_by_class)} classes)")
    print(f"   Output: {output_path}")
    
    # Save dataset info
    info = {
        "num_classes": len(images_by_class),
        "classes": list(images_by_class.keys()),
        "train_images": total_train,
        "val_images": total_val,
        "train_split": train_split
    }
    
    with open(output_path / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    return output_path

def analyze_dataset(dataset_path):
    """
    วิเคราะห์ dataset และแสดงสถิติ
    """
    dataset_path = Path(dataset_path)
    
    print("\n" + "="*60)
    print("📊 Dataset Analysis")
    print("="*60)
    
    # Check for train/val folders
    splits = []
    if (dataset_path / 'train').exists():
        splits.append(('train', dataset_path / 'train'))
    if (dataset_path / 'valid').exists() or (dataset_path / 'val').exists():
        val_path = dataset_path / 'valid' if (dataset_path / 'valid').exists() else dataset_path / 'val'
        splits.append(('valid', val_path))
    if (dataset_path / 'test').exists():
        splits.append(('test', dataset_path / 'test'))
    
    if not splits:
        # Single folder with classes
        splits = [('all', dataset_path)]
    
    for split_name, split_path in splits:
        print(f"\n📁 {split_name.upper()}:")
        print("-" * 60)
        
        class_folders = [d for d in split_path.iterdir() if d.is_dir()]
        
        if not class_folders:
            print("  No class folders found")
            continue
        
        total_images = 0
        class_counts = {}
        
        for class_folder in sorted(class_folders):
            images = list(class_folder.glob("*.jpg")) + \
                    list(class_folder.glob("*.jpeg")) + \
                    list(class_folder.glob("*.png"))
            
            count = len(images)
            total_images += count
            class_counts[class_folder.name] = count
        
        print(f"  Total classes: {len(class_folders)}")
        print(f"  Total images: {total_images}")
        print(f"  Avg images per class: {total_images / len(class_folders):.1f}")
        print(f"  Min images: {min(class_counts.values())}")
        print(f"  Max images: {max(class_counts.values())}")
        
        # Show classes with few images
        few_images = [(k, v) for k, v in class_counts.items() if v < 10]
        if few_images:
            print(f"\n  ⚠️  Classes with <10 images:")
            for class_name, count in sorted(few_images, key=lambda x: x[1])[:5]:
                print(f"    - {class_name}: {count} images")
    
    print("\n" + "="*60)

# Main execution
if __name__ == "__main__":
    print("🎴 Playing Card Dataset Converter")
    print("="*60)
    print()
    print("Options:")
    print("1. Analyze existing dataset")
    print("2. Merge two datasets")
    print("3. Convert flat structure to class folders")
    print("4. Detect dataset format")
    print()
    
    choice = input("Select option (1-4): ").strip()
    
    if choice == "1":
        path = input("Enter dataset path: ").strip()
        analyze_dataset(path)
    
    elif choice == "2":
        dataset1 = input("Enter first dataset path: ").strip()
        dataset2 = input("Enter second dataset path: ").strip()
        output = input("Enter output path: ").strip()
        
        merge_datasets(dataset1, dataset2, output)
        analyze_dataset(output)
    
    elif choice == "3":
        source = input("Enter source path (flat structure): ").strip()
        output = input("Enter output path: ").strip()
        
        convert_flat_to_class_folders(source, output)
    
    elif choice == "4":
        path = input("Enter dataset path: ").strip()
        format_type = detect_dataset_format(path)
        print(f"\nDetected format: {format_type}")
    
    print("\n✅ Done!")
