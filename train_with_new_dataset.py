"""
Train with New Dataset
เทรนโมเดลด้วย dataset เดิม + dataset ใหม่ที่รวมกัน
"""

import os
import sys

def main():
    print("🎴 Training with Additional Dataset")
    print("="*70)
    print()
    print("📝 Steps:")
    print("1. Place your new dataset in any folder")
    print("2. Run this script to merge datasets")
    print("3. Train model with combined dataset")
    print()
    
    # Get paths
    print("Current dataset path:")
    print("  C:\\Users\\User\\.cache\\kagglehub\\datasets\\gpiosenka\\cards-image-datasetclassification\\versions\\2")
    print()
    print("💡 Examples of NEW dataset path:")
    print("  - C:\\playing-card-recognition\\playing-card-recognition\\data\\new_dataset")
    print("  - D:\\my_cards")
    print("  - C:\\Users\\YourName\\Downloads\\card_dataset")
    print()
    print("⚠️  Make sure it's a FOLDER path, not a file!")
    print()
    
    new_dataset = input("Enter path to NEW dataset folder: ").strip()
    
    # Remove quotes if user copied path with quotes
    new_dataset = new_dataset.strip('"').strip("'")
    
    if not os.path.exists(new_dataset):
        print(f"\n❌ Path not found: {new_dataset}")
        print("\n💡 Tips:")
        print("  1. Copy folder path from File Explorer address bar")
        print("  2. Make sure it's a folder, not a file")
        print("  3. Use format like: C:\\folder\\subfolder")
        input("\nPress Enter to exit...")
        return
    
    if not os.path.isdir(new_dataset):
        print(f"\n❌ This is not a folder: {new_dataset}")
        print("\n💡 Please provide a folder path containing your dataset images")
        input("\nPress Enter to exit...")
        return
    
    # Check if folder has any images
    import glob
    images = glob.glob(os.path.join(new_dataset, "**/*.jpg"), recursive=True) + \
             glob.glob(os.path.join(new_dataset, "**/*.png"), recursive=True) + \
             glob.glob(os.path.join(new_dataset, "**/*.jpeg"), recursive=True)
    
    if len(images) == 0:
        print(f"\n❌ No images found in: {new_dataset}")
        print("\n💡 Make sure the folder contains:")
        print("  - .jpg, .png, or .jpeg files")
        print("  - Either directly or in subfolders")
        input("\nPress Enter to exit...")
        return
    
    print(f"\n✅ Found {len(images)} images in the dataset!")
    
    output_merged = "data/merged_dataset"
    
    print(f"\n📂 Will merge to: {output_merged}")
    confirm = input("Continue? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Import converter
    from convert_dataset import merge_datasets, analyze_dataset
    
    # Original dataset
    original_dataset = r"C:\Users\User\.cache\kagglehub\datasets\gpiosenka\cards-image-datasetclassification\versions\2\train"
    
    print("\n🔄 Merging datasets...")
    merged_path = merge_datasets(original_dataset, new_dataset, output_merged, train_split=0.85)
    
    print("\n📊 Analyzing merged dataset...")
    analyze_dataset(merged_path)
    
    print("\n" + "="*70)
    print("✅ Datasets merged successfully!")
    print()
    print("📝 Next steps:")
    print("1. Review the dataset analysis above")
    print("2. Update train_cnn_model.py to use new dataset path:")
    print(f"   dataset_path = r'{os.path.abspath(output_merged)}'")
    print("3. Run: python train_cnn_model.py")
    print()
    print("💡 Tips:")
    print("  - More data = better accuracy")
    print("  - Aim for balanced classes (similar image counts)")
    print("  - Check for duplicate images between datasets")
    print("="*70)

if __name__ == "__main__":
    main()
