"""
Download Cards Image Dataset from Kaggle
Dataset: gpiosenka/cards-image-datasetclassification

Dataset Info:
- 7,624 training images
- 265 test images
- 265 validation images
- Image size: 224 x 224 x 3 (jpg format)
- 53 classes (52 cards + 1 joker)
- All images cropped to show single card
"""

import kagglehub
import os
import shutil

def download_dataset():
    """Download the cards dataset from Kaggle"""
    print("🎴 Downloading Cards Image Dataset from Kaggle...")
    print("This may take a few minutes depending on your internet speed.")
    print("-" * 60)
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("gpiosenka/cards-image-datasetclassification")
        
        print("\n✅ Download completed!")
        print(f"📁 Dataset downloaded to: {path}")
        
        # Display dataset structure
        print("\n📊 Dataset Structure:")
        print("-" * 60)
        
        for root, dirs, files in os.walk(path):
            level = root.replace(path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            
            # Only show first level directories
            if level == 0:
                for d in sorted(dirs):
                    print(f'{indent}  {d}/')
                    
                    # Count files in each directory
                    dir_path = os.path.join(root, d)
                    if os.path.isdir(dir_path):
                        subdirs = [s for s in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, s))]
                        if subdirs:
                            total_files = 0
                            for subdir in subdirs:
                                subdir_path = os.path.join(dir_path, subdir)
                                file_count = len([f for f in os.listdir(subdir_path) if f.endswith('.jpg')])
                                total_files += file_count
                            print(f'{indent}    └─ {len(subdirs)} classes, {total_files} images total')
                break
        
        # Create symbolic link or copy to data directory
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        if not os.path.exists(data_dir):
            print(f"\n📂 Creating data directory at: {data_dir}")
            os.makedirs(data_dir, exist_ok=True)
        
        # Create a reference file
        ref_file = os.path.join(data_dir, 'dataset_path.txt')
        with open(ref_file, 'w') as f:
            f.write(path)
        print(f"✅ Dataset path saved to: {ref_file}")
        
        print("\n" + "=" * 60)
        print("🎉 Dataset ready to use!")
        print("=" * 60)
        print("\n📝 Next steps:")
        print("1. Run 'python explore_dataset.py' to explore the data")
        print("2. Run 'python train_model.py' to train the model")
        
        return path
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\n💡 Tips:")
        print("1. Make sure you have Kaggle API credentials set up")
        print("2. Visit: https://www.kaggle.com/docs/api")
        print("3. Download kaggle.json and place it in: ~/.kaggle/")
        return None

if __name__ == "__main__":
    download_dataset()
