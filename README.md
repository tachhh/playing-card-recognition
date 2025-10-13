# 🎴 Playing Card Recognition System

A computer vision system that detects and classifies 53 types of playing cards in real-time using CNN deep learning with **93.58% accuracy**.

## ✨ Features
- ✅ Real-time card recognition (30 FPS)
- ✅ 93.58% validation accuracy
- ✅ CNN with 26M parameters
- ✅ Fixed Frame detection mode (stable & fast)
- ✅ Easy-to-use batch launchers
- ✅ Comprehensive documentation

## 🚀 Quick Start

### Option 1: Double-click to run (Easiest!)
```
1. Double-click: start.bat
2. Choose option 1 (Run Camera)
3. Place card in the blue frame
4. Done!
```

### Option 2: Command Line
```bash
# Activate virtual environment (Windows)
C:\playing-card-recognition\.venv\Scripts\Activate.ps1

# Run camera
python camera_simple.py

# Test with real images
python test_real_image.py

# Quick model test
python quick_model_test.py
```

## 📋 Requirements
- Python 3.11+
- PyTorch
- OpenCV (cv2)
- torchvision
- PIL (Pillow)
- NumPy

## 🔧 Installation
```bash
git clone https://github.com/tachhh/playing-card-recognition.git
cd playing-card-recognition

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
python download_dataset.py
```

## 📁 Project Structure
```
playing-card-recognition/
├── camera_simple.py          # Main camera app (93.58% accuracy)
├── train_cnn_model.py        # Training script
├── test_real_image.py        # Test with real images
├── quick_model_test.py       # Quick model test
├── diagnose_inference.py     # Diagnose issues
├── download_dataset.py       # Download Kaggle dataset
├── start.bat                 # Main menu launcher
├── run_camera.bat            # Quick camera launcher
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── models/
│   ├── card_classifier_cnn.pth       # Trained model (93.58%)
│   ├── class_mapping_cnn.json        # Class mapping
│   └── training_history_*.json       # Training history
│
├── data/
│   └── dataset_path.txt              # Dataset location
│
├── captured_cards/
│   └── README.md                     # Saved images folder
│
└── docs/                             # 📚 All Documentation
    ├── PROJECT_HISTORY.md            # Development history
    ├── LINEAR_ALGEBRA_GUIDE.md       # Math theory & problems
    ├── DIAGNOSIS_COMPLETE.md         # Problem solving guide
    ├── MODEL_MANAGEMENT.md           # Model management
    ├── ADDING_DATASET.md             # Add custom dataset
    ├── AFTER_MERGE.md                # Merge dataset guide
    ├── HOW_TO_ADD_DATASET.md         # Dataset instructions
    ├── LOW_CONFIDENCE_FIX.md         # Fix low confidence
    └── SETUP_FIXED.md                # Setup guide
```

## 🎯 Model Performance
- **Validation Accuracy**: 93.58%
- **Train Accuracy**: 85.69%
- **Real Image Test**: 90% (9/10 correct)
- **Confidence**: 89-100%
- **Speed**: 30 FPS
- **Classes**: 53 card types

## 📚 Documentation

### Main Documentation (in `docs/` folder):
- 📖 **[PROJECT_HISTORY.md](docs/PROJECT_HISTORY.md)** - Complete development journey from 0% to 93.58%
- 📐 **[LINEAR_ALGEBRA_GUIDE.md](docs/LINEAR_ALGEBRA_GUIDE.md)** - 10 Linear Algebra theories used in this project
- 🔧 **[DIAGNOSIS_COMPLETE.md](docs/DIAGNOSIS_COMPLETE.md)** - Problem diagnosis and solutions
- 📦 **[MODEL_MANAGEMENT.md](docs/MODEL_MANAGEMENT.md)** - How to manage trained models

### Additional Guides:
- 📝 **[ADDING_DATASET.md](docs/ADDING_DATASET.md)** - Add your own card images
- 🔄 **[AFTER_MERGE.md](docs/AFTER_MERGE.md)** - After merging datasets
- 📥 **[HOW_TO_ADD_DATASET.md](docs/HOW_TO_ADD_DATASET.md)** - Dataset instructions
- ⚡ **[LOW_CONFIDENCE_FIX.md](docs/LOW_CONFIDENCE_FIX.md)** - Fix low confidence issues
- ⚙️ **[SETUP_FIXED.md](docs/SETUP_FIXED.md)** - Initial setup guide

## 🧮 Mathematical Foundation
This project employs Linear Algebra concepts (see [docs/LINEAR_ALGEBRA_GUIDE.md](docs/LINEAR_ALGEBRA_GUIDE.md)):
1. **Vectors** - Bias vectors, image representation
2. **Matrices** - Weight matrices, transformations
3. **Dot Product** - Fully connected layers
4. **Convolution** - Feature extraction
5. **Linear Transformations** - Normalization, activation
6. **Gaussian Blur** - Noise reduction

## 🎓 Key Learnings
1. **Dataset Quality Matters**: Kaggle dataset (93.58%) >> merged_dataset (65.28%)
2. **Learning Rate is Critical**: 0.0001 >> 0.001 (28% accuracy difference)
3. **More Epochs Needed**: 50 epochs >> 30 epochs
4. **Fixed Frame > Auto Detection**: More stable, faster, more accurate

## 🙏 Credits
- **Dataset**: [Kaggle Cards Image Dataset](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification) by gpiosenka
- **Framework**: PyTorch
- **Author**: tachhh

## 📄 License
MIT License - see LICENSE file for details

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

---

**Made with ❤️ for Linear Algebra course project**  
**Final Accuracy: 93.58% ⭐**