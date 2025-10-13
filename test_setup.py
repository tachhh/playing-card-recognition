"""
Test script to verify all imports work correctly
"""
import cv2
import numpy as np
import torch
import scipy
from src.preprocessing.image_preprocessing import preprocess_image
from src.feature_extraction.feature_extractor import extract_features
from src.classification.card_classifier import CardClassifier

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    print(f"✓ OpenCV version: {cv2.__version__}")
    print(f"✓ NumPy version: {np.__version__}")
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ SciPy version: {scipy.__version__}")
    print("✓ All module imports successful!")
    
    # Test that functions are importable
    print("\nTesting function imports...")
    print(f"✓ preprocess_image: {preprocess_image}")
    print(f"✓ extract_features: {extract_features}")
    print(f"✓ CardClassifier: {CardClassifier}")
    
    print("\n✅ All tests passed! The project is properly configured.")

if __name__ == "__main__":
    test_imports()
