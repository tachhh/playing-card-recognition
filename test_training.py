"""
Test Model Training Components
Test feature extraction, data loading, and model before full training
"""

import os
import cv2
import torch
import numpy as np
from src.feature_extraction.feature_extractor import extract_features
from src.preprocessing.image_preprocessing import preprocess_image
from src.classification.card_classifier import CardClassifierNN

def test_feature_extraction():
    """Test feature extraction on sample images"""
    print("🧪 Test 1: Feature Extraction")
    print("-" * 70)
    
    # Load dataset path
    ref_file = os.path.join('data', 'dataset_path.txt')
    if not os.path.exists(ref_file):
        print("❌ Dataset not found. Run 'download_dataset.py' first.")
        return False
    
    with open(ref_file, 'r') as f:
        dataset_path = f.read().strip()
    
    train_path = os.path.join(dataset_path, 'train')
    
    # Test on 5 random images
    classes = sorted(os.listdir(train_path))[:5]
    
    for cls in classes:
        cls_path = os.path.join(train_path, cls)
        images = [f for f in os.listdir(cls_path) if f.endswith('.jpg')][:1]
        
        for img_name in images:
            img_path = os.path.join(cls_path, img_name)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"   ❌ Failed to load: {cls}/{img_name}")
                continue
            
            # Extract features
            features = extract_features(image)
            
            if features is None:
                print(f"   ❌ Failed to extract features: {cls}/{img_name}")
                continue
            
            print(f"   ✅ {cls:30s} | Shape: {image.shape} | Features: {features.shape} | Mean: {features.mean():.2f}")
    
    print("   ✅ Feature extraction working!\n")
    return True

def test_model_architecture():
    """Test model architecture"""
    print("🧪 Test 2: Model Architecture")
    print("-" * 70)
    
    input_size = 70
    num_classes = 53
    batch_size = 4
    
    # Create model
    model = CardClassifierNN(input_size, num_classes)
    print(f"   Model created: {model.__class__.__name__}")
    
    # Test forward pass
    dummy_input = torch.randn(batch_size, input_size)
    output = model(dummy_input)
    
    print(f"   Input shape:  {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output sum:   {output.sum(dim=1)} (should be ~1.0 for each sample)")
    
    assert output.shape == (batch_size, num_classes), "Output shape mismatch!"
    assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=0.01), "Softmax not summing to 1!"
    
    print("   ✅ Model architecture working!\n")
    return True

def test_data_loading():
    """Test data loading and batching"""
    print("🧪 Test 3: Data Loading")
    print("-" * 70)
    
    # Load dataset path
    ref_file = os.path.join('data', 'dataset_path.txt')
    with open(ref_file, 'r') as f:
        dataset_path = f.read().strip()
    
    train_path = os.path.join(dataset_path, 'train')
    
    # Count samples
    total_images = 0
    classes = sorted([d for d in os.listdir(train_path) 
                     if os.path.isdir(os.path.join(train_path, d))])
    
    for cls in classes[:5]:  # Test first 5 classes
        cls_path = os.path.join(train_path, cls)
        images = [f for f in os.listdir(cls_path) if f.endswith('.jpg')]
        total_images += len(images)
    
    print(f"   Number of classes (sample): {len(classes[:5])}")
    print(f"   Number of images (sample):  {total_images}")
    
    # Test loading a batch
    print("\n   Testing batch loading...")
    batch_features = []
    batch_labels = []
    
    for idx, cls in enumerate(classes[:3]):
        cls_path = os.path.join(train_path, cls)
        images = [f for f in os.listdir(cls_path) if f.endswith('.jpg')][:2]
        
        for img_name in images:
            img_path = os.path.join(cls_path, img_name)
            image = cv2.imread(img_path)
            features = extract_features(image)
            
            if features is not None:
                batch_features.append(features)
                batch_labels.append(idx)
    
    batch_features = torch.tensor(np.array(batch_features), dtype=torch.float32)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)
    
    print(f"   Batch features shape: {batch_features.shape}")
    print(f"   Batch labels shape:   {batch_labels.shape}")
    print(f"   Labels:               {batch_labels.tolist()}")
    
    print("   ✅ Data loading working!\n")
    return True

def test_training_step():
    """Test a single training step"""
    print("🧪 Test 4: Single Training Step")
    print("-" * 70)
    
    input_size = 70
    num_classes = 53
    batch_size = 4
    
    # Create model, loss, optimizer
    model = CardClassifierNN(input_size, num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create dummy batch
    features = torch.randn(batch_size, input_size)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    print(f"   Batch size: {batch_size}")
    print(f"   Features shape: {features.shape}")
    print(f"   Labels: {labels.tolist()}")
    
    # Forward pass
    outputs = model(features)
    loss = criterion(outputs, labels)
    
    print(f"   Initial loss: {loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Forward pass again
    outputs = model(features)
    loss_after = criterion(outputs, labels)
    
    print(f"   Loss after step: {loss_after.item():.4f}")
    print(f"   Loss decreased: {loss.item() > loss_after.item()}")
    
    # Check predictions
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == labels).sum().item() / batch_size * 100
    
    print(f"   Predictions: {predicted.tolist()}")
    print(f"   Accuracy: {accuracy:.2f}%")
    
    print("   ✅ Training step working!\n")
    return True

def estimate_training_time():
    """Estimate training time"""
    print("⏱️  Training Time Estimation")
    print("-" * 70)
    
    import time
    
    # Load dataset info
    ref_file = os.path.join('data', 'dataset_path.txt')
    with open(ref_file, 'r') as f:
        dataset_path = f.read().strip()
    
    train_path = os.path.join(dataset_path, 'train')
    
    # Count total images
    total_images = 0
    classes = sorted([d for d in os.listdir(train_path) 
                     if os.path.isdir(os.path.join(train_path, d))])
    
    for cls in classes:
        cls_path = os.path.join(train_path, cls)
        images = [f for f in os.listdir(cls_path) if f.endswith('.jpg')]
        total_images += len(images)
    
    print(f"   Total training images: {total_images}")
    
    # Time 10 image loads and feature extractions
    print("\n   Timing feature extraction...")
    
    start_time = time.time()
    count = 0
    
    for cls in classes[:2]:
        cls_path = os.path.join(train_path, cls)
        images = [f for f in os.listdir(cls_path) if f.endswith('.jpg')][:5]
        
        for img_name in images:
            img_path = os.path.join(cls_path, img_name)
            image = cv2.imread(img_path)
            features = extract_features(image)
            count += 1
    
    elapsed = time.time() - start_time
    time_per_image = elapsed / count
    
    print(f"   Time per image: {time_per_image:.4f} seconds")
    
    # Estimate total training time
    batch_size = 32
    num_epochs = 50
    
    batches_per_epoch = total_images // batch_size
    total_time_sec = time_per_image * total_images * num_epochs
    total_time_min = total_time_sec / 60
    
    print(f"\n   Estimated training time:")
    print(f"   - Batches per epoch: {batches_per_epoch}")
    print(f"   - Total batches: {batches_per_epoch * num_epochs}")
    print(f"   - Estimated time: {total_time_min:.1f} minutes ({total_time_min/60:.1f} hours)")
    
    if total_time_min > 60:
        print(f"\n   ⚠️  Training will take over 1 hour!")
        print(f"   💡 Consider reducing epochs to 20-30 for faster training")
    
    print()
    return True

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("🧪 TESTING MODEL TRAINING COMPONENTS")
    print("=" * 70 + "\n")
    
    tests = [
        ("Feature Extraction", test_feature_extraction),
        ("Model Architecture", test_model_architecture),
        ("Data Loading", test_data_loading),
        ("Training Step", test_training_step),
        ("Time Estimation", estimate_training_time),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"   ❌ Test failed with error: {e}\n")
            results.append((name, False))
    
    # Summary
    print("=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} - {name}")
    
    all_passed = all(success for _, success in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 All tests passed! Ready to train the model.")
        print("=" * 70)
        print("\n📝 Next step:")
        print("   Run: python train_model.py")
    else:
        print("⚠️  Some tests failed. Please fix issues before training.")
        print("=" * 70)
    
    return all_passed

if __name__ == "__main__":
    run_all_tests()
