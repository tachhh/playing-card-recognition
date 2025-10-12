# Playing Card Recognition System

A computer vision system that detects and classifies playing cards in real-time using linear algebra and computer vision techniques.

## Features

- **Single Playing Card Detection**: Detect playing cards from static images using contour detection and perspective transformation
- **Card Classification**: Classify playing cards by suit (Hearts, Diamonds, Clubs, Spades) and rank (A, 2-10, J, Q, K)
- **Real-time Recognition**: Live webcam feed processing for real-time card detection and classification
- **Linear Algebra Applications**: 
  - Matrix transformations for perspective correction (homography)
  - Vector operations for point ordering and corner detection
  - Normalized cross-correlation using dot products for template matching
  - Contour analysis using geometric features

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/tachhh/playing-card-recognition.git
cd playing-card-recognition
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Single Card Detection

Detect and classify playing cards from an image file:

```bash
python detect_single_card.py <image_path> [output_path]
```

Example:
```bash
python detect_single_card.py card.jpg result.jpg
```

This will:
- Load the image
- Detect all playing cards in the image
- Classify each card by rank and suit
- Display or save the annotated result

### Real-time Webcam Recognition

Run real-time card recognition using your webcam:

```bash
python realtime_recognition.py
```

Controls:
- Press `q` to quit
- Press `s` to save the current frame

### Programmatic Usage

You can also use the modules in your own Python scripts:

```python
from card_detector import CardDetector
from card_classifier import CardClassifier
import cv2

# Initialize detector and classifier
detector = CardDetector()
classifier = CardClassifier()

# Load image
image = cv2.imread('card.jpg')

# Detect cards
cards = detector.detect_cards(image)

# Classify each detected card
for card_img, contour in cards:
    rank, suit = classifier.classify_card(card_img)
    print(f"Detected: {rank} of {suit}")
```

## System Architecture

### Card Detection (`card_detector.py`)

The `CardDetector` class implements card detection using:

1. **Image Preprocessing**: Converts to grayscale, applies Gaussian blur, and adaptive thresholding
2. **Contour Detection**: Finds contours in the binary image
3. **Shape Filtering**: Filters contours by area and shape (4-sided polygons)
4. **Perspective Transform**: Applies homography matrix to extract warped card images
   - Uses `order_points()` to determine corner ordering via vector operations
   - Computes perspective transformation matrix using linear algebra
   - Warps the perspective to obtain a normalized card view

### Card Classification (`card_classifier.py`)

The `CardClassifier` class implements card classification using:

1. **Corner Extraction**: Extracts the top-left corner containing rank and suit symbols
2. **Region Segmentation**: Separates rank and suit regions
3. **Feature Extraction**: Computes geometric features (area, perimeter, circularity, aspect ratio, solidity)
4. **Classification**: Uses contour analysis and geometric heuristics to determine rank and suit
   - Template matching with normalized cross-correlation
   - Shape analysis using convex hull and contour properties

### Real-time Recognition (`realtime_recognition.py`)

The `RealtimeCardRecognition` class combines detection and classification for webcam processing:

1. Captures frames from webcam
2. Processes each frame for card detection
3. Classifies detected cards
4. Overlays results on the video feed
5. Displays real-time annotated video

## Linear Algebra Techniques

This project demonstrates practical applications of linear algebra in computer vision:

### 1. Perspective Transformation (Homography)
- Uses 3x3 transformation matrix to map quadrilateral to rectangle
- Implemented in `get_perspective_transform()` method
- Enables normalization of card orientation and perspective

### 2. Vector Operations
- Point ordering using sum and difference operations
- Centroid calculation using moments (weighted averages)
- Distance computations for shape analysis

### 3. Matrix Operations
- Gaussian blur kernel convolution
- Image transformations and rotations
- Contour approximation using least squares

### 4. Normalized Cross-Correlation
- Template matching using dot product: `correlation = Σ(query · template) / (||query|| · ||template||)`
- Implemented in `match_template()` method
- Provides similarity score between 0 and 1

## Technical Details

### Image Processing Pipeline

1. **Preprocessing**
   - Convert to grayscale
   - Apply Gaussian blur (5x5 kernel)
   - Adaptive thresholding (11x11 block size)

2. **Card Detection**
   - Find external contours
   - Filter by area (5000 - 200000 pixels)
   - Approximate to 4-sided polygon
   - Apply perspective transform

3. **Classification**
   - Extract corner region (20% width, 25% height)
   - Separate rank and suit regions
   - Extract geometric features
   - Apply classification heuristics

### Supported Cards

- **Ranks**: A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K
- **Suits**: Hearts (♥), Diamonds (♦), Clubs (♣), Spades (♠)

## Limitations

- Works best with standard poker-size playing cards
- Requires good lighting conditions
- Cards should be clearly visible and not overlapping
- Classification accuracy depends on card design and image quality
- Current implementation uses simple heuristics; deep learning models would provide better accuracy

## Future Enhancements

- Implement deep learning-based classification (CNN)
- Add support for multiple overlapping cards
- Improve classification accuracy with template database
- Add card tracking across frames
- Support for different card designs and decks
- GPU acceleration for real-time processing

## Dependencies

- **numpy**: Numerical operations and array handling
- **opencv-python**: Computer vision and image processing
- **opencv-contrib-python**: Additional OpenCV modules
- **Pillow**: Image handling utilities

## License

This project is open source and available for educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

This project demonstrates practical applications of:
- Computer vision techniques
- Linear algebra in image processing
- Geometric transformations
- Shape analysis and feature extraction