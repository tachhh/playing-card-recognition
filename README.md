# Playing Card Recognition System

A computer vision system that detects and classifies playing cards in real-time using linear algebra and deep learning techniques.

## Features
- Single playing card detection from images
- Playing card suit and rank classification
- Real-time recognition using webcam
- Application of vector and matrix theory in computer vision

## Requirements
- Python 3.8+
- OpenCV
- NumPy
- PyTorch/TensorFlow
- Matplotlib

## Installation
```bash
git clone https://github.com/yourusername/playing-card-recognition.git
cd playing-card-recognition
pip install -r requirements.txt
```

## Usage
For real-time detection using webcam:
```bash
python app/run_camera.py
```

For testing on images:
```bash
python app/test_image.py --image path/to/image.jpg
```

## Project Structure
- `data/`: Dataset of playing card images
- `src/`: Source code for preprocessing, feature extraction, and classification
- `notebooks/`: Jupyter notebooks for experimentation and demonstration
- `app/`: Application code for real-time recognition
- `models/`: Saved model weights

## Mathematical Foundation
This project employs several linear algebra concepts:
- Matrix transformations for image preprocessing
- Eigenvectors and eigenvalues for feature extraction
- Singular value decomposition (SVD) for dimensionality reduction
- Vector spaces and projections for classification

## License
[MIT](LICENSE)