import cv2
import numpy as np
from scipy import linalg

def extract_features(image):
    """
    Extract features from the preprocessed card image using linear algebra techniques.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Preprocessed card image
        
    Returns:
    --------
    features : numpy.ndarray
        Feature vector for classification
    """
    if image is None:
        return None
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Extract corner of the card (where rank and suit are typically located)
    h, w = gray.shape
    corner = gray[0:int(h/3), 0:int(w/3)]
    
    # Resize to a fixed size for consistency
    corner_resized = cv2.resize(corner, (50, 75))
    
    # Apply SVD (Singular Value Decomposition)
    # This is a key linear algebra technique for dimensionality reduction
    U, sigma, Vt = linalg.svd(corner_resized)
    
    # Use the singular values as features
    # This represents the "energy" distribution in the image
    features = sigma[:20]  # Take top 20 singular values
    
    # Additional features: HOG (Histogram of Oriented Gradients)
    # This uses vector operations to compute gradients
    win_size = (50, 75)
    block_size = (10, 10)
    block_stride = (5, 5)
    cell_size = (5, 5)
    nbins = 9
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_features = hog.compute(corner_resized)
    
    # Combine SVD and HOG features
    combined_features = np.concatenate([features, hog_features.flatten()[:50]])
    
    return combined_features
