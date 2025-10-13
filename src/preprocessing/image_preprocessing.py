import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocess the input image for card detection.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image in BGR format
        
    Returns:
    --------
    processed_img : numpy.ndarray
        Processed image ready for feature extraction
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    # Using a matrix operation (convolution with Gaussian kernel)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive threshold to get binary image
    # This is another matrix operation (element-wise comparison)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (assuming it's the card)
    if contours:
        card_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(card_contour)
        
        # Extract the card region
        card_img = image[y:y+h, x:x+w]
        
        # Resize to a standard size (matrix transformation)
        card_img_resized = cv2.resize(card_img, (200, 300))
        
        return card_img_resized
    
    return None
