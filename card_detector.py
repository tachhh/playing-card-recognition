"""
Card Detector Module
Implements single playing card detection using computer vision techniques
including linear algebra operations (matrix transformations, contour detection)
"""

import cv2
import numpy as np


class CardDetector:
    """Detects playing cards in images using contour detection and perspective transforms"""
    
    def __init__(self, min_area=5000, max_area=200000):
        """
        Initialize the card detector
        
        Args:
            min_area: Minimum contour area to consider as a card
            max_area: Maximum contour area to consider as a card
        """
        self.min_area = min_area
        self.max_area = max_area
        
    def preprocess_image(self, image):
        """
        Preprocess image for card detection
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed binary image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return thresh
    
    def find_card_contours(self, thresh_image):
        """
        Find card contours in preprocessed image
        
        Args:
            thresh_image: Binary threshold image
            
        Returns:
            List of valid card contours
        """
        # Find contours
        contours, _ = cv2.findContours(
            thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        card_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if self.min_area < area < self.max_area:
                # Approximate contour to polygon
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # Check if contour has 4 corners (card shape)
                if len(approx) == 4:
                    card_contours.append(approx)
        
        return card_contours
    
    def order_points(self, pts):
        """
        Order points in top-left, top-right, bottom-right, bottom-left order
        Uses vector operations to determine point ordering
        
        Args:
            pts: Array of 4 points
            
        Returns:
            Ordered array of points
        """
        rect = np.zeros((4, 2), dtype="float32")
        
        # Sum and difference for corner identification
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        rect[0] = pts[np.argmin(s)]      # Top-left: smallest sum
        rect[2] = pts[np.argmax(s)]      # Bottom-right: largest sum
        rect[1] = pts[np.argmin(diff)]   # Top-right: smallest difference
        rect[3] = pts[np.argmax(diff)]   # Bottom-left: largest difference
        
        return rect
    
    def get_perspective_transform(self, image, contour):
        """
        Apply perspective transform to extract card region
        Uses matrix transformation (homography) to warp perspective
        
        Args:
            image: Source image
            contour: Card contour with 4 points
            
        Returns:
            Warped card image
        """
        # Reshape contour points
        pts = contour.reshape(4, 2)
        rect = self.order_points(pts)
        
        # Define card dimensions (standard poker card ratio ~1.4)
        width = 200
        height = 280
        
        # Destination points for perspective transform
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")
        
        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Apply perspective transformation
        warped = cv2.warpPerspective(image, M, (width, height))
        
        return warped
    
    def detect_cards(self, image):
        """
        Detect all cards in an image
        
        Args:
            image: Input BGR image
            
        Returns:
            List of tuples (card_image, contour)
        """
        # Preprocess image
        thresh = self.preprocess_image(image)
        
        # Find card contours
        contours = self.find_card_contours(thresh)
        
        # Extract card regions
        cards = []
        for contour in contours:
            card_img = self.get_perspective_transform(image, contour)
            cards.append((card_img, contour))
        
        return cards
    
    def draw_card_contours(self, image, contours, color=(0, 255, 0), thickness=3):
        """
        Draw detected card contours on image
        
        Args:
            image: Image to draw on
            contours: List of contours to draw
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Image with drawn contours
        """
        result = image.copy()
        cv2.drawContours(result, contours, -1, color, thickness)
        return result
