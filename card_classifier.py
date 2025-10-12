"""
Card Classifier Module
Implements playing card suit and rank classification using template matching
and feature extraction based on linear algebra operations
"""

import cv2
import numpy as np
import os


class CardClassifier:
    """Classifies playing cards by suit and rank using template matching"""
    
    # Card ranks and suits
    RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    SUIT_SYMBOLS = {'Hearts': '♥', 'Diamonds': '♦', 'Clubs': '♣', 'Spades': '♠'}
    
    def __init__(self):
        """Initialize the card classifier"""
        self.rank_templates = None
        self.suit_templates = None
        
    def extract_corner_region(self, card_image, top_left=True):
        """
        Extract corner region containing rank and suit
        
        Args:
            card_image: Card image
            top_left: If True, extract top-left corner, else bottom-right
            
        Returns:
            Corner region image
        """
        height, width = card_image.shape[:2]
        
        if top_left:
            # Extract top-left corner (20% width, 25% height)
            corner = card_image[0:int(height*0.25), 0:int(width*0.2)]
        else:
            # Extract bottom-right corner (rotated)
            corner = card_image[int(height*0.75):height, int(width*0.8):width]
            corner = cv2.rotate(corner, cv2.ROTATE_180)
        
        return corner
    
    def preprocess_corner(self, corner):
        """
        Preprocess corner region for feature extraction
        
        Args:
            corner: Corner region image
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale if needed
        if len(corner.shape) == 3:
            gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        else:
            gray = corner
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def extract_rank_suit_regions(self, corner):
        """
        Extract separate regions for rank and suit from corner
        
        Args:
            corner: Corner region image
            
        Returns:
            Tuple of (rank_region, suit_region)
        """
        height, width = corner.shape[:2]
        
        # Rank is typically in top 40% of corner
        rank_region = corner[0:int(height*0.4), :]
        
        # Suit is typically in middle 40-80% of corner
        suit_region = corner[int(height*0.4):int(height*0.8), :]
        
        return rank_region, suit_region
    
    def match_template(self, query_img, template_img):
        """
        Match query image against template using normalized correlation
        Uses matrix operations for template matching
        
        Args:
            query_img: Query image to match
            template_img: Template image
            
        Returns:
            Matching score (0-1, higher is better)
        """
        # Resize query to match template size
        query_resized = cv2.resize(query_img, 
                                   (template_img.shape[1], template_img.shape[0]))
        
        # Normalize both images
        query_norm = query_resized.astype(float) / 255.0
        template_norm = template_img.astype(float) / 255.0
        
        # Compute normalized cross-correlation
        # This uses vector dot product: correlation = sum(query * template) / (||query|| * ||template||)
        correlation = np.sum(query_norm * template_norm)
        query_norm_val = np.sqrt(np.sum(query_norm ** 2))
        template_norm_val = np.sqrt(np.sum(template_norm ** 2))
        
        if query_norm_val > 0 and template_norm_val > 0:
            score = correlation / (query_norm_val * template_norm_val)
        else:
            score = 0
        
        return score
    
    def classify_rank_simple(self, rank_region):
        """
        Simple rank classification based on contour analysis
        
        Args:
            rank_region: Preprocessed rank region
            
        Returns:
            Predicted rank or 'Unknown'
        """
        # Find contours in rank region
        contours, _ = cv2.findContours(rank_region, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 'Unknown'
        
        # Get largest contour (should be the rank symbol)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate contour features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if area == 0 or perimeter == 0:
            return 'Unknown'
        
        # Circularity = 4*pi*area / perimeter^2
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Simple heuristic classification based on shape features
        # This is a simplified approach - in practice, template matching or ML would be better
        if circularity > 0.7:
            return 'A'  # Aces tend to be circular
        elif aspect_ratio > 1.5:
            return '10'  # 10s are wider
        else:
            return 'Unknown'
    
    def classify_suit_simple(self, suit_region):
        """
        Simple suit classification based on contour analysis
        
        Args:
            suit_region: Preprocessed suit region
            
        Returns:
            Predicted suit or 'Unknown'
        """
        # Find contours in suit region
        contours, _ = cv2.findContours(suit_region, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 'Unknown'
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate convexity
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(largest_contour)
        
        if hull_area == 0:
            return 'Unknown'
        
        solidity = float(contour_area) / hull_area
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Simple heuristic classification
        # Hearts and Spades tend to have lower solidity (due to curves/points)
        # Diamonds are more elongated, Clubs have multiple components
        if solidity < 0.8 and aspect_ratio < 1.2:
            return 'Hearts'
        elif aspect_ratio > 1.0 and aspect_ratio < 1.3:
            return 'Diamonds'
        elif solidity > 0.8:
            return 'Clubs'
        else:
            return 'Spades'
    
    def classify_card(self, card_image):
        """
        Classify a card image to determine rank and suit
        
        Args:
            card_image: Warped card image
            
        Returns:
            Tuple of (rank, suit)
        """
        # Extract corner region
        corner = self.extract_corner_region(card_image, top_left=True)
        
        # Preprocess corner
        corner_preprocessed = self.preprocess_corner(corner)
        
        # Extract rank and suit regions
        rank_region, suit_region = self.extract_rank_suit_regions(corner_preprocessed)
        
        # Classify rank and suit
        rank = self.classify_rank_simple(rank_region)
        suit = self.classify_suit_simple(suit_region)
        
        return rank, suit
    
    def get_card_name(self, rank, suit):
        """
        Get full card name from rank and suit
        
        Args:
            rank: Card rank
            suit: Card suit
            
        Returns:
            Card name string
        """
        suit_symbol = self.SUIT_SYMBOLS.get(suit, '')
        return f"{rank}{suit_symbol} of {suit}"
