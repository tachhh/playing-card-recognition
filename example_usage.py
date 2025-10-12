"""
Example Usage Script
Demonstrates how to use the card detection and classification modules
"""

import cv2
import numpy as np
from card_detector import CardDetector
from card_classifier import CardClassifier


def create_sample_card_image():
    """
    Create a simple sample card-like image for testing
    
    Returns:
        Sample image with a white rectangle (simulating a card)
    """
    # Create a dark background
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    image[:] = (50, 50, 50)
    
    # Draw a white rectangle to simulate a card
    card_points = np.array([
        [200, 100],
        [450, 120],
        [440, 450],
        [190, 430]
    ], dtype=np.int32)
    
    cv2.fillPoly(image, [card_points], (255, 255, 255))
    
    # Add some simple markings
    cv2.putText(image, 'A', (250, 180), cv2.FONT_HERSHEY_SIMPLEX, 
                2, (0, 0, 0), 3)
    cv2.circle(image, (270, 250), 20, (255, 0, 0), -1)
    
    return image


def example_basic_detection():
    """Example: Basic card detection"""
    print("=" * 60)
    print("Example 1: Basic Card Detection")
    print("=" * 60)
    
    # Create sample image
    image = create_sample_card_image()
    
    # Initialize detector
    detector = CardDetector()
    
    # Detect cards
    cards = detector.detect_cards(image)
    
    print(f"Number of cards detected: {len(cards)}")
    
    # Draw contours
    contours = [contour for _, contour in cards]
    result_image = detector.draw_card_contours(image, contours)
    
    # Display result
    cv2.imshow('Basic Detection', result_image)
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return cards


def example_card_classification(cards):
    """Example: Card classification"""
    print("\n" + "=" * 60)
    print("Example 2: Card Classification")
    print("=" * 60)
    
    if not cards:
        print("No cards to classify")
        return
    
    # Initialize classifier
    classifier = CardClassifier()
    
    # Classify each card
    for i, (card_img, contour) in enumerate(cards):
        rank, suit = classifier.classify_card(card_img)
        card_name = classifier.get_card_name(rank, suit)
        
        print(f"Card {i+1}: {card_name}")
        
        # Display the warped card image
        cv2.imshow(f'Card {i+1} - {rank} {suit}', card_img)
    
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def example_perspective_transform():
    """Example: Demonstrate perspective transformation"""
    print("\n" + "=" * 60)
    print("Example 3: Perspective Transformation")
    print("=" * 60)
    
    # Create sample image
    image = create_sample_card_image()
    
    # Initialize detector
    detector = CardDetector()
    
    # Get threshold image
    thresh = detector.preprocess_image(image)
    
    # Find contours
    contours = detector.find_card_contours(thresh)
    
    if contours:
        print(f"Found {len(contours)} card contour(s)")
        
        # Apply perspective transform to first contour
        warped = detector.get_perspective_transform(image, contours[0])
        
        print("Original image shape:", image.shape)
        print("Warped card shape:", warped.shape)
        
        # Display both
        cv2.imshow('Original', image)
        cv2.imshow('Warped Card', warped)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No contours found")


def example_corner_extraction():
    """Example: Corner extraction for classification"""
    print("\n" + "=" * 60)
    print("Example 4: Corner Extraction")
    print("=" * 60)
    
    # Create sample image
    image = create_sample_card_image()
    
    # Initialize detector and classifier
    detector = CardDetector()
    classifier = CardClassifier()
    
    # Detect cards
    cards = detector.detect_cards(image)
    
    if cards:
        card_img, _ = cards[0]
        
        # Extract corner
        corner = classifier.extract_corner_region(card_img, top_left=True)
        
        # Preprocess corner
        corner_preprocessed = classifier.preprocess_corner(corner)
        
        # Extract rank and suit regions
        rank_region, suit_region = classifier.extract_rank_suit_regions(corner_preprocessed)
        
        print("Card image shape:", card_img.shape)
        print("Corner shape:", corner.shape)
        print("Rank region shape:", rank_region.shape)
        print("Suit region shape:", suit_region.shape)
        
        # Display regions
        cv2.imshow('Full Card', card_img)
        cv2.imshow('Corner', corner)
        cv2.imshow('Corner Preprocessed', corner_preprocessed)
        cv2.imshow('Rank Region', rank_region)
        cv2.imshow('Suit Region', suit_region)
        
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No cards detected")


def main():
    """Run all examples"""
    print("\n")
    print("=" * 60)
    print("Playing Card Recognition - Example Usage")
    print("=" * 60)
    print("\nThis script demonstrates the card detection and classification")
    print("functionality using a simple test image.")
    print("\n")
    
    # Run examples
    cards = example_basic_detection()
    example_card_classification(cards)
    example_perspective_transform()
    example_corner_extraction()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
