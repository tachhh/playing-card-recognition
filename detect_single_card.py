"""
Single Card Detection Script
Detects and classifies a single playing card from an image file
"""

import cv2
import sys
import os
from card_detector import CardDetector
from card_classifier import CardClassifier


def detect_and_classify_card(image_path, output_path=None):
    """
    Detect and classify playing card from image
    
    Args:
        image_path: Path to input image
        output_path: Optional path to save annotated output image
        
    Returns:
        List of detected cards with their classifications
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        return None
    
    # Read image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read image '{image_path}'")
        return None
    
    # Initialize detector and classifier
    detector = CardDetector()
    classifier = CardClassifier()
    
    # Detect cards
    cards = detector.detect_cards(image)
    
    print(f"Detected {len(cards)} card(s)")
    
    # Classify each card
    results = []
    result_image = image.copy()
    
    for i, (card_img, contour) in enumerate(cards):
        # Classify the card
        rank, suit = classifier.classify_card(card_img)
        card_name = classifier.get_card_name(rank, suit)
        
        print(f"Card {i+1}: {card_name}")
        
        results.append({
            'rank': rank,
            'suit': suit,
            'name': card_name,
            'contour': contour
        })
        
        # Draw contour and label on result image
        cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 3)
        
        # Get contour center for text placement
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Draw card label
            label = f"{rank} {suit}"
            cv2.putText(result_image, label, (cx - 50, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Save or display result
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"Saved result to '{output_path}'")
    else:
        # Display result
        cv2.imshow('Card Detection Result', result_image)
        print("Press any key to close the window")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return results


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python detect_single_card.py <image_path> [output_path]")
        print("Example: python detect_single_card.py card.jpg result.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    detect_and_classify_card(image_path, output_path)


if __name__ == "__main__":
    main()
