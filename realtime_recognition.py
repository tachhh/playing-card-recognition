"""
Real-time Playing Card Recognition
Implements real-time card detection and classification using webcam
"""

import cv2
import numpy as np
from card_detector import CardDetector
from card_classifier import CardClassifier


class RealtimeCardRecognition:
    """Real-time card recognition using webcam"""
    
    def __init__(self):
        """Initialize the real-time recognition system"""
        self.detector = CardDetector()
        self.classifier = CardClassifier()
        
    def process_frame(self, frame):
        """
        Process a single frame for card detection and classification
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            Processed frame with annotations
        """
        # Detect cards in frame
        cards = self.detector.detect_cards(frame)
        
        # Draw results on frame
        result_frame = frame.copy()
        
        for card_img, contour in cards:
            # Classify the card
            rank, suit = self.classifier.classify_card(card_img)
            
            # Draw contour
            cv2.drawContours(result_frame, [contour], -1, (0, 255, 0), 3)
            
            # Get contour center for text placement
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw card label
                label = f"{rank} {suit}"
                cv2.putText(result_frame, label, (cx - 50, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display card count
        card_count_text = f"Cards detected: {len(cards)}"
        cv2.putText(result_frame, card_count_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_frame
    
    def run(self, camera_index=0):
        """
        Run real-time card recognition
        
        Args:
            camera_index: Camera device index (default 0)
        """
        # Initialize webcam
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting real-time card recognition...")
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        
        frame_count = 0
        
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame")
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display frame
            cv2.imshow('Playing Card Recognition', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"card_detection_{frame_count}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"Saved frame as {filename}")
                frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function to run real-time card recognition"""
    recognition = RealtimeCardRecognition()
    recognition.run()


if __name__ == "__main__":
    main()
