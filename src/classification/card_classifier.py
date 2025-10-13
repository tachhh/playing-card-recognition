import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CardClassifierNN(nn.Module):
    """Neural network model for card classification."""
    
    def __init__(self, input_size, num_classes):
        """
        Initialize the neural network.
        
        Parameters:
        -----------
        input_size : int
            Size of the input feature vector
        num_classes : int
            Number of card classes (52 for a standard deck)
        """
        super(CardClassifierNN, self).__init__()
        
        # Define the network architecture using linear transformations (matrices)
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """Forward pass through the network."""
        # First layer: Wx + b followed by ReLU activation
        # This is a matrix-vector product followed by element-wise nonlinearity
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Second layer: another matrix-vector product and nonlinearity
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer: final matrix-vector product
        x = self.fc3(x)
        
        # Return softmax probabilities
        return F.softmax(x, dim=1)

class CardClassifier:
    """Class to handle card classification logic."""
    
    def __init__(self):
        """Initialize the classifier."""
        self.model = None
        self.classes = [
            '2C', '2D', '2H', '2S',  # 2 of Clubs, Diamonds, Hearts, Spades
            '3C', '3D', '3H', '3S',
            '4C', '4D', '4H', '4S',
            '5C', '5D', '5H', '5S',
            '6C', '6D', '6H', '6S',
            '7C', '7D', '7H', '7S',
            '8C', '8D', '8H', '8S',
            '9C', '9D', '9H', '9S',
            '10C', '10D', '10H', '10S',
            'JC', 'JD', 'JH', 'JS',  # Jack
            'QC', 'QD', 'QH', 'QS',  # Queen
            'KC', 'KD', 'KH', 'KS',  # King
            'AC', 'AD', 'AH', 'AS'   # Ace
        ]
        
    def load_model(self, model_path):
        """
        Load a trained model from file.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model
        """
        try:
            # Define model architecture
            input_size = 70  # Combined SVD and HOG features
            num_classes = 52  # 52 cards in a standard deck
            
            # Create model instance
            self.model = CardClassifierNN(input_size, num_classes)
            
            # Load weights
            self.model.load_state_dict(torch.load(model_path))
            
            # Set model to evaluation mode
            self.model.eval()
            
            print(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Default to a dummy model for testing
            print("Using dummy model for testing")
            self.model = "dummy"
    
    def predict(self, features):
        """
        Predict card class from features.
        
        Parameters:
        -----------
        features : numpy.ndarray
            Feature vector extracted from card image
            
        Returns:
        --------
        tuple
            (predicted_class, confidence)
        """
        if self.model == "dummy":
            # Return random prediction for testing
            import random
            idx = random.randint(0, len(self.classes) - 1)
            return self.classes[idx], 0.85
        
        # Convert features to PyTorch tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(features_tensor)
            
        # Get the predicted class and confidence
        confidence, predicted = torch.max(outputs, 1)
        
        return self.classes[predicted.item()], confidence.item()
