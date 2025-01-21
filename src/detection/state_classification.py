import torch
import cv2
import sys
import os
from torchvision import transforms
from models.cnn_model import Custom_CNN


class StateClassification:
    def __init__(self, eye_model_path, mouth_model_path):
        # Initialize models with the same parameters used during training
        self.eye_model = Custom_CNN(
            input_shape=1,  # Grayscale input
            hidden_units=10,  # Adjust based on your training parameters
            output_shape=2   # Binary classification (open/closed)
        )
        self.mouth_model = Custom_CNN(
            input_shape=1,
            hidden_units=10,
            output_shape=2   # Binary classification (yawning/not yawning)
        )
        
        # Load state dictionaries
        self.eye_model.load_state_dict(torch.load(eye_model_path, map_location=torch.device('cpu'),  weights_only=True))
        self.mouth_model.load_state_dict(torch.load(mouth_model_path, map_location=torch.device('cpu'),  weights_only=True))
        
        # Set models to evaluation mode
        self.eye_model.eval()
        self.mouth_model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def classify_eye(self, eye_image):
        # Convert image to grayscale if it's RGB
        if len(eye_image.shape) == 3:
            eye_image = cv2.cvtColor(eye_image, cv2.COLOR_RGB2GRAY)
            
        # Resize image to match model's expected input size (224x224 for example)
        eye_image = cv2.resize(eye_image, (128, 128))  # Adjust size according to your model's training
        
        with torch.no_grad():  # Disable gradient calculation
            # Add channel dimension for grayscale image
            image_tensor = self.transform(eye_image).unsqueeze(0)  # Add batch dimension
            output = self.eye_model(image_tensor)
            return torch.argmax(output).item()  # 0: Open, 1: Closed

    def classify_mouth(self, mouth_image):
        # Convert image to grayscale if it's RGB
        if len(mouth_image.shape) == 3:
            mouth_image = cv2.cvtColor(mouth_image, cv2.COLOR_RGB2GRAY)
            
        # Resize image to match model's expected input size
        mouth_image = cv2.resize(mouth_image, (128, 128))  # Adjust size according to your model's training
        
        with torch.no_grad():  # Disable gradient calculation
            # Add channel dimension for grayscale image
            image_tensor = self.transform(mouth_image).unsqueeze(0)  # Add batch dimension
            output = self.mouth_model(image_tensor)
            return torch.argmax(output).item()  # 0: Not yawning, 1: Yawning