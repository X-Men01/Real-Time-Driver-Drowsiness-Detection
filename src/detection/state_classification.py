import torch
import cv2
from torchvision import transforms

class StateClassification:
    def __init__(self, eye_model_path, mouth_model_path):
        self.eye_model = torch.load(eye_model_path)
        self.mouth_model = torch.load(mouth_model_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def classify_eye(self, eye_image):
        image_tensor = self.transform(eye_image).unsqueeze(0)
        output = self.eye_model(image_tensor)
        return torch.argmax(output).item()  # 0: Open, 1: Closed

    def classify_mouth(self, mouth_image):
        image_tensor = self.transform(mouth_image).unsqueeze(0)
        output = self.mouth_model(image_tensor)
        return torch.argmax(output).item()  # 0: Not yawning, 1: Yawning
