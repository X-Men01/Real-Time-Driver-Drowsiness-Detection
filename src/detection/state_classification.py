import torch
import cv2
from torchvision import transforms
from models.cnn_model import Custom_CNN, Custom_CNN_V1
from pathlib import Path
from typing import Union, Optional, NamedTuple, Tuple
import numpy as np
from detection.feature_extraction import FacialFeatures
from detection.head_pose_estimator import HeadPoseEstimator
from detection.config import Config
import torchvision
import torch.nn as nn
class StateResults(NamedTuple):
    """
    Container for all classification results
    class_names_eye = {0: "Close_Eye", 1: "Open_Eye"}
    class_names_mouth = {0: "No_Yawn", 1: "Yawn"}
    """
    left_eye_state: int
    right_eye_state: int
    mouth_state: int
    confidence_left: float
    confidence_right: float
    confidence_mouth: float
    head_pose: Optional[Tuple[float, float, float]]  # (pitch, yaw, roll)
    success: bool
    
    
class StateClassification:
    INPUT_SIZE = (128, 128)
    
    DEFAULT_STATES = {
        'eye_open': (1, 0.0),    # (state, confidence)
        'eye_closed': (0, 0.0),
        'mouth_normal': (0, 0.0)
    }
    
    def __init__(self, config: Config) -> None:
        
        
        self.device = torch.device(config.DEVICE)
        
        #! i think the input shape and hidden units should be set in training phase and built in the model class
        self.eye_model =  self._initialize_model(config.EYE_MODEL_PATH, input_shape=1, hidden_units=16, output_shape=2)
        self.mouth_model = self._initialize_model(config.MOUTH_MODEL_PATH, input_shape=3, hidden_units=12, output_shape=2)
        self.head_pose_estimator = HeadPoseEstimator((config.FRAME_WIDTH, config.FRAME_HEIGHT))

        
      
        self.eyes_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(128, 128)),
            transforms.Grayscale(),            # No need to specify num_output_channels
            
            transforms.Normalize([0.5], [0.5])
          
         
          ])
        self.mouth_transform = transforms.Compose([
          
            transforms.ToTensor(),
            transforms.Resize(size=self.INPUT_SIZE),
          
         
          ])
    
    
    def process_features(self, features: FacialFeatures) -> StateResults:
        """Process all facial features at once.
        
        Args:
            features: FacialFeatures containing eye and mouth regions
            
        Returns:
            StateResults containing all classification results
        """
        if not features.success:
            #! make sure this what we want
            return StateResults( *self.DEFAULT_STATES['eye_open'],*self.DEFAULT_STATES['eye_open'],*self.DEFAULT_STATES['mouth_normal'],None, False)
            
        try:
            # Classify left eye (default to open=1)  #! make sure this what we want
            left_eye_state, conf_left = self._classify_image(features.left_eye, self.eye_model, default_prediction=1,transform=self.eyes_transform)
            
            # Classify right eye (default to open=1)  #! make sure this what we want
            right_eye_state, conf_right = self._classify_image(features.right_eye, self.eye_model, default_prediction=1,transform=self.eyes_transform)
            
            # Classify mouth (default to not yawning=0)  #! make sure this what we want
            mouth_state, conf_mouth = self._classify_image(features.mouth, self.mouth_model, default_prediction=0,transform=self.mouth_transform)
            
            # Add head pose estimation
            head_pose = self.head_pose_estimator.estimate_pose(features.head_pose_landmarks)
            
            return StateResults(
                left_eye_state=left_eye_state,
                right_eye_state=right_eye_state,
                mouth_state=mouth_state,
                confidence_left=conf_left,
                confidence_right=conf_right,
                confidence_mouth=conf_mouth,
                head_pose=head_pose,
                success=True
            )
            
        except Exception as e:
            print(f"\033[95mError processing features: {str(e)}\033[0m")
            return StateResults(1, 1, 0, 0.0, 0.0, 0.0, None, False)
        
        
    def _initialize_model(self, model_path: Union[str, Path], input_shape: int, hidden_units: int, output_shape: int ) -> Custom_CNN:
        
        
        try:
            #! why not put the input shape and hidden units in the model class?
          
            if hidden_units == 12:
                model = Custom_CNN_V1().to(self.device)
                checkpoint = torch.load(model_path,map_location=self.device,weights_only=True)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
            else: 
                model = Custom_CNN(input_shape, hidden_units, output_shape).to(self.device)
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        
                # Clean up the state dict if it contains "_orig_mod" prefix
                state_dict = checkpoint["model_state_dict"]
                if any("_orig_mod." in key for key in state_dict.keys()):
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        new_key = key.replace("_orig_mod.", "")
                        new_state_dict[new_key] = value
                    state_dict = new_state_dict
                
                # Load the cleaned state dict
                model.load_state_dict(state_dict, strict=True)  # Now we can use strict=True
                model.eval()
            
           
            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

        
        
    def _classify_image(self, image: Optional[np.ndarray],model: Custom_CNN, default_prediction: int = 0,transform: transforms.Compose = None) -> tuple[int, float]:
        """Classify an image using the specified model.

        Args:
            image: Input image array
            model: Model to use for classification
            default_prediction: Default prediction value on error

        Returns:
            Tuple of (prediction, confidence)
        """
        if image is None or image.size == 0:
            return default_prediction, 0.0
        


        try:
            tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.inference_mode():
                output = model(tensor)
                
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(output).item()
                confidence = probabilities[0][prediction].item() # classify_eye tensor([[0.0203, 0.9558]])
               
                return prediction, confidence
        except Exception as e:
            print(f"Error in classification: {str(e)}")
            return default_prediction, 0.0