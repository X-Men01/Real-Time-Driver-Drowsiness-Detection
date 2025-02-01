import torch
import cv2
from torchvision import transforms
from models.cnn_model import Custom_CNN, Custom_CNN_V1
from pathlib import Path
from typing import Union, Optional, NamedTuple, Tuple
import numpy as np
from detection.feature_extraction import FacialFeatures
from detection.head_pose_estimator import HeadPoseEstimator
from config import Config

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
        self.eye_model =  self._initialize_model(config.EYE_MODEL_PATH, input_shape=3, hidden_units=10, output_shape=2)
        self.mouth_model = self._initialize_model(config.MOUTH_MODEL_PATH, input_shape=3, hidden_units=12, output_shape=2)
        self.head_pose_estimator = HeadPoseEstimator((config.FRAME_WIDTH, config.FRAME_HEIGHT))

        
      
        self.transform = transforms.Compose([
           
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
            left_eye_state, conf_left = self._classify_image(features.left_eye, self.eye_model, default_prediction=1 )
            
            # Classify right eye (default to open=1)  #! make sure this what we want
            right_eye_state, conf_right = self._classify_image(features.right_eye, self.eye_model, default_prediction=1 )
            
            # Classify mouth (default to not yawning=0)  #! make sure this what we want
            mouth_state, conf_mouth = self._classify_image(features.mouth, self.mouth_model, default_prediction=0)
            
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
            model = Custom_CNN(input_shape=input_shape, hidden_units=hidden_units, output_shape=output_shape).to(self.device)
            # if hidden_units == 12:
            #     model = Custom_CNN_V1().to(self.device)
            checkpoint = torch.load(model_path,map_location=self.device,weights_only=True)
            
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

        
        
    def _classify_image(self, image: Optional[np.ndarray],model: Custom_CNN, default_prediction: int = 0) -> tuple[int, float]:
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
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.inference_mode():
                output = model(tensor)
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(output).item()
                confidence = probabilities[0][prediction].item() # classify_eye tensor([[0.0203, 0.9558]])
               
                return prediction, confidence
        except Exception as e:
            print(f"Error in classification: {str(e)}")
            return default_prediction, 0.0