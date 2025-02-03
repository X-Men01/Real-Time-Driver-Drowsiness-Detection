from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import torch
@dataclass(frozen=True)
class Config:
    """
    *Central configuration for drowsiness detection system.
    
    Using frozen=True makes the config immutable after creation.
    """
    # Camera settings
    CAMERA_INDEX: int = 1
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    FPS: int = 30
    
    # Face Detection settings
    FACE_DETECTION_CONFIDENCE: float = 0.5
    FACE_PADDING_PERCENT: float = 20.0
    FACE_DETECTION_MODEL_SELECTION: int = 0
    
    # Feature Extraction settings
    STATIC_IMAGE_MODE: bool = False
    MAX_NUM_FACES: int = 1
    REFINE_LANDMARKS: bool = False
    MIN_DETECTION_CONF: float = 0.5
    MIN_TRACKING_CONF: float = 0.5
    FEATURE_PADDING: int = 10 
    
    # Model settings
    MODEL_DIR: Path = Path("../../src/models")
    EYE_MODEL_PATH: Path = MODEL_DIR / "eye_state_model.pt"
    MOUTH_MODEL_PATH: Path = MODEL_DIR / "mouth_state_model.pt"
    DEVICE: str =  "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Classification thresholds
    EYE_CONFIDENCE_THRESHOLD: float = 0.7
    MOUTH_CONFIDENCE_THRESHOLD: float = 0.6
    MIN_CONFIDENCE: float = 0.5
    HEAD_POSE_THRESHOLD: float = 20.0
    
    
    # Alarm settings
    ALARM_FILE: Path = Path("../../assets/alarm_sound.mp3")
    
    # Temporal analysis settings
    WINDOW_SIZE: int = 20
    DROWSY_THRESHOLD: float = 0.6
    
   
    # Need to add more validation for the thresholds and other parameters
    #! and why not validate everything here? and remove the validation in the other classes?
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not (0 <= self.FACE_DETECTION_CONFIDENCE <= 1):
            raise ValueError("Face detection confidence must be between 0 and 1")
            
        if not self.MODEL_DIR.exists():
            raise ValueError(f"Model directory not found: {self.MODEL_DIR}")
            
        if not self.EYE_MODEL_PATH.exists():
            raise ValueError(f"Eye model not found: {self.EYE_MODEL_PATH}")
            
        if not self.MOUTH_MODEL_PATH.exists():
            raise ValueError(f"Mouth model not found: {self.MOUTH_MODEL_PATH}")
        
        if not (0 <= self.MIN_DETECTION_CONF <= 1) or not (0 <= self.MIN_TRACKING_CONF <= 1):
            raise ValueError("Confidence values must be between 0 and 1")
        if self.MAX_NUM_FACES < 1:
            raise ValueError("max_num_faces must be greater than 0")
    
   
