from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import torch
@dataclass(frozen=True)
class Config:
    """Central configuration for drowsiness detection system.
    
    Using frozen=True makes the config immutable after creation.
    """
    # Camera settings
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    FPS: int = 30
    
    # Face Detection settings
    FACE_DETECTION_CONFIDENCE: float = 0.5
    FACE_PADDING_PERCENT: float = 10.0
    FACE_DETECTION_MODEL_SELECTION: int = 0
    
    # Feature Extraction settings
    STATIC_IMAGE_MODE: bool = False
    MAX_NUM_FACES: int = 1
    REFINE_LANDMARKS: bool = False
    MIN_DETECTION_CONF: float = 0.5
    MIN_TRACKING_CONF: float = 0.5
    
    # Model settings
    MODEL_DIR: Path = Path("../../src/models")
    EYE_MODEL_PATH: Path = MODEL_DIR / "eye_state_model.pt"
    MOUTH_MODEL_PATH: Path = MODEL_DIR / "mouth_state_model.pt"
    DEVICE: str =  "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Classification thresholds
    EYE_CONFIDENCE_THRESHOLD: float = 0.5
    MOUTH_CONFIDENCE_THRESHOLD: float = 0.5
    MIN_CONFIDENCE: float = 0.5
    # Performance monitoring
    PERFORMANCE_WINDOW_SIZE: int = 30  # Number of frames for rolling average
    
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
    
   
