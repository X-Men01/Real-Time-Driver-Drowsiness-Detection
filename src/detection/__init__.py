"""
Detection package for drowsiness detection system.

This package provides modules for real-time drowsiness detection using computer vision
and machine learning techniques.

Main Components:
- Camera handling
- Face detection
- Feature extraction
- State classification
- Decision logic
- Alarm system
"""

from .camera_module import CameraModule
from .face_detection import FaceDetection, FaceRegion
from .feature_extraction import FeatureExtraction, FacialFeatures
from .state_classification import StateClassification, StateResults
from .decision_logic import DecisionLogic, DecisionResult
from .alarm_system import AlarmSystem
from .config import Config

__version__ = "1.0.0"

__all__ = [
    "CameraModule",
    "FaceDetection",
    "FaceRegion",
    "FeatureExtraction", 
    "FacialFeatures",
    "StateClassification",
    "StateResults",
    "DecisionLogic",
    "DecisionResult",
    "AlarmSystem",
    "Config"
]
