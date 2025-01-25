import cv2
import mediapipe as mp
from typing import NamedTuple, Optional, Tuple
import numpy as np

class FaceRegion(NamedTuple):
    """Container for face detection results"""
    face: Optional[np.ndarray] 
    confidence: float
    success: bool
    
class FaceDetection:
    def  __init__(self,min_detection_confidence: float = 0.5,model_selection: int = 0) -> None:
        
        if not 0 <= min_detection_confidence <= 1:
            raise ValueError("Detection confidence must be between 0 and 1")
        
        self.face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence,model_selection)

    def detect_face(self, frame: np.ndarray) -> FaceRegion:
        
        try:
          
            results = self.face_detector.process(frame)

            if not results.detections:
                print("No face detected")
                return FaceRegion(None, 0.0, False)

            # Get the first detected face
            detection = results.detections[0]
            confidence = detection.score[0]
            
            # Extract face region
            face_image = self._extract_face_region(frame, detection)
            
            if face_image is None:
                return FaceRegion(None, confidence, False)

            return FaceRegion(face=face_image, confidence=confidence, success=True)

        except Exception as e:
            print(f"Error detecting face: {str(e)}")
            return FaceRegion(None,  0.0, False)
        
        
    def _extract_face_region( self, frame ,detection,padding_percent= 10.0):
  
        try:
            bboxC = detection.location_data.relative_bounding_box
            frame_height, frame_width = frame.shape[:2]

            # Calculate padding in pixels
            pad_x = int((bboxC.width * frame_width) * (padding_percent / 100))
            pad_y = int((bboxC.height * frame_height) * (padding_percent / 100))

            # Calculate bbox coordinates with padding and bounds checking
            x = int(max(0, (bboxC.xmin * frame_width) - pad_x))
            y = int(max(0, (bboxC.ymin * frame_height) - pad_y))
            width = int(min(frame_width - x, (bboxC.width * frame_width) + (2 * pad_x)))
            height = int(min(frame_height - y, (bboxC.height * frame_height) + (2 * pad_y)))
            # Validate bbox dimensions
            if width <= 0 or height <= 0:
                return None

            # Extract face region
            face_image = frame[y:y + height, x:x + width]
            
            # Validate extracted region
            if face_image is None or face_image.size == 0:
                return None

            return face_image

        except Exception as e:
            print(f"Error extracting face region: {str(e)}")
            return None