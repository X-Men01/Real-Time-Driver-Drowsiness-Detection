import cv2
import mediapipe as mp
from typing import NamedTuple, Optional, Tuple
import numpy as np
from config import Config


class FaceRegion(NamedTuple):
    """Container for face detection results"""

    face: Optional[np.ndarray]
    confidence: float
    success: bool


class FaceDetection:

    def __init__(self, config: Config) -> None:

        self.face_detector = mp.solutions.face_detection.FaceDetection(config.FACE_DETECTION_CONFIDENCE, config.FACE_DETECTION_MODEL_SELECTION)
        self.padding_percent = config.FACE_PADDING_PERCENT

    def detect_face(self, frame: np.ndarray) -> FaceRegion:

        try:

            results = self.face_detector.process(frame)

            if not results.detections:
                return FaceRegion(None, 0.0, False)

            # Get the first detected face
            detection = results.detections[0]
            confidence = detection.score[0]

            # Extract face region
            face_image = self._extract_face_region(frame, detection, self.padding_percent)

            return FaceRegion(face=face_image, confidence=confidence, success=face_image is not None)

        except Exception as e:
            print(f"\033[91mError detecting face: {str(e)}\033[0m")
            return FaceRegion(None, 0.0, False)

    def _extract_face_region(self, frame, detection, padding_percent=10.0):

        try:

            bbox = detection.location_data.relative_bounding_box
            frame_height, frame_width = frame.shape[:2]

            # Calculate padding in pixels
            pad_x = int((bbox.width * frame_width) * (padding_percent / 100))
            pad_y = int((bbox.height * frame_height) * (padding_percent / 100))

           
            # Calculate bbox coordinates with padding and bounds checking
            x = int(max(0, (bbox.xmin * frame_width) - pad_x))
            y = int(max(0, (bbox.ymin * frame_height) - pad_y))
            width = int(min(frame_width - x, (bbox.width * frame_width) + (2 * pad_x)))
            height = int(min(frame_height - y, (bbox.height * frame_height) + (2 * pad_y)))

            # Validate bbox dimensions
            if width <= 0 or height <= 0:
                return None

            # Extract face region
            face_image = frame[y : y + height, x : x + width]
           
            return face_image if face_image.size > 0 else None

        except Exception as e:
            print(f"\033[91mError extracting face region: {str(e)}\033[0m")
            return None
