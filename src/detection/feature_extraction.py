import mediapipe as mp
from typing import Optional, NamedTuple
import numpy as np


class FacialFeatures(NamedTuple):
    """Container for facial feature regions"""

    left_eye: Optional[np.ndarray]
    right_eye: Optional[np.ndarray]
    mouth: Optional[np.ndarray]
    success: bool


class FeatureExtraction:
    LEFT_EYE_INDICES = (63, 107, 128, 117)
    RIGHT_EYE_INDICES = (336, 293, 346, 357)
    MOUTH_INDICES = (216, 322, 424, 210, 200)

    def __init__(self,static_image_mode=False,max_num_faces=1,refine_landmarks=False,min_detection_con=0.5,min_tracking_con=0.5):
        """Initialize the feature extraction with MediaPipe Face Mesh.

        Args:
            static_image_mode: Whether to treat input as static images
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Whether to refine face landmarks
            min_detection_con: Minimum detection confidence threshold
            min_tracking_con: Minimum tracking confidence threshold

        Raises:
            ValueError: If confidence values are invalid or max_num_faces < 1
        """

        if not (0 <= min_detection_con <= 1) or not (0 <= min_tracking_con <= 1):
            raise ValueError("Confidence values must be between 0 and 1")
        if max_num_faces < 1:
            raise ValueError("max_num_faces must be greater than 0")

        self.faceMesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_con,
            min_tracking_confidence=min_tracking_con,
        )
    
    def process_face(self, face: np.ndarray) -> FacialFeatures:
        """Process a face image to extract facial features.

        Args:
            face: Input face image array

        Returns:
            FacialFeatures containing extracted regions and success status
        """ 
        
        if face is None or face.size == 0:
            return FacialFeatures(None, None, None, False)
        try:
            landmarks = self.extract_features(face)
            
            left_eye_region = self.get_feature_region(face, landmarks["left_eye_landmarks"])
            right_eye_region = self.get_feature_region(face, landmarks["right_eye_landmarks"])
            mouth_region = self.get_feature_region(face, landmarks["mouth_landmarks"])
            
            success = all(region is not None for region in [left_eye_region, right_eye_region, mouth_region])
            
            return FacialFeatures(left_eye_region, right_eye_region, mouth_region, success)
        
        except Exception as e:
            print(f"Error processing face: {e}")
            return FacialFeatures(None, None, None, False)

    def extract_features(self, face):
        """Extract facial landmarks from the input image.

        Args:
            face: Input face image array

        Returns:
            Dictionary containing landmark coordinates for facial features

        Raises:
            ValueError: If input image is invalid
        """
        

        if face is None or face.size == 0:
            raise ValueError("Invalid input image (face)")

        landmarks = {
            "left_eye_landmarks": [],
            "right_eye_landmarks": [],
            "mouth_landmarks": [],
        }

        try:
            results = self.faceMesh.process(face)
            if results.multi_face_landmarks:

                face_landmarks = results.multi_face_landmarks[0]
                h, w, ic = face.shape
                for i, lm in enumerate(face_landmarks.landmark):

                    x, y = int(lm.x * w), int(lm.y * h)  # Convert normalized coordinates to pixel values

                    if i in self.LEFT_EYE_INDICES:
                        landmarks["left_eye_landmarks"].append((x, y))
                    if i in self.RIGHT_EYE_INDICES:
                        landmarks["right_eye_landmarks"].append((x, y))
                    if i in self.MOUTH_INDICES:
                        landmarks["mouth_landmarks"].append((x, y))
        except Exception as e:
            print(f"Error processing face: {e}")

        return landmarks

    def get_feature_region(self, img, landmarks, padding=10):
        """Extract a region around specified landmarks.

        Args:
            img: Input image array
            landmarks: List of landmark coordinates
            padding: Padding around the region

        Returns:
            Extracted image region or None if invalid
        """
        if not landmarks or img is None or img.size == 0:
            return None

       
        x_coords, y_coords = zip(*landmarks)

        left = max(0, min(x_coords) - padding)
        right = min(img.shape[1], max(x_coords) + padding)
        top = max(0, min(y_coords) - padding)
        bottom = min(img.shape[0], max(y_coords) + padding)

        # Extract the region
        region = img[top:bottom, left:right]


        return region if region.size > 0 else None
