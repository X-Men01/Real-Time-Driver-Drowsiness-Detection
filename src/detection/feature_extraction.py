import mediapipe as mp
from typing import Optional, NamedTuple
import numpy as np
from detection.config import Config
from detection.face_detection import FaceRegion


class FacialFeatures(NamedTuple):
    """Container for facial feature regions"""

    left_eye: Optional[np.ndarray]
    right_eye: Optional[np.ndarray]
    mouth: Optional[np.ndarray]
    head_pose_landmarks: Optional[np.ndarray]
    success: bool


class FeatureExtraction:
    LEFT_EYE_INDICES = (63, 107, 128, 117)
    RIGHT_EYE_INDICES = (336, 293, 346, 357)
    MOUTH_INDICES = (216, 322, 424, 210, 152)
    HEAD_POSE_LANDMARKS = (33, 263, 1, 61, 291, 199)  # Nose, eyes, ears

    def __init__(self, config: Config):
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
        self.faceMesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=config.STATIC_IMAGE_MODE,
            max_num_faces=config.MAX_NUM_FACES,
            refine_landmarks=config.REFINE_LANDMARKS,
            min_detection_confidence=config.MIN_DETECTION_CONF,
            min_tracking_confidence=config.MIN_TRACKING_CONF,
        )

        self.feature_padding = config.FEATURE_PADDING

    def process_face(self, face_region: FaceRegion) -> FacialFeatures:

        if not face_region.success or face_region.face is None:
            return FacialFeatures(None, None, None, None, False)

        try:
            landmarks = self._extract_features(face_region.face)

            left_eye_region = self.get_feature_region(face_region.face, landmarks["left_eye_landmarks"], self.feature_padding)
            right_eye_region = self.get_feature_region(face_region.face, landmarks["right_eye_landmarks"], self.feature_padding)
            mouth_region = self.get_feature_region(face_region.face, landmarks["mouth_landmarks"], self.feature_padding)

            success = all(
                region is not None
                for region in [left_eye_region, right_eye_region, mouth_region])

            return FacialFeatures(
                left_eye_region,
                right_eye_region,
                mouth_region,
                landmarks["head_pose_landmarks"],
                success,)

        except Exception as e:
            return FacialFeatures(None, None, None, None, False)

    def _extract_features(self, face):

        landmarks = {
            "left_eye_landmarks": [],
            "right_eye_landmarks": [],
            "mouth_landmarks": [],
            "head_pose_landmarks": [],}

        try:
            results = self.faceMesh.process(face)
            if results.multi_face_landmarks:

                face_landmarks = results.multi_face_landmarks[0]
                h, w = face.shape[:2]
                for i, landmark in enumerate(face_landmarks.landmark):

                    x, y = int(landmark.x * w), int(landmark.y * h)  # Convert normalized coordinates to pixel values

                    if i in self.LEFT_EYE_INDICES:
                        landmarks["left_eye_landmarks"].append((x, y))
                    elif i in self.RIGHT_EYE_INDICES:
                        landmarks["right_eye_landmarks"].append((x, y))
                    elif i in self.MOUTH_INDICES:
                        landmarks["mouth_landmarks"].append((x, y))
                    elif i in self.HEAD_POSE_LANDMARKS:
                        landmarks["head_pose_landmarks"].append([x, y, landmark.z])
                        
                return landmarks 
        except Exception as e:
           raise ValueError(f"\033[95mFeature Extraction Error: {e}\033[0m")

        

    def get_feature_region(self, img, landmarks, padding=10):
        
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
