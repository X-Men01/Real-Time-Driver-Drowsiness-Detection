import numpy as np
from typing import Dict, Tuple, NamedTuple
from detection.feature_extraction import FacialFeatures


class FacialMetrics(NamedTuple):
    """Container for facial measurements"""
    avg_ear: float
    left_ear: float
    right_ear: float
    mar: float


class FacialMeasurements:
    def calculate_metrics(facial_features: FacialFeatures) -> FacialMetrics:
        """Calculate all facial measurements from facial features"""
        left_eye = FacialMeasurements.measure_eye(facial_features.ear_left_landmarks, True)
        right_eye = FacialMeasurements.measure_eye(facial_features.ear_right_landmarks, False)
        avg_eye = (left_eye + right_eye) / 2.0
        mouth = FacialMeasurements.measure_mouth(facial_features.mar_mouth_landmarks)

        return FacialMetrics(
            avg_ear=avg_eye,
            left_ear=left_eye,
            right_ear=right_eye,
            mar=mouth
        )

    def measure_eye(eye_landmarks: Dict[str, Tuple[int, int]], left_eye: bool) -> float:
        """Measure eye openness ratio"""
        if not eye_landmarks or len(eye_landmarks) != 6:
            return -1.0

        try:
            if left_eye:
                landmarks = np.array(list((
                    eye_landmarks.get("362"),
                    eye_landmarks.get("385"),
                    eye_landmarks.get("387"),
                    eye_landmarks.get("263"),
                    eye_landmarks.get("373"),
                    eye_landmarks.get("380"),
                )))
            else:
                landmarks = np.array(list((
                    eye_landmarks.get("33"),
                    eye_landmarks.get("160"),
                    eye_landmarks.get("158"),
                    eye_landmarks.get("133"),
                    eye_landmarks.get("153"),
                    eye_landmarks.get("144"),
                )))

            vert_dist1 = np.linalg.norm(landmarks[1] - landmarks[5])
            vert_dist2 = np.linalg.norm(landmarks[2] - landmarks[4])
            horz_dist = np.linalg.norm(landmarks[0] - landmarks[3])

            return float((vert_dist1 + vert_dist2) / (2.0 * horz_dist))
        except Exception as e:
            print(str(e))
            return -1.0

    def measure_mouth(mouth_landmarks: Dict[str, Tuple[int, int]]) -> float:
        """Measure mouth opening ratio"""
        if not mouth_landmarks or len(mouth_landmarks) != 8:
            return -1.0

        try:
            landmarks = np.array(list((
                mouth_landmarks.get("61"),
                mouth_landmarks.get("291"),
                mouth_landmarks.get("39"),
                mouth_landmarks.get("181"),
                mouth_landmarks.get("0"),
                mouth_landmarks.get("17"),
                mouth_landmarks.get("269"),
                mouth_landmarks.get("405"),
            )))

            vert_dist1 = np.linalg.norm(landmarks[2] - landmarks[3])
            vert_dist2 = np.linalg.norm(landmarks[4] - landmarks[5])
            vert_dist3 = np.linalg.norm(landmarks[6] - landmarks[7])
            horz_dist = np.linalg.norm(landmarks[0] - landmarks[1])

            return float((vert_dist1 + vert_dist2 + vert_dist3) / (2.0 * horz_dist))
        except Exception as e:
            print(str(e))
            return -1.0
