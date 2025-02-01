import cv2
import numpy as np
import sys
from typing import Tuple, List, Optional

# https://www.youtube.com/watch?v=-toNMaS4SeQ
# https://medium.com/@susanne.thierfelder/head-pose-estimation-with-mediapipe-and-opencv-in-javascript-c87980df3acb


class HeadPoseEstimator:
    def __init__(self, image_size: Tuple[int, int] = (1280, 720)):
       
        # Use actual image dimensions for focal length calculation
        focal_length = min(image_size[0], image_size[1])
        
        # Camera matrix
        self.camera_matrix = np.array(
            [
                [focal_length, 0, image_size[0] / 2],
                [0, focal_length, image_size[1] / 2],
                [0, 0, 1],
            ],
            dtype=np.float64,)

        # Distortion coefficients
        self.dist_coeffs = np.zeros((4, 1))

    def estimate_pose(self, landmarks: List[Tuple[int, int, float]]) -> Optional[Tuple[float, float, float]]:
        """
        Estimate head pose angles from facial landmarks
        Returns: (pitch, yaw, roll) in degrees
        """
        if not landmarks or len(landmarks) < 6:
            return None

        try:
           

            image_points_3d = np.array(landmarks, dtype=np.float64)

            image_points_2d = image_points_3d[:, :2].astype(np.float64)

            # Solve PnP
            success, rotation_vec, _ = cv2.solvePnP(
                image_points_3d,
                image_points_2d,
                self.camera_matrix,
                self.dist_coeffs,)

            if not success:
                return None

            # Convert rotation vector to rotation matrix
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)

            # Get Euler angles
            angles, _, _,_, _, _ = cv2.RQDecomp3x3(rotation_mat)
            
            return tuple(angle * 360 for angle in angles)

        except Exception as e:
            print(f"Error estimating head pose: {e}")
            return None

   
