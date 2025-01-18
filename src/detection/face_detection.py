import cv2
import mediapipe as mp

class FaceDetection:
    def __init__(self):
        self.face_detector = mp.solutions.face_detection.FaceDetection()

    def detect_face(self, frame):
        results = self.face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            # Return the first detected face
            return results.detections[0]
        return None
