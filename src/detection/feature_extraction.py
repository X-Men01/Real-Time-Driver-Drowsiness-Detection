import mediapipe as mp

class FeatureExtraction:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh()

    def extract_features(self, frame):
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            return face_landmarks
        return None
