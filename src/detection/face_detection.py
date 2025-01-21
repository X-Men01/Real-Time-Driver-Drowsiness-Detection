import cv2
import mediapipe as mp

class FaceDetection:
    def __init__(self):
        self.face_detector = mp.solutions.face_detection.FaceDetection()

    def detect_face(self, frame):
        results = self.face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            # Get the first detected face
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
        
            h, w, _ = frame.shape
            x = max(0, bboxC.xmin * w)  # Ensure x is not negative
            y = max(0, bboxC.ymin * h)  # Ensure y is not negative
            width = min(w - x, bboxC.width * w)  # Ensure width does not exceed image bounds
            height = min(h - y, bboxC.height * h)  # Ensure height does not exceed image bounds

            # print("x, y, width, height", x, y, width, height)
            # Crop the face region from the original frame
            face_image = frame[int(y):int(y + height), int(x):int(x + width)]
            return face_image  # Return the cropped face image
        print("No face detected")
        return None  # Return None if no face is detected
    
    
