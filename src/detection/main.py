from camera_module import CameraModule
from face_detection import FaceDetection
from feature_extraction import FeatureExtraction
from state_classification import StateClassification
from decision_logic import DecisionLogic
from alarm_system import AlarmSystem

def main():
    # Initialize modules
    camera = CameraModule()
    face_detector = FaceDetection()
    feature_extractor = FeatureExtraction()
    state_classifier = StateClassification("models/eye_state_model.pth", "models/mouth_state_model.pth")
    decision_logic = DecisionLogic()
    alarm_system = AlarmSystem()

    try:
        while True:
            frame = camera.capture_frame()

            # Face Detection
            face = face_detector.detect_face(frame)
            if face:
                landmarks = feature_extractor.extract_features(frame)

                # Extract and classify eye and mouth states
                eye_state = state_classifier.classify_eye(landmarks["eyes"])
                mouth_state = state_classifier.classify_mouth(landmarks["mouth"])

                # Decision Logic
                if decision_logic.determine_drowsiness(eye_state, mouth_state):
                    alarm_system.trigger_alarm()

    except KeyboardInterrupt:
        print("Stopping system...")
    finally:
        camera.release()

if __name__ == "__main__":
    main()
