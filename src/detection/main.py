import os
import cv2

from camera_module import CameraModule
from face_detection import FaceDetection
from feature_extraction import FeatureExtraction
from state_classification import StateClassification
from decision_logic import DecisionLogic
from alarm_system import AlarmSystem
from visualization.plot_results import plot_pipeline



def main():
   
    # Initialize modules
    camera = CameraModule(camera_index=0, frame_width=224, frame_height=224)
    face_detector = FaceDetection()
    feature_extractor = FeatureExtraction()
    state_classifier = StateClassification("../../src/models/eye_state_model.pth", "../../src/models/mouth_state_model.pth")
    decision_logic = DecisionLogic()
    alarm_system = AlarmSystem()
   
    try:
        while True:
            frame = cv2.imread("/Users/ahmedalkhulayfi/Downloads/images.jpeg")
            # frame = camera.capture_frame()

            # Face Detection
            face = face_detector.detect_face(frame)
            
            if face is not None:
                landmarks = feature_extractor.extract_features(frame)
                
                # Get eye and mouth regions
                left_eye_region = feature_extractor.get_feature_region(frame, landmarks["left_eye_landmarks"])
                right_eye_region = feature_extractor.get_feature_region(frame, landmarks["right_eye_landmarks"])
                mouth_region = feature_extractor.get_feature_region(frame, landmarks["mouth_landmarks"])
                
                if (left_eye_region is not None) and (right_eye_region is not None) and (mouth_region is not None):
                    # Extract and classify eye and mouth states
                    eye_state = state_classifier.classify_eye(left_eye_region)
                    mouth_state = state_classifier.classify_mouth(mouth_region)
                    print("Eye state:", eye_state)
                    print("Mouth state:", mouth_state)

                    # Decision Logic
                    decision = decision_logic.determine_drowsiness(eye_state, mouth_state)
                    if decision:
                        # alarm_system.trigger_alarm()
                        print("Drowsiness detected!")
                    
                    plot_pipeline(frame, face, left_eye_region, right_eye_region, mouth_region, decision)
                        
                break

    except KeyboardInterrupt:
        print("Stopping system...")
    finally:
        pass
        # camera.release()
        

if __name__ == "__main__":
    main()
