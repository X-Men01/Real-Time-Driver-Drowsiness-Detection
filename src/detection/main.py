import os
import cv2

from camera_module import CameraModule
from face_detection import FaceDetection
from feature_extraction import FeatureExtraction
from state_classification import StateClassification
from decision_logic import DecisionLogic
from alarm_system import AlarmSystem
from config import Config
from visualization.plot_results import plot_pipeline
import sys

def main():
    
    config = Config()
    config.validate()
    
    
    # camera = CameraModule(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    face_detector = FaceDetection(config.FACE_DETECTION_CONFIDENCE, config.FACE_DETECTION_MODEL_SELECTION, config.FACE_PADDING_PERCENT)
    feature_extractor = FeatureExtraction(config.STATIC_IMAGE_MODE, config.MAX_NUM_FACES, config.REFINE_LANDMARKS, config.MIN_DETECTION_CONF, config.MIN_TRACKING_CONF)
    
    state_classifier = StateClassification(config.EYE_MODEL_PATH, config.MOUTH_MODEL_PATH, config.DEVICE)
    decision_logic = DecisionLogic(config.EYE_CONFIDENCE_THRESHOLD, config.MOUTH_CONFIDENCE_THRESHOLD, config.MIN_CONFIDENCE)
    alarm_system = AlarmSystem(config.ALARM_FILE)
    
    
    
    try:
        while True:
            # frame = camera.capture_frame()
            frame = cv2.cvtColor(cv2.imread("/Users/ahmedalkhulayfi/Downloads/Data_to_test/drowsy/001_glasses_sleepyCombination_607_drowsy.jpg"), cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))
            # Face Detection
            face_result = face_detector.detect_face(frame)
            
            if face_result.success:
                
                facial_features = feature_extractor.process_face(face_result)
                states = state_classifier.process_features(facial_features)
                decision = decision_logic.determine_drowsiness(states)
                
                if decision.success and decision.is_drowsy:
                    # alarm_system.trigger_alarm()
                    print("Drowsiness detected!")
                     
                    
                plot_pipeline(frame, face_result, facial_features, states, decision)
                
                        
                

    except KeyboardInterrupt:
        print("Stopping system...")
    finally:
        # camera.release()
        pass
        

if __name__ == "__main__":
    main()
