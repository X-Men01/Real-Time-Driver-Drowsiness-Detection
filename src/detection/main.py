import cv2
import sys

from camera_module import CameraModule
from face_detection import FaceDetection
from feature_extraction import FeatureExtraction
from state_classification import StateClassification
from decision_logic import DecisionLogic
from alarm_system import AlarmSystem
from config import Config
from visualization.plot_results import plot_pipeline, display_frame


def main():
    
    config = Config()
    config.validate()
    
    
    camera = CameraModule(config)
    face_detector = FaceDetection(config)
    feature_extractor = FeatureExtraction(config)
    
    state_classifier = StateClassification(config)
    decision_logic = DecisionLogic(config)
    alarm_system = AlarmSystem(config.ALARM_FILE)
    
    cv2.namedWindow('Driver Monitoring', cv2.WINDOW_NORMAL)
    
    
    try:
        while True:
            success, frame = camera.capture_frame() #! /Users/ahmedalkhulayfi/Downloads/FaceImages/drowsy/image__12963 (2).jpg
            # frame = cv2.cvtColor(cv2.imread("/Users/ahmedalkhulayfi/Downloads/Roboflow_dataset/drowsy/381_jpg.rf.9af6e1fbd62609cae3b658b6e6cf7818.jpg"), cv2.COLOR_BGR2RGB)
            
           
            
            # Face Detection
            face_result = face_detector.detect_face(frame)
            
          
            if face_result.success:
                
                facial_features = feature_extractor.process_face(face_result)
                states = state_classifier.process_features(facial_features)
                decision = decision_logic.determine_drowsiness(states)
                
               
                
                if decision.success and decision.is_drowsy:
                    # alarm_system.trigger_alarm()
                    print("Drowsiness detected!")
                    
                output_frame = display_frame(
                    frame, 
                    face_result,
                    facial_features,
                    states,
                    decision
                )
                
                cv2.imshow('Driver Monitoring', output_frame)
            # sys.exit()
            # Check for exit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break     
                     
                    
                # plot_pipeline(frame, face_result, facial_features, states, decision)
                
                        
            # sys.exit()

    except KeyboardInterrupt:
        print("Stopping system...")
    finally:
        # camera.release()
        pass
        
        

if __name__ == "__main__":
    main()
