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
from drowsiness_tracker import DrowsinessTracker

def main():
    
    config = Config()
    config.validate()
    
    
    camera = CameraModule(config)
    face_detector = FaceDetection(config)
    feature_extractor = FeatureExtraction(config)
    
    state_classifier = StateClassification(config)
    decision_logic = DecisionLogic(config)
    alarm_system = AlarmSystem(config)
    
    tracker = DrowsinessTracker(config)
    
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
                
                tracker.add_decision(decision)
                
                aggregated_state = tracker.is_drowsy()
                aggregated_conf = tracker.aggregated_confidence()
                
                if aggregated_state:
                    alarm_system.trigger_alarm()
                    print("\033[95mDrowsiness detected!\033[0m")
               
                    
                output_frame = display_frame(
                    frame, 
                    face_result,
                    facial_features,
                    states,
                    decision,
                    aggregated_state,
                    aggregated_conf)
                
               
                
                
            else:
                output_frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 1)
                cv2.putText(output_frame, "Face not detected", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
               
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
       camera.release()
       cv2.destroyAllWindows()
      
        
        
        

if __name__ == "__main__":
    main()
