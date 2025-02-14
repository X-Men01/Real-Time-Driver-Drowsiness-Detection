import cv2

from camera_module import CameraModule
from face_detection import FaceDetection
from feature_extraction import FeatureExtraction
from state_classification import StateClassification
from decision_logic import DecisionLogic
from alarm_system import AlarmSystem
from config import Config
from visualization.plot_results import display_frame
from tracker import Tracker
from facial_measurement import FacialMeasurements
from calibration import CalibrationPhase


def main():

    config = Config()
    config.validate()
    camera = CameraModule(config)
    face_detector = FaceDetection(config)
    feature_extractor = FeatureExtraction(config)

    state_classifier = StateClassification(config)
    decision_logic = DecisionLogic(config)
    alarm_system = AlarmSystem(config)

    tracker = Tracker(config)
    facial_measurements = FacialMeasurements(config)
    
    
      # Run the calibration phase using CNN confidence outputs
    calibration_phase = CalibrationPhase(
        config, camera, face_detector, feature_extractor, facial_measurements, state_classifier
    )
    calibration_info = calibration_phase.run(required_frame_count=10)  # each stage runs for 5 seconds

    cv2.namedWindow("Driver Monitoring", cv2.WINDOW_NORMAL)

    try:
        while True:
            
            success, frame = camera.capture_frame()
            
            if not success:
                continue

            face_result = face_detector.detect_face(frame) 
            
            
            if face_result.success:

                facial_features = feature_extractor.process_face(face_result)
                metrics = facial_measurements.calculate_metrics(facial_features)
                
                states = state_classifier.process_features(facial_features)
                decision = decision_logic.determine_drowsiness(states)

                tracker.add_decision(decision)

                drowsy_state = tracker.is_drowsy()
                drowsy_conf = tracker.aggregated_confidence()

                if drowsy_state:
                    alarm_system.trigger_alarm("Drowsiness")
                    print("\033[95mDrowsiness detected!\033[0m")

                if tracker.is_head_pose_alert():
                    alarm_system.trigger_alarm("Distraction")
                    print("\033[33mDistraction alert!\033[0m")
                    
                    
                alarm_active = alarm_system.is_active
                output_frame = display_frame(frame,face_result,facial_features,states,decision,alarm_active,drowsy_conf,True ,metrics)

            else:
                alarm_system.trigger_alarm("Face_not_detected")
                output_frame = display_frame(frame,face_result,facial_features,states,decision,alarm_active,drowsy_conf,False,metrics)
            
            cv2.imshow("Driver Monitoring", output_frame)
           
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

               
    except KeyboardInterrupt:
        print("Stopping system...")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()