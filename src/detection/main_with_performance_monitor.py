import os
import cv2

from camera_module import CameraModule
from face_detection import FaceDetection
from feature_extraction import FeatureExtraction
from state_classification import StateClassification
from decision_logic import DecisionLogic
from alarm_system import AlarmSystem
from config import Config
from performance_monitor import PerformanceMonitor
from visualization.plot_results import plot_pipeline
import sys

def main():
    # Load and validate configuration
    config = Config()
    config.validate()
   
    # Initialize modules
    camera = CameraModule(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    face_detector = FaceDetection(config.FACE_DETECTION_CONFIDENCE, config.FACE_DETECTION_MODEL_SELECTION)
    feature_extractor = FeatureExtraction(config.STATIC_IMAGE_MODE, config.MAX_NUM_FACES, config.REFINE_LANDMARKS, config.MIN_DETECTION_CONF, config.MIN_TRACKING_CONF)
    
    state_classifier = StateClassification(config.EYE_MODEL_PATH, config.MOUTH_MODEL_PATH, config.DEVICE)
    decision_logic = DecisionLogic(config.EYE_CONFIDENCE_THRESHOLD, config.MOUTH_CONFIDENCE_THRESHOLD, config.MIN_CONFIDENCE)
    alarm_system = AlarmSystem()
    monitor = PerformanceMonitor(config.PERFORMANCE_WINDOW_SIZE)
    
    
    try:
        while True:
            # Start frame timing
            monitor.start_frame()
            
            # Capture frame
            start_time = monitor.start_component('camera')
            frame_result = camera.capture_frame()
            monitor.stop_component('camera', start_time)
            
            if frame_result is None:
                continue
            
            # Detect face
            start_time = monitor.start_component('face_detection')
            face_result = face_detector.detect_face(frame_result)
            monitor.stop_component('face_detection', start_time)
            
            if face_result.success:
                # Process features
                start_time = monitor.start_component('feature_extraction')
                features = feature_extractor.process_face(face_result.face)
                monitor.stop_component('feature_extraction', start_time)
                
                if features.success:
                    # Classify states
                    start_time = monitor.start_component('classification')
                    states = state_classifier.process_features(features)
                    monitor.stop_component('classification', start_time)
                    
                    if states.success:
                        # Make decision
                        start_time = monitor.start_component('decision')
                        decision = decision_logic.determine_drowsiness(states)
                        monitor.stop_component('decision', start_time)
            
            # Print performance metrics every 30 frames
            if len(monitor.frame_times) % 30 == 0:
                monitor.print_metrics()
                
    except KeyboardInterrupt:
        print("\nStopping system...")
    finally:
        camera.release()
        

if __name__ == "__main__":
    main()
