import cv2
import time
import numpy as np
from detection.config import Config
from detection.camera_module import CameraModule
from detection.face_detection import FaceDetection
from detection.feature_extraction import FeatureExtraction
from detection.facial_measurement import FacialMeasurements
from detection.state_classification import StateClassification

class CalibrationPhase:
    def __init__(self, config: Config, camera: CameraModule, face_detector: FaceDetection, 
                 feature_extractor: FeatureExtraction, facial_measurements: FacialMeasurements,
                 state_classifier: StateClassification):
        self.config = config
        self.camera = camera
        self.face_detector = face_detector
        self.feature_extractor = feature_extractor
        self.facial_measurements = facial_measurements
        self.state_classifier = state_classifier

    def run(self, required_frame_count: int = 10):
       
        # Stage 1: Mouth Open Calibration
        mouth_instruction = "Calibration: Please open your mouth widely"
        avg_mouth_open = self._run_calibration_step(
            instruction=mouth_instruction,
            required_frame_count=required_frame_count,
            target_state="mouth_open"
        )
        
      
        
        # Stage 2: Eyes Closed Calibration
        eyes_closed_instruction = "Calibration: Please close your eyes"
        avg_eyes_closed = self._run_calibration_step(
            instruction=eyes_closed_instruction,
            required_frame_count=required_frame_count,
            target_state="eyes_closed"
        )
        
        
         # Stage 3: Eyes Open Calibration
        eyes_open_instruction = "Calibration: Please keep your eyes open"
        avg_eyes_open = self._run_calibration_step(
            instruction=eyes_open_instruction,
            required_frame_count=required_frame_count,
            target_state="eyes_open"
        )
        
        
        
        # Stage 4: Mouth Closed Calibration
        mouth_closed_instruction = "Calibration: Please close your mouth"
        avg_mouth_closed = self._run_calibration_step(
            instruction=mouth_closed_instruction,
            required_frame_count=required_frame_count,
            target_state="mouth_closed"
        )

        # Compute new thresholds based on calibration results.
        new_eyes_threshold = ((avg_eyes_open + avg_eyes_closed) / 2 ) * 0.9
        new_mouth_threshold = ((avg_mouth_open + avg_mouth_closed) / 2 ) * 0.9

        # Dynamically update the hyperparameters in the config.
        self.config.EYE_CONFIDENCE_THRESHOLD = new_eyes_threshold
        self.config.MOUTH_CONFIDENCE_THRESHOLD = new_mouth_threshold 

        self.config.MIN_CONFIDENCE = (new_eyes_threshold + new_mouth_threshold) / 2
        
        final_frame = self._get_frame_with_text(f"Calibration complete. New Eye Threshold: {new_eyes_threshold:.3f}, New Mouth Threshold: {new_mouth_threshold:.3f} \nPress any key to continue...")
      
        final_frame = cv2.putText(final_frame, "Press any key to continue...", (30, 60*2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0 ), 2)
        cv2.imshow("Calibration", final_frame)
        cv2.waitKey(0)
        cv2.destroyWindow("Calibration")

        return {
            "new_eyes_threshold": new_eyes_threshold,
            "new_mouth_threshold": new_mouth_threshold
        }
    
    def _run_calibration_step(self, instruction: str, required_frame_count: int, target_state: str):
        

        while True:
            collected_confidences = []
           
            
           
            while len(collected_confidences) < required_frame_count:
                success, frame = self.camera.capture_frame()
                if not success:
                    continue

                
                progress_text = f"{instruction}. Captured: {len(collected_confidences)}/{required_frame_count}"
                display_frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),1)
                display_frame = self._get_frame_with_text(progress_text, display_frame)
                cv2.imshow("Calibration", display_frame)
                cv2.waitKey(1)

               
                face_result = self.face_detector.detect_face(frame)
                if not face_result.success:
                    continue

                features = self.feature_extractor.process_face(face_result)
                if not features.success:
                    continue

                state_results = self.state_classifier.process_features(features)
                if not state_results.success:
                    continue

                if target_state in ("eyes_open", "eyes_closed"):
                    desired_state = 1 if target_state == "eyes_open" else 0
                    if (state_results.left_eye_state == desired_state and state_results.right_eye_state == desired_state):
                        confidence = (state_results.confidence_left + state_results.confidence_right) / 2.0
                        collected_confidences.append(confidence)
                    
                elif target_state in ("mouth_open", "mouth_closed"):
                   
                    desired_state = 1 if target_state == "mouth_open" else 0
                    if state_results.mouth_state == desired_state:
                        collected_confidences.append(state_results.confidence_mouth)

            avg_confidence = np.mean(collected_confidences)
            print(f"\033[96m{instruction} - Collected {required_frame_count} frames. Average confidence for {target_state}: {avg_confidence:.3f}\033[00m")

            
            if avg_confidence:
                print(f"\033[92m Calibration for '{target_state}' successful with average confidence {avg_confidence:.3f}.\033[00m")
                return avg_confidence
            else:
                warning_message = f"\033[91m'{target_state}' calibration not successful. Please follow the instructions precisely and try again.\033[00m"
                warning_frame = self._get_frame_with_text(warning_message)
                cv2.imshow("Calibration", warning_frame)
                cv2.waitKey(2000)  # Pause for 2 seconds before retrying.
                print(f"Retrying calibration for '{target_state}' ...")

    def _get_frame_with_text(self, text: str, frame: np.ndarray = None):
        
        if frame is None:
            
            frame = 255 * np.ones((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame 