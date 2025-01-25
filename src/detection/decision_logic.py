from typing import NamedTuple
from detection.state_classification import StateResults

class DecisionResult(NamedTuple):
    """Container for drowsiness decision results"""
    is_drowsy: bool
    eye_status: bool  # True if eyes indicate drowsiness
    yawn_status: bool  # True if yawning detected
    confidence: float  # Overall confidence in the decision
    success: bool
    
    
class DecisionLogic:
    
    EYES_CLOSED = 0
    EYES_OPEN = 1
    YAWNING = 1
    NOT_YAWNING = 0
    
    def __init__(self, eye_threshold=0.5, yawn_threshold=0.5, min_confidence=0.5):
        
        if not all(0 <= threshold <= 1 for threshold in [eye_threshold, yawn_threshold, min_confidence]):
            raise ValueError("All thresholds must be between 0 and 1")

        self.eye_threshold = eye_threshold
        self.yawn_threshold = yawn_threshold
        self.min_confidence = min_confidence

    def determine_drowsiness(self, states: StateResults) -> DecisionResult:
       
        if not states.success:
            return DecisionResult(False, False, False, 0.0, False)

        try:
            # Check confidence levels
            if not self._check_confidence(states):
                return DecisionResult(False, False, False, 0.0, False)

            # Analyze eye states
            eyes_indicate_drowsy = self._check_eye_state(states.left_eye_state,states.right_eye_state,states.confidence_left,states.confidence_right)

            # Analyze yawn state
            is_yawning = self._check_yawn_state(states.mouth_state, states.confidence_mouth)

            # Calculate overall confidence
            confidence = self._calculate_confidence(states.confidence_left,states.confidence_right,states.confidence_mouth)

            # Make final decision
            is_drowsy = eyes_indicate_drowsy or is_yawning

            return DecisionResult( is_drowsy=is_drowsy, eye_status=eyes_indicate_drowsy, yawn_status=is_yawning, confidence=confidence, success=True)

        except Exception as e:
            print(f"Error in decision logic: {str(e)}")
            return DecisionResult(False, False, False, 0.0, False)
        
        
    def _check_confidence(self, states: StateResults) -> bool:
        return all(conf >= self.min_confidence
            for conf in [states.confidence_left, states.confidence_right, states.confidence_mouth])
        

    def _check_eye_state(self, left_eye: int, right_eye: int, left_conf: float, right_conf: float) -> bool:
        
        if left_conf >= self.eye_threshold and right_conf >= self.eye_threshold:
            return left_eye == self.EYES_CLOSED and right_eye == self.EYES_CLOSED
        return False
    
    
    def _check_yawn_state(self, mouth_state: int, mouth_conf: float) -> bool:
        return mouth_state == self.YAWNING and mouth_conf >= self.yawn_threshold

    def _calculate_confidence(self, left_conf: float, right_conf: float, mouth_conf: float) -> float:
        return (left_conf + right_conf + mouth_conf) / 3.0