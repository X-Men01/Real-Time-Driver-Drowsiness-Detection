from typing import NamedTuple, Optional, Tuple
from detection.state_classification import StateResults
from config import Config
"""

?| **Scenario** | **Left Eye State** | **Right Eye State** | **Mouth State** | **Interpretation**                                   |
?|--------------|---------------------|---------------------|-----------------|---------------------------------------------------|
*| 1            | Open (1)           | Open (1)           | No Yawn (0)     | Driver is alert.                                   |
!| 2            | Open (1)           | Open (1)           | Yawn (1)        | Early sign of drowsiness or fatigue.             |
*| 3            | Closed (0)         | Open (1)           | No Yawn (0)     | Possible blinking or obstruction, needs checks.  |
!| 4            | Closed (0)         | Open (1)           | Yawn (1)        | Potential fatigue, needs monitoring.             |
*| 5            | Open (1)           | Closed (0)         | No Yawn (0)     | Possible blinking or obstruction, needs checks.  |
!| 6            | Open (1)           | Closed (0)         | Yawn (1)        | Potential fatigue, needs monitoring.             |
!| 7            | Closed (0)         | Closed (0)         | No Yawn (0)     | High chance of drowsiness or microsleep.         |
!| 8            | Closed (0)         | Closed (0)         | Yawn (1)        | Strong indicator of drowsiness and imminent risk.|


?Our code identifies RED as DROWSY, GREEN as ALERT.

"""




class DecisionResult(NamedTuple):
    """Container for drowsiness decision results"""
    is_drowsy: bool
    eye_status: bool  # True if eyes indicate drowsiness
    yawn_status: bool  # True if yawning detected
    confidence: float  # Overall confidence in the decision
    head_pose_status: str  
    success: bool
    
    
class DecisionLogic:
    
    EYES_CLOSED = 0
    EYES_OPEN = 1
    YAWNING = 1
    NOT_YAWNING = 0
    
    # Head position states
    HEAD_FORWARD = "Forward"
    HEAD_LEFT = "Looking Left"
    HEAD_RIGHT = "Looking Right"
    HEAD_UP = "Looking Up"
    HEAD_DOWN = "Looking Down"
    
    def __init__(self, config: Config):
        
        self.eye_threshold = config.EYE_CONFIDENCE_THRESHOLD
        self.yawn_threshold = config.MOUTH_CONFIDENCE_THRESHOLD
        self.min_confidence = config.MIN_CONFIDENCE
        self.head_pose_threshold = config.HEAD_POSE_THRESHOLD

    def determine_drowsiness(self, states: StateResults) -> DecisionResult:
        
        if not states.success:
            return self._get_default_result()

        try:
            # Check confidence levels
            if not self._check_confidence(states):
                return self._get_default_result()

            # Analyze eye states
            eyes_indicate_drowsy = self._check_eye_state(states.left_eye_state,states.right_eye_state,states.confidence_left,states.confidence_right)

            # Analyze yawn state
            is_yawning = self._check_yawn_state(states.mouth_state, states.confidence_mouth)
            
            # Head pose check
            head_pose_position = self._get_head_position(states.head_pose)

            # Calculate overall confidence
            confidence = confidence = self._calculate_confidence(
                                        states.confidence_left,
                                        states.confidence_right,
                                        states.confidence_mouth,
                                        eyes_indicate_drowsy,
                                        is_yawning)

            # Make final decision
            is_drowsy = eyes_indicate_drowsy or is_yawning

            return DecisionResult( is_drowsy=is_drowsy, eye_status=eyes_indicate_drowsy, yawn_status=is_yawning, confidence=confidence, head_pose_status=head_pose_position, success=True)

        except Exception as e:
            print(f"Error in decision logic: {str(e)}")
            return DecisionResult(False, False, False, 0.0, False, False)
        
        
    def _get_default_result(self) -> DecisionResult:
        """Return default result when processing fails."""
        return DecisionResult(False, False, False, 0.0, self.HEAD_FORWARD, False)
        
    def _check_confidence(self, states: StateResults) -> bool:
        return all(conf >= self.min_confidence for conf in [states.confidence_left, states.confidence_right, states.confidence_mouth])
        

    def _check_eye_state(self, left_eye: int, right_eye: int, left_conf: float, right_conf: float) -> bool:
        
        if left_conf >= self.eye_threshold and right_conf >= self.eye_threshold:
            return left_eye == self.EYES_CLOSED and right_eye == self.EYES_CLOSED
        return False
    
    
    def _check_yawn_state(self, mouth_state: int, mouth_conf: float) -> bool:
        return mouth_state == self.YAWNING and mouth_conf >= self.yawn_threshold

    def _calculate_confidence(self, left_conf: float, right_conf: float, mouth_conf: float, eyes_indicate_drowsy: bool, is_yawning: bool) -> float:
       
        # Calculate base confidence from measurements
        eye_confidence = (left_conf + right_conf) / 2
        
        # If alert state (not drowsy)
        if not eyes_indicate_drowsy and not is_yawning:
            
            # High confidence in alert state when all measurements are confident
            return (eye_confidence + mouth_conf) / 2
        
        
         # If both eyes closed and yawning (most severe case)
        if eyes_indicate_drowsy and is_yawning:
            return 0.6 * eye_confidence + 0.4 * mouth_conf  # Weight eyes more heavily

        # If only eyes are closed
        if eyes_indicate_drowsy:
            return (left_conf + right_conf) / 2  # Average of eye confidences

        # If only yawning
        if is_yawning:
            return mouth_conf  # Just use yawn confidence

    # If neither (shouldn't reach here if called correctly)
        return 0.0
    
    
    
    def _get_head_position(self, head_pose: Optional[Tuple[float, float, float]]) -> str:
        """
        Determine head position based on pose angles.
        Returns one of five states: Forward, Looking Left, Looking Right, Looking Up, Looking Down
        """
        if head_pose is None:
            return self.HEAD_FORWARD
            
        pitch, yaw, roll = head_pose
        threshold = self.head_pose_threshold
        
        # Check pitch (nod)
        if pitch < -threshold:      # Looking up
            return self.HEAD_DOWN
        if pitch > threshold:       # Looking down
            return self.HEAD_UP
            
        # Check yaw (turn)
        if yaw < -threshold:        # Looking left
            return self.HEAD_RIGHT
        if yaw > threshold:         # Looking right
            return self.HEAD_LEFT
        
        return self.HEAD_FORWARD