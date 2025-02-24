from collections import deque
from detection.decision_logic import DecisionResult
from detection.config import Config


class Tracker:
   
    def __init__(self, config: Config):
        self.drowsy_threshold = config.DROWSY_THRESHOLD
        self.drowsy_buffer = deque(maxlen=config.WINDOW_SIZE_DROWSINESS)
        
       
        
        self.head_pose_non_forward_threshold = config.HEAD_POSE_NON_FORWARD_THRESHOLD  
        self.head_pose_buffer = deque(maxlen=config.WINDOW_SIZE_HEAD_POSE)  

    def add_decision(self, decision: DecisionResult):
        
        self.drowsy_buffer.append(decision)
        self.head_pose_buffer.append(decision.head_pose_status)

    def is_drowsy(self) -> bool:

        if not self.drowsy_buffer or len(self.drowsy_buffer) < self.drowsy_buffer.maxlen:
            return False

        
        drowsy_count = sum(1 for decision in self.drowsy_buffer if decision.is_drowsy)
        ratio = drowsy_count / len(self.drowsy_buffer)
        if ratio >= self.drowsy_threshold:
            self.drowsy_buffer.clear()
            return True
        else:
            return False
    
    def aggregated_confidence(self) -> float:
          
           
           drowsy_confidences = [decision.confidence for decision in self.drowsy_buffer if decision]
           if drowsy_confidences:
               return sum(drowsy_confidences) / len(drowsy_confidences)
           return 0.0
       
    def is_head_pose_alert(self) -> bool:
        
        if not self.head_pose_buffer or len(self.head_pose_buffer) < self.head_pose_buffer.maxlen:
            return False
        
        non_forward_count = sum(1 for status in self.head_pose_buffer if status != "Forward")
        ratio = non_forward_count / len(self.head_pose_buffer)
        
        
        if ratio >= self.head_pose_non_forward_threshold:
            self.head_pose_buffer.clear() 
            return True
        else:
            return False