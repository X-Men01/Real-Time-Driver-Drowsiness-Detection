from collections import deque
from detection.decision_logic import DecisionResult
from config import Config


class DrowsinessTracker:
   
    def __init__(self, config: Config):
        self.window_size = config.WINDOW_SIZE
        self.drowsy_threshold = config.DROWSY_THRESHOLD
        self.buffer = deque(maxlen=self.window_size)

    def add_decision(self, decision: DecisionResult):
        
        self.buffer.append(decision)

    def is_drowsy(self) -> bool:

        if not self.buffer:
            return False

        # Count the frames where decision indicates drowsiness.
        drowsy_count = sum(1 for decision in self.buffer if decision.is_drowsy)
        ratio = drowsy_count / len(self.buffer)
        if ratio >= self.drowsy_threshold:
            self.buffer.clear()
            return True
        else:
            return False
    
    def aggregated_confidence(self) -> float:
           """
           Calculate the overall confidence for drowsiness detection by averaging
           the confidence scores only of the frames flagged as drowsy.
           """
           # Filter only decisions where drowsiness was detected.
           drowsy_confidences = [decision.confidence for decision in self.buffer if decision.is_drowsy]
           if drowsy_confidences:
               return sum(drowsy_confidences) / len(drowsy_confidences)
           return 0.0