class DecisionLogic:
    def __init__(self, eye_threshold=0.8, yawning_threshold=0.5):
        self.eye_threshold = eye_threshold
        self.yawning_threshold = yawning_threshold

    def determine_drowsiness(self, eye_state, mouth_state):
        if eye_state == 1 or mouth_state == 1:  # Closed or Yawning
            return True  # Drowsy
        return False
