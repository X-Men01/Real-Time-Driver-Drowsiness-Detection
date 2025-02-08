import playsound
import threading
from detection.config import Config

class AlarmSystem:
    def __init__(self,config: Config):
        self.alarm_files = {
            "Drowsiness": config.DROWSINESS_ALARM_FILE,
            "Distraction": config.DISTRACTION_ALARM_FILE,
            "Face_not_detected": config.FACE_NOT_DETECTED_ALARM_FILE
        }
        self.alarm_thread = None

    def _play_alarm(self,alarm_type: str):
        playsound.playsound(self.alarm_files[alarm_type])

    def trigger_alarm(self , alarm_type: str):
        if self.alarm_thread and self.alarm_thread.is_alive():
            return
    
        self.alarm_thread = threading.Thread(target=self._play_alarm, daemon=True, args=(alarm_type,))
        self.alarm_thread.start()
        
    @property
    def is_active(self):
        return self.alarm_thread is not None and self.alarm_thread.is_alive()   