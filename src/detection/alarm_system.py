import playsound
from config import Config

class AlarmSystem:
    def __init__(self,config: Config):
        self.alarm_file = config.ALARM_FILE

    def trigger_alarm(self):
        playsound.playsound(self.alarm_file)