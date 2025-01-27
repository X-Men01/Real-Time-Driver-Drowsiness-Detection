import playsound

class AlarmSystem:
    def __init__(self,alarm_file: str):
        self.alarm_file = alarm_file

    def trigger_alarm(self):
        print("Drowsiness detected! Triggering alarm...")
        playsound.playsound(self.alarm_file)