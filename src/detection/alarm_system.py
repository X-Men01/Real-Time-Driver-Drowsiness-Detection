import playsound

class AlarmSystem:
    def __init__(self):
        self.alarm_file = '../../assets/alarm_sound.mp3'

    def trigger_alarm(self):
        print("Drowsiness detected! Triggering alarm...")
        playsound.playsound(self.alarm_file)