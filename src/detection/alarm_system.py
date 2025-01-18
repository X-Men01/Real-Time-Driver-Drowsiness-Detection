import winsound  # Use for Windows; use `playsound` or GPIO for other platforms

class AlarmSystem:
    def __init__(self):
        pass

    def trigger_alarm(self):
        print("Drowsiness detected! Triggering alarm...")
        winsound.Beep(1000, 1000)  # Beep at 1000 Hz for 1 second
