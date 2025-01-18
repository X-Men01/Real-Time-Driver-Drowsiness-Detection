import cv2

class CameraModule:
    def __init__(self, camera_index=0, frame_width=224, frame_height=224):
        self.cap = cv2.VideoCapture(camera_index)
        self.frame_width = frame_width
        self.frame_height = frame_height

    def capture_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            return frame
        else:
            raise Exception("Failed to capture frame")

    def release(self):
        self.cap.release()
