import cv2

class CameraModule:
    def __init__(self, camera_index=0, frame_width=640, frame_height=640):
        self.cap = cv2.VideoCapture(camera_index)
        self.frame_width = frame_width
        self.frame_height = frame_height

    def capture_frame(self):
        ret, frame = self.cap.read()
        if ret:
            #! you need to specify the interpolation method
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        else:
            raise Exception("Failed to capture frame")

    def release(self):
        self.cap.release()

    def get_camera_info(self) -> dict:
        """Get current camera settings and status.
        
        Returns:
            Dictionary containing camera information
        """
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
            'is_open': True
        }
