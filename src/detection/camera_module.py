import cv2
from config import Config


class CameraModule:

    def __init__(self, config: Config):

        self.config = config
        self._capture = None
        self.initialize_camera()

    def initialize_camera(self) -> None:

        try:
            self._capture = cv2.VideoCapture(self.config.CAMERA_INDEX)

            if not self._capture.isOpened():
                raise RuntimeError(
                    f"\033[91mFailed to open camera at index: {self.config.CAMERA_INDEX} please check if the camera is connected and try again\033[0m"
                )

            # Set camera properties if specified in config
            if hasattr(self.config, "FRAME_WIDTH"):
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
            if hasattr(self.config, "FRAME_HEIGHT"):
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)

        except Exception as e:
            raise RuntimeError(f"Camera initialization failed: {str(e)}")

    def capture_frame(self):
        if not self._capture or not self._capture.isOpened():
            return False, None
        
        success, frame = self._capture.read()
        
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        return success, frame

    def release(self) -> None:
        if self._capture:
            self._capture.release()
            self._capture = None
