import mediapipe as mp
import cv2

class FeatureExtraction:
    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=False, min_detection_con=0.5, min_tracking_con=0.5):
        # Initialize the parameters for face mesh detection
        self.static_image_mode = static_image_mode  # Whether to process images (True) or video stream (False)
        self.max_num_faces = max_num_faces  # Maximum number of faces to detect
        self.refine_landmarks = refine_landmarks  # Whether to refine iris landmarks for better precision
        self.min_detection_con = min_detection_con  # Minimum confidence for face detection
        self.min_tracking_con = min_tracking_con  # Minimum confidence for tracking

        # Initialize Mediapipe FaceMesh solution
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode,
                                                 self.max_num_faces,
                                                 self.refine_landmarks,
                                                 self.min_detection_con,
                                                 self.min_tracking_con)

        # Store the landmark indices for specific facial features
        # These are predefined Mediapipe indices for left and right eyes and mouth

        self.LEFT_EYE_LANDMARKS = [63, 107, 128 ,117 ]  # Left eye landmarks

        self.RIGHT_EYE_LANDMARKS = [336, 293, 346, 357]  # Right eye landmarks
        
        self.MOUTH_LANDMARKS = [216 , 436, 430, 210]  # Mouth landmarks

    def extract_features(self, img):
         # Initialize a dictionary to store the landmarks for facial features
        landmarks = {}

        # Convert the input image to RGB as Mediapipe expects RGB images
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to find face landmarks using the FaceMesh model
        results = self.faceMesh.process(imgRGB)

        # Check if any faces were detected
        if results.multi_face_landmarks:
            # Iterate over detected faces (here, max_num_faces = 1, so usually one face)
            for faceLms in results.multi_face_landmarks:
                # Initialize lists in the landmarks dictionary to store each facial feature's coordinates
                landmarks["left_eye_landmarks"] = []
                landmarks["right_eye_landmarks"] = []
                landmarks["mouth_landmarks"] = []
                # landmarks["all_landmarks"] = []  # Store all face landmarks for complete face mesh
                
                # Loop through all face landmarks
                for i, lm in enumerate(faceLms.landmark):
                    h, w, ic = img.shape  # Get image height, width, and channel count
                    x, y = int(lm.x * w), int(lm.y * h)  # Convert normalized coordinates to pixel values
                    
                    # Store the coordinates of all landmarks
                    # landmarks["all_landmarks"].append((x, y))

                    # Store specific feature landmarks based on the predefined indices
                    if i in self.LEFT_EYE_LANDMARKS:
                        landmarks["left_eye_landmarks"].append((x, y))  # Left eye
                    if i in self.RIGHT_EYE_LANDMARKS:
                        landmarks["right_eye_landmarks"].append((x, y))  # Right eye
                    if i in self.MOUTH_LANDMARKS:
                        landmarks["mouth_landmarks"].append((x, y))  # Mouth

        # Return the processed image and the dictionary of feature landmarks
        return landmarks
    
    
    def get_feature_region(self, img, landmarks, padding=10):
        if not landmarks:
            return None
            
        # Get bounding box coordinates
        x_coords = [x for x, y in landmarks]
        y_coords = [y for x, y in landmarks]
        
        left = max(0, min(x_coords) - padding)
        right = min(img.shape[1], max(x_coords) + padding)
        top = max(0, min(y_coords) - padding)
        bottom = min(img.shape[0], max(y_coords) + padding)
        
        # Extract the region
        region = img[top:bottom, left:right]
        
        # Ensure the region is not empty
        if region.size == 0:
            return None
        
        return region
