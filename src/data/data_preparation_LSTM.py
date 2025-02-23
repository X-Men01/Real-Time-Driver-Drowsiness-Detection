import cv2
import os
import glob
import json
from datetime import datetime
from tqdm import tqdm
from detection.face_detection import FaceDetection
from detection.feature_extraction import FeatureExtraction
from detection.facial_measurement import FacialMeasurements
from detection.config import Config
from detection.state_classification import StateClassification
import numpy as np
from pathlib import Path

class BasicDataPreparation:
    def __init__(self, config: Config, sequence_length=2):
        self.sequence_length = sequence_length
        self.face_detector = FaceDetection(config)
        self.feature_extractor = FeatureExtraction(config)
        self.facial_measurements = FacialMeasurements(config)
        self.state_classifier = StateClassification(config)
        self.image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        
        # Define features we actually have in the project
        self.feature_names = [
            'ear_left',      # Left Eye Aspect Ratio
            'ear_right',     # Right Eye Aspect Ratio
            'mar',           # Mouth Aspect Ratio
            'pitch',         # Head pose pitch
            'yaw',          # Head pose yaw
            'roll'          # Head pose roll
        ]

    def process_single_image(self, image_path):
        """Process one image and extract features"""
        try:
            # 1. Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_path}")
                return None
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 2. Detect face
            face_result = self.face_detector.detect_face(image)
            if not face_result.success:
                return None
            
            # 3. Extract features
            facial_features = self.feature_extractor.process_face(face_result)
            if not facial_features.success:
                return None
                
            # 4. Calculate metrics
            metrics = self.facial_measurements.calculate_metrics(facial_features)
            states = self.state_classifier.process_features(facial_features)
            
            # 5. Get head pose
            head_pose = states.head_pose  # (pitch, yaw, roll)
            pitch, yaw, roll = (0.0, 0.0, 0.0) if head_pose is None else head_pose
            
            # 6. Return all available features
            return {
                # Eye features
                'ear_left': metrics.left_ear,
                'ear_right': metrics.right_ear,
                
                # Mouth features
                'mar': metrics.mar,
                
                # Head pose features
                'pitch': pitch,
                'yaw': yaw,
                'roll': roll
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def process_directory(self, directory_path, label):
        """Process all images in a directory with progress bar"""
        features_list = []
        image_paths = []
        for ext in self.image_extensions:
            image_paths.extend(glob.glob(os.path.join(directory_path, ext)))
        
        image_paths = sorted(image_paths)
        print(f"Found {len(image_paths)} images in {directory_path}")
        
        # Add progress bar
        for img_path in tqdm(image_paths, desc="Processing images"):
            features = self.process_single_image(img_path)
            if features is not None:
                features_list.append(features)
        
        return features_list

    def create_sequences(self, features_list, label, sequence_length=2):
        """Create sequences from features list with labels"""
        sequences = []
        
        # Create sequences with their labels
        for i in range(0, len(features_list) - sequence_length + 1):
            sequence = features_list[i:i + sequence_length]
            sequences.append({
                'features': sequence,
                'label': label
            })
            
        return sequences

    def save_processed_data(self, drowsy_sequences, not_drowsy_sequences, save_dir="processed_data"):
        """Save sequences as a single numpy array with their corresponding labels"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Combine drowsy and not drowsy sequences
        all_features = np.array([
            [
                [frame[feature_name] for feature_name in self.feature_names]
                for frame in seq['features']
            ]
            for seq in (drowsy_sequences + not_drowsy_sequences)
        ], dtype=np.float32)
        
        # Create corresponding labels
        all_labels = np.array([seq['label'] for seq in (drowsy_sequences + not_drowsy_sequences)], dtype=np.int32)
        
        
        expected_shape = (
            len(drowsy_sequences) + len(not_drowsy_sequences),
            self.sequence_length,
            len(self.feature_names)
        )
        if all_features.shape != expected_shape:
            raise ValueError(f"Unexpected shape: {all_features.shape}, expected {expected_shape}")
        
        # Save as a single file
        np.savez(
            save_dir / 'processed_sequences.npz',
            features=all_features,
            labels=all_labels
        )
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'creation_date': str(datetime.now()),
            'total_sequences': len(all_labels),
            'drowsy_count': len(drowsy_sequences),
            'not_drowsy_count': len(not_drowsy_sequences),
            'features_shape': all_features.shape
        }
        
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Saved processed data to {save_dir}")
        print(f"Features shape: {all_features.shape}")
        print(f"Labels shape: {all_labels.shape}")

def prepare_data(data_dir="train_data", force_reprocess=False, save_dir="processed_data"):
    prep = BasicDataPreparation(Config())
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        raise ValueError(f"Directory not found: {data_dir}")
    
    # Check for drowsy and not_drowsy directories
    drowsy_dir = os.path.join(data_dir, "drowsy")
    not_drowsy_dir = os.path.join(data_dir, "not_drowsy")
    
    if not os.path.exists(drowsy_dir) or not os.path.exists(not_drowsy_dir):
        raise ValueError(f"Missing required directories in {data_dir}")
    
    # Process both classes
    print("Processing drowsy images...")
    drowsy_features = prep.process_directory(drowsy_dir, label=1)
    print("drowsy_features",drowsy_features)
    print("\nProcessing not drowsy images...")
    not_drowsy_features = prep.process_directory(not_drowsy_dir, label=0)
    
    # Check if we have enough data
    if len(drowsy_features) < prep.sequence_length or len(not_drowsy_features) < prep.sequence_length:
        raise ValueError("Not enough valid images to create sequences")
    
    # Create sequences
    print("\nCreating sequences...")
    drowsy_sequences = prep.create_sequences(drowsy_features, label=1)
    not_drowsy_sequences = prep.create_sequences(not_drowsy_features, label=0)
    
    # Save the sequences
    prep.save_processed_data(drowsy_sequences, not_drowsy_sequences, save_dir)
    
    # Print statistics
    print(f"\nDataset statistics:")
    print(f"Drowsy sequences: {len(drowsy_sequences)}")
    print(f"Not drowsy sequences: {len(not_drowsy_sequences)}")
    print(f"Sequence length: {prep.sequence_length}")
    print(f"Features per frame: {len(drowsy_features[0]) if drowsy_features else 0}")

if __name__ == "__main__":
    try:
        prepare_data(
            data_dir="/Users/ahmedalkhulayfi/Downloads/benchmark/test_LSTM_perparation",
            force_reprocess=True,
            save_dir="processed_data"
        )
    except Exception as e:
        print(f"Error: {e}")