import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from detection.tracker import Tracker
from detection.face_detection import FaceDetection
from detection.feature_extraction import FeatureExtraction
from detection.state_classification import StateClassification
from detection.decision_logic import DecisionLogic
from detection.config import Config
import json

class VideoSystemEvaluator:
    def __init__(self,
                 face_detector: FaceDetection,
                 feature_extractor: FeatureExtraction,
                 state_classifier: StateClassification,
                 decision_logic: DecisionLogic,
                 tracker: Tracker):
        self.face_detector = face_detector
        self.feature_extractor = feature_extractor
        self.state_classifier = state_classifier
        self.decision_logic = decision_logic
        self.tracker = tracker
        
        # Initialize metrics storage
        self.video_results = []

    def process_video(self, video_path: Path, ground_truth_label: int) -> Dict:
        """Process a single video through the complete pipeline"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {
                'video_path': str(video_path),
                'success': False,
                'error': 'Failed to open video'
            }

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        analysis = {
            'video_path': str(video_path),
            'true_label': ground_truth_label,
            'frames_processed': 0,
            'drowsy_detections': [],  # List to store each drowsy detection
            'detection_count': 0,
            'face_detection_failures': 0,
            'temporal_decisions': [],
            'success': True
        }

        frame_results = []
        with tqdm(total=frame_count, desc=f"Processing {video_path.name}") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    # Face detection
                    face_region = self.face_detector.detect_face(frame)
                    if not face_region.success:
                        analysis['face_detection_failures'] += 1
                        continue

                    # Feature extraction
                    features = self.feature_extractor.process_face(face_region)
                    if not features.success:
                        continue

                    # State classification
                    states = self.state_classifier.process_features(features)
                    if not states.success:
                        continue

                    # Decision logic
                    decision = self.decision_logic.determine_drowsiness(states)
                    
                    # Add decision to tracker
                    self.tracker.add_decision(decision)
                    
                    # Use tracker's built-in temporal analysis
                    confidence = self.tracker.aggregated_confidence()
                 
                    is_drowsy = self.tracker.is_drowsy()
                    
                    
                    # Record drowsy detection (tracker will clear its buffer after detection)
                    if is_drowsy:
                        analysis['drowsy_detections'].append({
                            'frame': analysis['frames_processed'],
                            'time': analysis['frames_processed'] / fps,
                            'confidence': confidence,
                        })
                        analysis['detection_count'] += 1

                    frame_results.append({
                        'frame_idx': analysis['frames_processed'],
                        'is_drowsy': is_drowsy,
                        'confidence': confidence
                    })

                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue
                
                analysis['frames_processed'] += 1
                pbar.update(1)

        cap.release()
        
        # Final video classification and analysis
        if frame_results:
            analysis['predicted_label'] = 1 if analysis['detection_count'] > 0 else 0
            analysis['prediction_correct'] = analysis['predicted_label'] == ground_truth_label
            
            # Additional analysis metrics
            if analysis['drowsy_detections']:
                analysis.update({
                    'first_detection_frame': analysis['drowsy_detections'][0]['frame'],
                    'first_detection_time': analysis['drowsy_detections'][0]['time'],
                    'max_confidence': max(det['confidence'] for det in analysis['drowsy_detections']),
                    'multiple_detections': analysis['detection_count'] > 1
                })
                
                # Calculate time between detections if multiple detections occurred
                if len(analysis['drowsy_detections']) > 1:
                    time_between_detections = [
                        analysis['drowsy_detections'][i]['time'] - analysis['drowsy_detections'][i-1]['time']
                        for i in range(1, len(analysis['drowsy_detections']))
                    ]
                    analysis['time_between_detections'] = time_between_detections
        
        return analysis

    def evaluate_dataset(self, data_dir: Path, writer: SummaryWriter, log_dir: Path, config: Config) -> Dict:
        """Evaluate the complete system on a video dataset"""
        drowsy_dir = data_dir / "drowsy"
        not_drowsy_dir = data_dir / "not_drowsy"
        
        # Save configuration settings
        config_dict = {
            # Temporal analysis settings
            'WINDOW_SIZE_DROWSINESS': config.WINDOW_SIZE_DROWSINESS,
            'DROWSY_THRESHOLD': config.DROWSY_THRESHOLD,
            'WINDOW_SIZE_HEAD_POSE': config.WINDOW_SIZE_HEAD_POSE,
            'HEAD_POSE_NON_FORWARD_THRESHOLD': config.HEAD_POSE_NON_FORWARD_THRESHOLD,
            
            # Classification thresholds
            'EYE_CONFIDENCE_THRESHOLD': config.EYE_CONFIDENCE_THRESHOLD,
            'MOUTH_CONFIDENCE_THRESHOLD': config.MOUTH_CONFIDENCE_THRESHOLD,
            'MIN_CONFIDENCE': config.MIN_CONFIDENCE,
            'HEAD_POSE_THRESHOLD': config.HEAD_POSE_THRESHOLD,
            
            # Face Detection settings
            'FACE_DETECTION_CONFIDENCE': config.FACE_DETECTION_CONFIDENCE,
            'FACE_PADDING_PERCENT': config.FACE_PADDING_PERCENT,
            
            # Feature Extraction settings
            'MIN_DETECTION_CONF': config.MIN_DETECTION_CONF,
            'MIN_TRACKING_CONF': config.MIN_TRACKING_CONF,
            
            # Facial measurement thresholds
            'EAR_THRESHOLD': config.EAR_THRESHOLD,
            'MAR_THRESHOLD': config.MAR_THRESHOLD,
        }
        video_formats = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
        
        # Save config to JSON
        config_path = log_dir / "evaluation_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        video_results = []
        
        # Process drowsy videos (both .mp4 and .avi)
        for video_format in video_formats:
            for video_path in drowsy_dir.glob(video_format):
                analysis = self.process_video(video_path, ground_truth_label=1)
                video_results.append(analysis)
        
        # Process not drowsy videos (both .mp4 and .avi)
        for video_format in video_formats:
            for video_path in not_drowsy_dir.glob(video_format):
                analysis = self.process_video(video_path, ground_truth_label=0)
                video_results.append(analysis)
        
        # Convert to DataFrame and calculate metrics
        results_df = pd.DataFrame(video_results)
        
        # Calculate metrics
        metrics = self._calculate_metrics(results_df)
        
        # Save results and metrics
        results_df.to_csv(log_dir / "video_evaluation_results.csv")
        
        # Save metrics to JSON with proper formatting
        metrics_path = log_dir / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Print metrics to console
        print("\nEvaluation Metrics:")
        print("-" * 50)
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        print("-" * 50)
        
        # Log to tensorboard
        for metric_name, value in metrics.items():
            writer.add_scalar(f'Metrics/{metric_name}', value)
        
        return {
            'metrics': metrics,
            'results_df': results_df
        }

    def _calculate_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate evaluation metrics from results DataFrame"""
        successful_videos = results_df[results_df['success']]
        drowsy_videos = successful_videos[successful_videos['true_label'] == 1]
        alert_videos = successful_videos[successful_videos['true_label'] == 0]
        
        metrics = {
            'total_videos': len(results_df),
            'successful_videos': len(successful_videos),
            'accuracy': (successful_videos['prediction_correct'].sum() / len(successful_videos)) 
                if len(successful_videos) > 0 else 0,
            'drowsy_detection_rate': (drowsy_videos['detection_count'].sum() / len(drowsy_videos))
                if len(drowsy_videos) > 0 else 0,
            'false_positive_rate': (alert_videos['detection_count'] > 0).sum() / len(alert_videos)
                if len(alert_videos) > 0 else 0,
            'average_detection_latency': drowsy_videos['first_detection_time'].mean()
                if len(drowsy_videos) > 0 else 0,
            'multiple_detection_rate': (drowsy_videos['multiple_detections'].sum() / len(drowsy_videos))
                if len(drowsy_videos) > 0 else 0,
            'face_detection_failure_rate': (successful_videos['face_detection_failures'].sum() / 
                successful_videos['frames_processed'].sum())
        }
        
        return metrics

def main():
    config = Config()
    config.validate()
    
    # Initialize components
    face_detector = FaceDetection(config)
    feature_extractor = FeatureExtraction(config)
    state_classifier = StateClassification(config)
    decision_logic = DecisionLogic(config)
    tracker = Tracker(config)
    
    # Initialize evaluator
    evaluator = VideoSystemEvaluator(
        face_detector=face_detector,
        feature_extractor=feature_extractor,
        state_classifier=state_classifier,
        decision_logic=decision_logic,
        tracker=tracker
    )
    
    # Setup logging
    log_dir = Path("../../logs/video_evaluation/Badr_dataset_1-window_size_smaller")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        data_dir=Path("/Users/ahmedalkhulayfi/Downloads/benchmark/badr-01"),
        writer=writer,
        log_dir=log_dir,
        config=config  # Pass config to evaluate_dataset
    )
    
    # Print results
    print("\nEvaluation Results:")
    for metric_name, value in results['metrics'].items():
        print(f"{metric_name}: {value:.4f}")
    
    writer.close()

if __name__ == "__main__":
    main() 