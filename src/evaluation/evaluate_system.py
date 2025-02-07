import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# Import your pipeline components
from detection.face_detection import FaceDetection
from detection.feature_extraction import FeatureExtraction
from detection.state_classification import StateClassification
from detection.decision_logic import DecisionLogic
from detection.config import Config

class SystemEvaluator:
    def __init__(self, 
                 face_detector: FaceDetection,
                 feature_extractor: FeatureExtraction,
                 state_classifier: StateClassification,
                 decision_logic: DecisionLogic):
        self.face_detector = face_detector
        self.feature_extractor = feature_extractor
        self.state_classifier = state_classifier
        self.decision_logic = decision_logic
        
        # Initialize metrics storage
        self.predictions = []
        self.ground_truth = []
        self.failure_cases = []

    def process_single_image(self, image_path: Path) -> dict:
        """Process a single image through the complete pipeline and return detailed analysis"""
        analysis = {
            'image_path': str(image_path),
            'face_detection': {'success': False, 'confidence': 0.0},
            'feature_extraction': {'success': False},
            'state_classification': {
                'left_eye_state': None,
                'right_eye_state': None,
                'mouth_state': None,
                'confidence_left': 0.0,
                'confidence_right': 0.0,
                'confidence_mouth': 0.0
            },
            'decision': {
                'is_drowsy': None,
                'eye_status': None,
                'yawn_status': None,
                'confidence': 0.0,
                'success': False
            },
            'pipeline_success': False
        }
        
        try:
            # Read image
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Face detection
            face_region = self.face_detector.detect_face(image)
            analysis['face_detection'].update({
                'success': face_region.success,
                'confidence': face_region.confidence
            })
            if not face_region.success:
                return analysis
                
            # Feature extraction
            features = self.feature_extractor.process_face(face_region)
            analysis['feature_extraction']['success'] = features.success
            if not features.success:
                return analysis
                
            # State classification
            states = self.state_classifier.process_features(features)
            analysis['state_classification'].update({
                'left_eye_state': states.left_eye_state,
                'right_eye_state': states.right_eye_state,
                'mouth_state': states.mouth_state,
                'confidence_left': states.confidence_left,
                'confidence_right': states.confidence_right,
                'confidence_mouth': states.confidence_mouth
            })
            if not states.success:
                return analysis
            
            # Decision logic
            decision = self.decision_logic.determine_drowsiness(states)
            analysis['decision'].update({
                'is_drowsy': decision.is_drowsy,
                'eye_status': decision.eye_status,
                'yawn_status': decision.yawn_status,
                'confidence': decision.confidence,
                'success': decision.success
            })
            
            analysis['pipeline_success'] = True
            return analysis
                
        except Exception as e:
            analysis['error'] = str(e)
            return analysis

    def evaluate_dataset(self, data_dir: Path, writer: SummaryWriter, log_dir: Path) -> Dict:
        """Evaluate the complete system on a dataset"""
        drowsy_dir = data_dir / "drowsy"
        not_drowsy_dir = data_dir / "not_drowsy"
        
        all_cases = []
        
        # Process drowsy cases
        for img_path in tqdm(list(drowsy_dir.glob("*.jpg")), desc="Processing drowsy cases"):
            analysis = self.process_single_image(img_path)
            analysis['true_label'] = 1  # drowsy
            all_cases.append(analysis)
                    
        # Process not drowsy cases
        for img_path in tqdm(list(not_drowsy_dir.glob("*.jpg")), desc="Processing alert cases"):
            analysis = self.process_single_image(img_path)
            analysis['true_label'] = 0  # not drowsy
            all_cases.append(analysis)
                    
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(all_cases)
        
        # Calculate metrics for successful pipeline runs
        successful_cases = results_df[results_df['pipeline_success']]
        y_true = successful_cases['true_label']
        y_pred = successful_cases['decision'].apply(lambda x: int(x['is_drowsy']))
        
        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=['Alert', 'Drowsy'], output_dict=True)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(log_dir / 'confusion_matrix.png')
        plt.close()
        
        # Additional error analysis
        error_analysis = {
            'total_images': len(results_df),
            'face_detection_failures': len(results_df[~results_df['face_detection'].apply(lambda x: x['success'])]),
            'feature_extraction_failures': len(results_df[~results_df['feature_extraction'].apply(lambda x: x['success'])]),
            'pipeline_failures': len(results_df[~results_df['pipeline_success']]),
            'classification_failures': len(successful_cases[successful_cases['true_label'] != successful_cases['decision'].apply(lambda x: int(x['is_drowsy']))])
        }
        
        # Save detailed results
        results_df.to_csv(log_dir / "detailed_results.csv")
        
        # Create simplified results for quick review
        simplified_results = pd.DataFrame({
            'image_path': results_df['image_path'],
            'true_label': results_df['true_label'],
            'predicted_label': results_df['decision'].apply(lambda x: int(x['is_drowsy']) if x['is_drowsy'] is not None else None),
            'correct_classification': results_df.apply(
                lambda row: row['true_label'] == int(row['decision']['is_drowsy']) 
                if row['pipeline_success'] and row['decision']['is_drowsy'] is not None 
                else None, 
                axis=1
            )
        })
        simplified_results.to_csv(log_dir / "simplified_results.csv")
        
        # Log error analysis
        writer.add_text('Error_Analysis', str(error_analysis))
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'error_analysis': error_analysis,
            'results_df': results_df,
            'simplified_results': simplified_results
        }

    def plot_confidence_distribution(self, results_df: pd.DataFrame, log_dir: Path):
        """Plot confidence score distributions for correct and incorrect predictions"""
        successful_cases = results_df[results_df['pipeline_success']]
        correct_predictions = successful_cases[
            successful_cases['true_label'] == successful_cases['decision'].apply(lambda x: int(x['is_drowsy']))
        ]
        incorrect_predictions = successful_cases[
            successful_cases['true_label'] != successful_cases['decision'].apply(lambda x: int(x['is_drowsy']))
        ]

        plt.figure(figsize=(10, 6))
        plt.hist(
            [
                correct_predictions['decision'].apply(lambda x: x['confidence']),
                incorrect_predictions['decision'].apply(lambda x: x['confidence'])
            ],
            label=['Correct Predictions', 'Incorrect Predictions'],
            bins=20,
            alpha=0.7
        )
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.title('Confidence Score Distribution')
        plt.legend()
        plt.savefig(log_dir / 'confidence_distribution.png')
        plt.close()

def main():
    config = Config()
    config.validate()
    
    face_detector = FaceDetection(config)
    feature_extractor = FeatureExtraction(config)
    state_classifier = StateClassification(config)
    decision_logic = DecisionLogic(config)
    
    # Initialize evaluator
    evaluator = SystemEvaluator(
        face_detector=face_detector,
        feature_extractor=feature_extractor,
        state_classifier=state_classifier,
        decision_logic=decision_logic
    )
    
    # Setup tensorboard writer and log directory
    log_dir = Path("../../logs/system_evaluation/Closed_Eyes_dataset_V3")
   
    writer = SummaryWriter(log_dir)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        data_dir=Path("/Users/ahmedalkhulayfi/Downloads/dataset_B_FacialImages_highResolution"),
        writer=writer,
        log_dir=log_dir
    )
    
    # Print summary
    print("\nEvaluation Results:")
    print("\nClassification Report:")
    print(pd.DataFrame(results['classification_report']).T)
    
    # Print error analysis
    print("\nError Analysis:")
    for key, value in results['error_analysis'].items():
        print(f"{key}: {value}")
    
    # Plot confidence distribution
    evaluator.plot_confidence_distribution(results['results_df'], log_dir)
    
    # Close tensorboard writer
    writer.close()
    
    return results

if __name__ == "__main__":
    main()