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
        
    def process_single_image(self, image_path: Path) -> Tuple[bool, float]:
        """Process a single image through the complete pipeline"""
        try:
          
            
            # Read image
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Face detection
            face_region = self.face_detector.detect_face(image)
            if not face_region.success:
                raise Exception("Face detection failed")
                
            # Feature extraction
            features = self.feature_extractor.process_face(face_region)
            if not features.success:
                raise Exception("Feature extraction failed")
                
            # State classification
            states = self.state_classifier.process_features(features)
            if not states.success:
                raise Exception("State classification failed")
            
            # Decision logic with detailed logging
      
        
            # Decision logic
            decision = self.decision_logic.determine_drowsiness(states)
            if not decision.success:
                print("\nState Classification Results:")
                print(f"Left Eye: {states.left_eye_state} (conf: {states.confidence_left:.2f})")
                print(f"Right Eye: {states.right_eye_state} (conf: {states.confidence_right:.2f})")
                print(f"Mouth: {states.mouth_state} (conf: {states.confidence_mouth:.2f})")
                raise Exception("Decision logic failed")
                
            
            return decision.is_drowsy
            
        except Exception as e:
            self.failure_cases.append((str(image_path), str(e)))
            return None
            
    def evaluate_dataset(self, data_dir: Path, writer: SummaryWriter) -> Dict:
        """Evaluate the complete system on a dataset"""
        drowsy_dir = data_dir / "drowsy"
        not_drowsy_dir = data_dir / "not_drowsy"
        
        all_cases = []
        # Process drowsy cases
        for img_path in tqdm(list(drowsy_dir.glob("*.jpg")), desc="Processing drowsy cases"):
            prediction = self.process_single_image(img_path)
            if prediction is not None:
                all_cases.append({
                    'path': str(img_path),
                    'true_label': 1,  # drowsy
                    'predicted_label': int(prediction),
                })
                
        # Process not drowsy cases
        for img_path in tqdm(list(not_drowsy_dir.glob("*.jpg")), desc="Processing alert cases"):
            prediction = self.process_single_image(img_path)
            if prediction is not None:
                all_cases.append({
                    'path': str(img_path),
                    'true_label': 0,  # not drowsy
                    'predicted_label': int(prediction),
                    
                })
                
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(all_cases)
        
        # Calculate metrics
        y_true = results_df['true_label']
        y_pred = results_df['predicted_label']
        
        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=['Alert', 'Drowsy'], output_dict=True)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred )
        
        # Plot and save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Alert', 'Drowsy'],
                   yticklabels=['Alert', 'Drowsy'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        writer.add_figure('Evaluation/Confusion_Matrix', plt.gcf())
        
        # Log metrics to tensorboard
        writer.add_scalar('Evaluation/Accuracy', report['accuracy'])
        for class_name in ['Alert', 'Drowsy']:
            writer.add_scalar(f'Evaluation/Precision_{class_name}', report[class_name]['precision'])
            writer.add_scalar(f'Evaluation/Recall_{class_name}', report[class_name]['recall'])
            writer.add_scalar(f'Evaluation/F1_{class_name}', report[class_name]['f1-score'])
            
       
        
       
        # Log failure cases if any
        if self.failure_cases:
            writer.add_text('Failures', str(self.failure_cases))
            
        return {
            'classification_report': report,
            'confusion_matrix': cm,
          
            'failure_cases': self.failure_cases,
            'results_df': results_df
        }

def main():
    config = Config()
    config.validate()
    
    
   
    face_detector = FaceDetection(config.FACE_DETECTION_CONFIDENCE, config.FACE_DETECTION_MODEL_SELECTION, config.FACE_PADDING_PERCENT)
    feature_extractor = FeatureExtraction(config.STATIC_IMAGE_MODE, config.MAX_NUM_FACES, config.REFINE_LANDMARKS, config.MIN_DETECTION_CONF, config.MIN_TRACKING_CONF)
    
    state_classifier = StateClassification(config.EYE_MODEL_PATH, config.MOUTH_MODEL_PATH, config.DEVICE)
    decision_logic = DecisionLogic(config.EYE_CONFIDENCE_THRESHOLD, config.MOUTH_CONFIDENCE_THRESHOLD, config.MIN_CONFIDENCE)
   
    
    # Initialize evaluator
    evaluator = SystemEvaluator(
        face_detector=face_detector,
        feature_extractor=feature_extractor,
        state_classifier=state_classifier,
        decision_logic=decision_logic
    )
    
    # Setup tensorboard writer
    log_dir = Path("../../logs/system_evaluation/NTHUDDD_dataset_V1")
    writer = SummaryWriter(log_dir)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        data_dir=Path("/Users/ahmedalkhulayfi/Downloads/train_data"),
        writer=writer
    )
    
    # Print summary
    print("\nEvaluation Results:")
    print("\nClassification Report:")
    print(pd.DataFrame(results['classification_report']).T)
  
    
    print("\nFailure Cases:")
    for case in results['failure_cases']:
        print(f"- {case[0]}: {case[1]}")
    
    results['results_df'].to_csv(log_dir / "results_df.csv")
    writer.close()
    
    return results

if __name__ == "__main__":
    main() 