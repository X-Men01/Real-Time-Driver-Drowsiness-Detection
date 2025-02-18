import matplotlib.pyplot as plt
import cv2

def plot_pipeline(original_frame, face_region,facial_features, states,decision):
    
    class_names_eye = {0: "Close_Eye", 1: "Open_Eye"}
    class_names_mouth = {0: "No_Yawn", 1: "Yawn"}
    # Create a figure with subplots
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))

    # Plot original frame
    axs[0].imshow(original_frame)
    axs[0].set_title("Original Frame")
    axs[0].axis('off')
    # plt.show()
    # Plot face region
   
    axs[1].imshow(face_region.face)
    axs[1].set_title(f"Detected Face {face_region.confidence:.2f}")
    axs[1].axis('off')

    # Plot left eye region
    axs[2].imshow(facial_features.left_eye)
    axs[2].set_title("Extracted Left Eye")
    axs[2].axis('off')
    
    # Plot right eye region
    axs[3].imshow(facial_features.right_eye)
    axs[3].set_title("Extracted Right Eye")
    axs[3].axis('off')

    # Plot mouth region
    axs[4].imshow(facial_features.mouth)
    axs[4].set_title("Extracted Mouth")
    axs[4].axis('off')

    # Show decision
    decision_as_text = (
                        f"is_Drowsy: {decision.is_drowsy} ({decision.confidence:.2f})\n"
                        f"Left Eye: {class_names_eye[states.left_eye_state]} ({states.confidence_left:.2f}), "
                        f"Right Eye: {class_names_eye[states.right_eye_state]} ({states.confidence_right:.2f}), "
                        f"Mouth: {class_names_mouth[states.mouth_state]} ({states.confidence_mouth:.2f})"
                    )

    plt.suptitle(decision_as_text, fontsize=16)

    # Display the plot
    plt.show()

import cv2
import numpy as np
from typing import Tuple

def draw_status_overlay(frame: np.ndarray, states, decision, alarm_active, drowsy_conf, ratios,alarm_type) -> np.ndarray:
    """Draw status information on the video frame"""
    
    # Create a semi-transparent overlay for text background
    overlay = frame.copy()
    output = frame.copy()
    
    # Status box dimensions
    box_height = 180 * 2
    box_width = 400
    padding = 10
    
    # Draw semi-transparent black box
    cv2.rectangle(overlay, (padding, padding), 
                 (box_width, box_height), (0, 0, 0), -1)
    
    # Apply transparency
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_color = (255, 255, 255)  # White
    
    if alarm_active:
        confidence = drowsy_conf
    else:
        confidence = decision.confidence
    # Status texts
    class_names_eye = {0: "Close_Eye", 1: "Open_Eye"}
    texts = [
        f"Drowsiness: {'DROWSY!' if alarm_active else 'Normal'} ({confidence:.2f})",
        f"Eyes: {'Closed' if decision.eye_status else 'Open'} ({(states.confidence_left+states.confidence_right)/2:.2f})",
        f"Left Eye: {class_names_eye[states.left_eye_state]} ({states.confidence_left:.2f}) ",
        f"Right Eye: {class_names_eye[states.right_eye_state]} ({states.confidence_right:.2f}) ",
        f"Mouth: {'Yawning' if decision.yawn_status else 'Normal'} ({states.confidence_mouth:.2f})",
        f"Head Position: {decision.head_pose_status}",
   
        f"EAR: {ratios.avg_ear:.2f}  MAR: {ratios.mar:.2f}",
        f"Blinks: {ratios.blink_count}",
        f"Blink Rate: {ratios.blink_rate:.1f}  blinks per second",
        f"Yawning Count: {ratios.yawning_count}",
        f"Yawning Rate: {ratios.yawning_rate:.1f} yawns per second",
        
    ]
   
    
    
    # Draw status texts
    y_position = padding + 30
    for text in texts:
        cv2.putText(output, text, (padding + 10, y_position), 
                   font, font_scale, text_color, thickness)
        y_position += 30
    
    # Draw alert indicator
    if alarm_active:
        if alarm_type == "Drowsiness":
            text = "Drowsy"
            color = (0, 0, 255)
        else:
            text = "Distracted"
            color = (0, 255, 255)
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        
     
        x_position = output.shape[1] - text_width - 20
        
        cv2.putText(output, text, (x_position, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.rectangle(output, (0, 0), (frame.shape[1], frame.shape[0]), 
                     color, 10)  
    
    return output


def draw_feature_windows(frame: np.ndarray, facial_features) -> np.ndarray:
    """Draw small windows showing detected features"""

    if facial_features.left_eye is not None:
        # Scale feature images to small size
        feature_size = (100, 100)
        left_eye = cv2.cvtColor(cv2.resize(facial_features.left_eye, feature_size), cv2.COLOR_RGB2BGR)
        right_eye = cv2.cvtColor(cv2.resize(facial_features.right_eye, feature_size), cv2.COLOR_RGB2BGR)
        mouth = cv2.cvtColor(cv2.resize(facial_features.mouth, feature_size), cv2.COLOR_RGB2BGR)

        # Calculate positions (bottom-right corner)
        h, w = frame.shape[:2]
        margin = 10

        # Create regions for feature windows
        frame[
            h - feature_size[1] - margin : h - margin,
            w - 3 * feature_size[0] - 2 * margin : w - 2 * feature_size[0] - 2 * margin,
        ] = left_eye
        frame[
            h - feature_size[1] - margin : h - margin,
            w - 2 * feature_size[0] - margin : w - feature_size[0] - margin,
        ] = right_eye
        frame[h - feature_size[1] - margin : h - margin, w - feature_size[0] : w] = (
            mouth
        )

    return frame


def display_frame(
    frame: np.ndarray,
    face_region,
    facial_features,
    states,
    decision,
    alarm_active,
    alarm_type,
    drowsy_conf,
    flag_normal_state,
    ratios,
) -> np.ndarray:
    """Combine all visualization elements on the frame"""
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.flip(frame, 1)
    if flag_normal_state:
        # Draw status overlay
        output = draw_status_overlay(frame, states, decision,alarm_active,drowsy_conf ,ratios,alarm_type)

        # Draw feature windows
        if facial_features.success:
            output = draw_feature_windows(output, facial_features)
    else:
        output = frame
        # Get the size of the text
        text = "Face not detected"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

        # Calculate x position to align text to the right with some padding (e.g., 20 pixels)
        x_position = output.shape[1] - text_width - 20

        cv2.putText(output, text, (x_position, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(output, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)  # Red border for alert

    return output
