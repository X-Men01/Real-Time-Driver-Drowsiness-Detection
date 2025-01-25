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
   
    axs[1].imshow(face_region)
    axs[1].set_title("Detected Face")
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

