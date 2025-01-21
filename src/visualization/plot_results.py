import matplotlib.pyplot as plt
import cv2

def plot_pipeline(original_frame, face_region, left_eye_region, right_eye_region, mouth_region, decision):
    # Create a figure with subplots
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))

    # Plot original frame
    axs[0].imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Frame")
    axs[0].axis('off')
    # plt.show()
    # Plot face region
   
    axs[1].imshow(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Detected Face")
    axs[1].axis('off')

    # Plot left eye region
    axs[2].imshow(cv2.cvtColor(left_eye_region, cv2.COLOR_BGR2RGB))
    axs[2].set_title("Extracted Left Eye")
    axs[2].axis('off')
    
    # Plot right eye region
    axs[3].imshow(cv2.cvtColor(right_eye_region, cv2.COLOR_BGR2RGB))
    axs[3].set_title("Extracted Right Eye")
    axs[3].axis('off')

    # Plot mouth region
    axs[4].imshow(cv2.cvtColor(mouth_region, cv2.COLOR_BGR2RGB))
    axs[4].set_title("Extracted Mouth")
    axs[4].axis('off')

    # Show decision
    plt.suptitle(f"Decision: {decision}", fontsize=16)

    # Display the plot
    plt.show()

