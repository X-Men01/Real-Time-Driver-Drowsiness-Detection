#extract image from labeled datasets
import cv2
import os

images_path = ""
annotations_path = ""
output_path = ""
os.makedirs(output_path, exist_ok=True)

def crop_bounding_boxes(image_path, annotation_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    height, width, _ = image.shape

    with open(annotation_path, "r") as file:
        for line in file:
            parts = line.split()
            class_id = int(parts[0])  # Class ID
            x_center, y_center, bbox_width, bbox_height = map(float, parts[-4:])  # Bounding box info

            # Convert normalized coordinates to pixel coordinates
            x_min = int((x_center - bbox_width / 2) * width)
            y_min = int((y_center - bbox_height / 2) * height)
            x_max = int((x_center + bbox_width / 2) * width)
            y_max = int((y_center + bbox_height / 2) * height)

            # Clamp coordinates to image dimensions
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_max)
            y_max = min(height, y_max)

            # Skip invalid bounding boxes
            if x_min >= x_max or y_min >= y_max:
                print(f"Invalid bounding box for {image_path}: {x_min}, {y_min}, {x_max}, {y_max}")
                continue

            # Crop the bounding box
            cropped_image = image[y_min:y_max, x_min:x_max]
            if cropped_image.size == 0:
                print(f"Empty cropped image for {image_path}: {x_min}, {y_min}, {x_max}, {y_max}")
                continue

            # Create directory for the class
            class_dir = os.path.join(output_path, f"class_{class_id}")
            os.makedirs(class_dir, exist_ok=True)

            # Save the cropped image
            output_file = os.path.join(class_dir, f"{os.path.basename(image_path).split('.')[0]}_{class_id}.jpg")
            cv2.imwrite(output_file, cropped_image)

# Process all files
for annotation_file in os.listdir(annotations_path):
    if annotation_file.endswith(".txt"):
        image_file = annotation_file.replace(".txt", ".jpg")
        crop_bounding_boxes(
            os.path.join(images_path, image_file),
            os.path.join(annotations_path, annotation_file),
        )
