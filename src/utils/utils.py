import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from collections import Counter
from pathlib import Path


def display_images(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()
    # how to use it
    # example_img = next(iter(train_dataloader))[0]
    # img_grid = torchvision.utils.make_grid(example_img)
    # display_images(img_grid)







def organize_dataset(source_path, dest_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Create directory structure
    for split in ['train', 'test', 'val']:
        for class_name in ['Close_Eye', 'Open_Eye']:
            os.makedirs(os.path.join(dest_path, split, class_name), exist_ok=True)
    
    # Collect all image paths and their labels
    images = []
    labels = []
    for class_name in ['Close_Eye', 'Open_Eye']:
        class_path = os.path.join(source_path, 'Eyes_Dataset', class_name)
        class_images = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        images.extend([(class_name, img) for img in class_images])
        labels.extend([class_name] * len(class_images))
    
    # Print initial class distribution
    print("Initial class distribution:")
    print(Counter(labels))
    
    # First split: train vs rest
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1-train_ratio, random_state=42)
    indices = np.arange(len(images))
    
    for train_idx, temp_idx in sss1.split(indices, labels):
        train_indices = train_idx
        temp_indices = temp_idx
    
    # Second split: val vs test from the remaining data
    remaining_labels = [labels[i] for i in temp_indices]
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1-val_test_ratio, random_state=42)
    for val_idx, test_idx in sss2.split(temp_indices, remaining_labels):
        val_indices = temp_indices[val_idx]
        test_indices = temp_indices[test_idx]
    
    # Copy files to their respective directories
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    for split_name, indices in splits.items():
        split_images = [images[i] for i in indices]
        print(f"\n{split_name.upper()} set distribution:")
        print(Counter([labels[i] for i in indices]))
        
        for class_name, image_name in split_images:
            src = os.path.join(source_path, 'Eyes_Dataset', class_name, image_name)
            new_name = f"{os.path.splitext(image_name)[0]}_{class_name.lower()}{os.path.splitext(image_name)[1]}"
            dst = os.path.join(dest_path, split_name, class_name, new_name)
            shutil.copy2(src, dst)
    # how to use it, change the source and destination paths and the class names
    # if __name__ == "__main__":
    #     source_path = "datasets/processed"
    #     destination_path = "datasets/eye_dataset_stratified"   
    #     organize_dataset(source_path, destination_path)
    


def move_images(image_list, destination_dir):
    """
    Move a list of images to a new directory.

    Parameters:
    - image_list: List of image file paths (as strings or Path objects).
    - destination_dir: The directory to move the images to (as a string or Path object).
    """
    # Convert destination_dir to a Path object
    destination_dir = Path(destination_dir)
    
    # Create the destination directory if it doesn't exist
    destination_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_list:
        # Convert img_path to a Path object if it's a string
        img_path = Path(img_path)
        
        # Check if the image exists
        if img_path.exists() and img_path.is_file():
            try:
                # Move the image to the destination directory
                shutil.move(str(img_path), destination_dir / img_path.name)
                print(f"Moved: {img_path.name} to {destination_dir}")
            except Exception as e:
                print(f"Error moving {img_path.name}: {e}")
        else:
            print(f"File does not exist: {img_path}")

# Example usage
# image_list = ["path/to/image1.jpg", "path/to/image2.jpg"]
# move_images(image_list, "path/to/new_directory")

def move_specified_files(file_names, source_dir, destination_dir):
    """
    Move specified images and their corresponding labels to new directories.

    Parameters:
    file_names (list): List of image file names.
    source_dir (str): Base directory containing the original images and labels.
    destination_dir (str): Base directory where files will be moved.
    """
    # Define source and destination directories for images and labels
    source_images_dir = os.path.join(source_dir, 'images')
    
    
    destination_images_dir = os.path.join(destination_dir, 'images')
  
    
    # Create destination directories if they don't exist
    if not os.path.exists(destination_images_dir):
        os.makedirs(destination_images_dir)
        
    

    # Move specified images and their corresponding labels
    for file_name in file_names:
        # Source paths
        source_image_path = os.path.join(source_images_dir, file_name)
        label_name = os.path.splitext(file_name)[0] + '.txt'
        
        
        # Destination paths
        destination_image_path = os.path.join(destination_images_dir, file_name)
        
        
        # Move image if it exists
        if os.path.exists(source_image_path):
            shutil.move(source_image_path, destination_image_path)
        
       

    print("Specified images moved successfully.")


