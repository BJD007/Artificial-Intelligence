import os
import pandas as pd
import shutil

# Paths
images_dir = '/home/bhaskarhertzwell/Documents/Bhaskar_GITHUB/PhD_Thesis/data/ISIC_2020_Training_JPEG'
labels_csv_path = '/home/bhaskarhertzwell/Documents/Bhaskar_GITHUB/PhD_Thesis/data/ISIC_2020_Training_GroundTruth_v2.csv'
train_output_dir = '/home/bhaskarhertzwell/Documents/Bhaskar_GITHUB/PhD_Thesis/data/train'

# Create output directories if they don't exist
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(os.path.join(train_output_dir, 'benign'), exist_ok=True)
os.makedirs(os.path.join(train_output_dir, 'malignant'), exist_ok=True)

# Load the CSV file containing labels
labels_df = pd.read_csv(labels_csv_path)

# Get the list of actual files in the directory
actual_files = set(f.lower() for f in os.listdir(images_dir))  # Use lower() for case-insensitivity

# Organize images into class subfolders
for _, row in labels_df.iterrows():
    image_name = row['image_name']
    target = row['target']
    
    # Determine the class folder name
    class_folder = 'benign' if target == 0 else 'malignant'
    
    # Construct possible image paths
    image_filename = f"{image_name}.jpg"
    image_filename_lower = image_filename.lower()  # Lowercase for case-insensitive comparison
    
    if image_filename_lower in actual_files:
        src_image_path = os.path.join(images_dir, image_filename)
        dest_image_path = os.path.join(train_output_dir, class_folder, image_filename)
        
        # Move the image to the corresponding class folder
        shutil.move(src_image_path, dest_image_path)
        print(f"Moved: {src_image_path} -> {dest_image_path}")
    else:
        print(f"File not found: {image_filename}")

print("Images organized into class subfolders.")
