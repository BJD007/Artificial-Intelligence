import os
import pandas as pd
import shutil

# Paths
test_images_dir = '/home/bhaskarhertzwell/Documents/Bhaskar_GITHUB/PhD_Thesis/data/ISIC_2020_Test_Input'
test_output_dir = '/home/bhaskarhertzwell/Documents/Bhaskar_GITHUB/PhD_Thesis/data/test/unknown'

# Create output directory if it doesn't exist
os.makedirs(test_output_dir, exist_ok=True)

# Load the CSV file containing test metadata
test_metadata_df = pd.read_csv('/home/bhaskarhertzwell/Documents/Bhaskar_GITHUB/PhD_Thesis/data/ISIC_2020_Test_Metadata.csv')

# Move test images into the 'unknown' class folder
for _, row in test_metadata_df.iterrows():
    image_name = row['image']
    
    # Construct full image path
    src_image_path = os.path.join(test_images_dir, f"{image_name}.jpg")
    dest_image_path = os.path.join(test_output_dir, f"{image_name}.jpg")
    
    # Move the image to the 'unknown' folder
    if os.path.exists(src_image_path):
        shutil.move(src_image_path, dest_image_path)
        print(f"Moved: {src_image_path} -> {dest_image_path}")
    else:
        print(f"File not found: {src_image_path}")

print("Test images organized into 'unknown' class folder.")
