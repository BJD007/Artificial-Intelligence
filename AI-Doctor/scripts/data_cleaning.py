import os
import pandas as pd

# Paths to your dataset
train_images_dir = '/home/bhaskarhertzwell/Documents/Bhaskar_GITHUB/PhD_Thesis/data/train'
duplicate_list_path = '/home/bhaskarhertzwell/Documents/Bhaskar_GITHUB/PhD_Thesis/ISIC_2020_Training_Duplicates.csv'

# Load the duplicate list
duplicate_list = pd.read_csv(duplicate_list_path)

# Debug: Print DataFrame columns to verify the column names
print("Columns in duplicate list DataFrame:", duplicate_list.columns)

# Remove duplicates from the training set
for _, row in duplicate_list.iterrows():
    # Access image names from both columns
    image_name_1 = row['image_name_1']
    image_name_2 = row['image_name_2']
    
    # Construct paths for both images
    image_path_1 = os.path.join(train_images_dir, image_name_1 + '.jpg')  # Assuming images have a .jpg extension
    image_path_2 = os.path.join(train_images_dir, image_name_2 + '.jpg')  # Adjust the extension as needed
    
    # Check and remove if image 1 exists
    if os.path.exists(image_path_1):
        os.remove(image_path_1)
        print(f"Removed duplicate image: {image_name_1}")
    else:
        print(f"Image not found: {image_name_1}")
    
    # Check and remove if image 2 exists
    if os.path.exists(image_path_2):
        os.remove(image_path_2)
        print(f"Removed duplicate image: {image_name_2}")
    else:
        print(f"Image not found: {image_name_2}")

print("Duplicate images removed.")
