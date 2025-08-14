# Cleans up the export files from CVAT
import zipfile
import os
import shutil

# Paths
zip_path = "corrected_keypoints.zip"
extract_dir = "corrected_keypoints"
output_filename = "corrected_keypoints.json"

# Step 1: Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Step 2: Delete the zip file
if os.path.exists(zip_path):
    os.remove(zip_path)
    print(f"Deleted zip file: {zip_path}")

# Step 3: Rename and move JSON file to current directory
annotations_dir = os.path.join(extract_dir, "annotations")
old_path = os.path.join(annotations_dir, "person_keypoints_default.json")
new_path = os.path.join(".", output_filename)

if os.path.exists(old_path):
    shutil.move(old_path, new_path)
    print(f"Moved and renamed file to: {new_path}")
else:
    print(f"Error: {old_path} not found.")

# Step 4: Delete the extracted folder and all its contents
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)
    print(f"Deleted extracted folder: {extract_dir}")

