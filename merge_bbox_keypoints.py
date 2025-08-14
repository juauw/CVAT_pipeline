# This script takes a corrected COCO keypoints 1.0 .json file and merges it with a 
# corrected COCO 1.0 file and makes it compatible with the precision track pipeline.
import zipfile
import os
import shutil
import json
from collections import defaultdict

# Paths
zip_path = "corrected_bbox.zip"
extract_dir = "corrected_bbox"
output_filename = "corrected_bbox.json"

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
import json
from collections import defaultdict

# Loads the two .json files
with open("corrected_keypoints.json", "r") as f:
    keypoints_data = json.load(f)

with open("corrected_bbox.json", "r") as f:
    bbox_data = json.load(f)

# Group the annotations by image_id
annotations_by_image = defaultdict(lambda: {"keypoints": [], "bboxes": []})

for ann in keypoints_data.get("annotations", []):
    annotations_by_image[ann["image_id"]]["keypoints"].append(ann)

for ann in bbox_data.get("annotations", []):
    annotations_by_image[ann["image_id"]]["bboxes"].append(ann)

# Merges the annotations by image_id
merged_annotations = []
annotation_id = 1

for image_id, anns in annotations_by_image.items():
    for kp_ann in anns["keypoints"]:
        matching_bboxes = [
            bbox for bbox in anns["bboxes"]
            if bbox["category_id"] == kp_ann["category_id"]
        ]
        for bbox_ann in matching_bboxes:
            merged_ann = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": kp_ann["category_id"],
                "bbox": bbox_ann["bbox"],
                "area": bbox_ann.get("area", 0),
                "segmentation": bbox_ann.get("segmentation", []),
                "iscrowd": bbox_ann.get("iscrowd", 0),
                "keypoints": kp_ann["keypoints"],
                "num_keypoints": kp_ann.get("num_keypoints", 0)
            }
            merged_annotations.append(merged_ann)
            annotation_id += 1

# Create a dictionary without a COCO header
merged_data = {
    "images": keypoints_data.get("images", []),
    "annotations": merged_annotations
}

# Saves the merged file without the COCO header
with open("combined.json", "w") as f:
    json.dump(merged_data, f, indent=2)

