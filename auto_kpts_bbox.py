#This script converts a COCO Keypoints 1.0 file to a COCO 1.0 file by estimating
# bounding boxes based on the modified keypoints. After the script creates a new task
# and automatically uploads the images and annotations to CVAT. It can also directly
# upload bounding boxes to CVAT using the COCO 1.0 or CVAT 1.1 format.
import argparse
import os
import json
from cvat_sdk import make_client
from cvat_sdk.core.helpers import DeferredTqdmProgressReporter
import zipfile
import shutil
import cv2

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

def create_bbox_label(label_name):
    return [{
        "name": label_name,
        "type": "rectangle",
        "color": "#FF0000",
        "attributes": []
    }]

def keypoints_to_bboxes(keypoints, image_id, category_id, annotation_id):
    x_coords = keypoints[0::3]
    y_coords = keypoints[1::3]
    v_flags = keypoints[2::3]

    visible_x = [x for x, v in zip(x_coords, v_flags) if v > 0]
    visible_y = [y for y, v in zip(y_coords, v_flags) if v > 0]

    if not visible_x or not visible_y:
        return None

    x_min = min(visible_x)
    y_min = min(visible_y)
    x_max = max(visible_x)
    y_max = max(visible_y)

    width = x_max - x_min
    height = y_max - y_min

    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x_min, y_min, width, height],
        "area": width * height,
        "iscrowd": 0,
        "id": annotation_id
    }

# Cuts the video into it's component frames
def video_to_frames(video_path, output_dir):
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    success, frame = cap.read()

    while success:
        frame_name = output_dir / f"{frame_count:04d}.jpg"
        cv2.imwrite(str(frame_name), frame)
        frame_count += 1
        success, frame = cap.read()

    cap.release()
    return sorted(output_dir.glob("*.jpg"))

# Main
def main():
    parser = argparse.ArgumentParser(description="Convert COCO Keypoints to Bounding Boxes and import to CVAT.")
    parser.add_argument("--host", default="http://localhost:8080", help="CVAT server URL (e.g. http://localhost:8080)")
    parser.add_argument("--username", required=True, help="CVAT username")
    parser.add_argument("--password", required=True, help="CVAT password")
    parser.add_argument("--task-name", required=True, help="Task name to create in CVAT")
    parser.add_argument("--label-name", default="mouse", help="Label name to assign to bounding boxes")
    parser.add_argument("--folder", required=True, help="Path to folder containing task images or videos")
    parser.add_argument("--input", required=True, help="Path to input COCO Keypoints or Bounding Box XML")
    parser.add_argument("--output-json", required=True, help="Path to output COCO Bounding Box JSON (ignored if already in bbox format)")
    args = parser.parse_args()

    skip_conversion = os.path.basename(args.input) == "converted_bbox.xml"
    if skip_conversion:
        print("Detected 'converted_bbox.xml' — skipping keypoint-to-bbox conversion.")
        args.output_json = args.input_json
    else:
        print("Converting keypoints to bounding boxes...")
        with open(args.input_json, "r") as f:
            data = json.load(f)

        bbox_annotations = []
        for ann in data["annotations"]:
            bbox = keypoints_to_bboxes(ann["keypoints"], ann["image_id"], ann["category_id"], ann["id"])
            if bbox:
                bbox_annotations.append(bbox)

        coco_bbox = {
            "info": data.get("info", {}),
            "licenses": data.get("licenses", []),
            "images": data["images"],
            "categories": [{
                "id": data["categories"][0]["id"],
                "name": data["categories"][0]["name"],
                "supercategory": data["categories"][0].get("supercategory", "")
            }],
            "annotations": bbox_annotations
        }

        with open(args.output_json, "w") as f:
            json.dump(coco_bbox, f)
        print(f"Bounding boxes extracted and saved to {args.output_json}")

    # Step 2: create task and upload
    with make_client(args.host, credentials=(args.username, args.password)) as client:
        image_paths = [
            os.path.join(args.image_folder, f)
            for f in os.listdir(args.image_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        task = client.tasks.create_from_data(
            spec={"name": args.task_name, "labels": create_bbox_label(args.label_name)},
            resources=image_paths,
            pbar=DeferredTqdmProgressReporter()
        )
        print(f"Task created with ID: {task.id}")

        task.import_annotations(
            format_name="CVAT 1.1",
            filename=args.output_json,
            pbar=DeferredTqdmProgressReporter()
        )
        print("Bounding box annotations imported successfully.")

if __name__ == "__main__":
    main()





