#This script converts a COCO Keypoints 1.0 file to a COCO 1.0 file by estimating
# bounding boxes based on the modified keypoints. After the script creates a new task
# and automatically uploads the images and annotations to CVAT. It can also directly
# upload bounding boxes to CVAT using the COCO 1.0 or CVAT 1.1 format.
import argparse
import os
import json
from pathlib import Path
from cvat_sdk import make_client
from cvat_sdk.core.helpers import DeferredTqdmProgressReporter
import zipfile
import shutil
import cv2

# Auto-handle corrected_keypoints.zip if present
zip_path = "corrected_keypoints.zip"
extract_dir = "corrected_keypoints"
output_filename = "corrected_keypoints.json"

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    os.remove(zip_path)
    print(f"Deleted zip file: {zip_path}")

    annotations_dir = os.path.join(extract_dir, "annotations")
    old_path = os.path.join(annotations_dir, "person_keypoints_default.json")
    new_path = os.path.join(".", output_filename)

    if os.path.exists(old_path):
        shutil.move(old_path, new_path)
        print(f"Moved and renamed file to: {new_path}")
    else:
        print(f"Error: {old_path} not found.")

    shutil.rmtree(extract_dir)
    print(f"Deleted extracted folder: {extract_dir}")

def create_bbox_label(label_name):
    return [{
        "name": label_name,
        "type": "rectangle",
        "color": "#FF0000",
        "attributes": []
    }]

# Converts COCO Keypoints 1.0 to COCO 1.0 bounding boxes
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

# Cuts the video into its component frames
def video_to_frames(video_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frame_count = 0
    success, frame = cap.read()

    while success:
        frame_name = output_dir / f"{frame_count:04d}.jpg"
        cv2.imwrite(str(frame_name), frame)
        frame_count += 1
        success, frame = cap.read()

    cap.release()
    print(f"Extracted {frame_count} frames to {output_dir}")
    return sorted(output_dir.glob("*.jpg"))

# Main
def main():
    parser = argparse.ArgumentParser(description="Convert COCO Keypoints to Bounding Boxes and import to CVAT.")
    parser.add_argument("--host", default="http://localhost:8080", help="CVAT server URL (e.g. http://localhost:8080)")
    parser.add_argument("--username", required=True, help="CVAT username")
    parser.add_argument("--password", required=True, help="CVAT password")
    parser.add_argument("--task-name", required=True, help="Task name to create in CVAT")
    parser.add_argument("--label-name", default="mouse", help="Label name to assign to bounding boxes")
    parser.add_argument("--folder", required=True, help="Path to folder of images OR a single video file")
    parser.add_argument("--input", required=True, help="Path to input COCO Keypoints JSON or CVAT 1.1/XML with bboxes")
    parser.add_argument("--output-json", required=True, help="Path to output COCO Bounding Box JSON (ignored if XML input)")
    args = parser.parse_args()

    # Detect if --folder is a video; if so, extract frames and remember to clean up
    folder_path = Path(args.folder)
    frames_dir = None
    cleanup_frames_dir = False

    if folder_path.is_file() and folder_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"]:
        print(f"Video detected: {folder_path}, converting to frames...")
        frames_dir = folder_path.parent / f"{folder_path.stem}_frames"
        video_to_frames(folder_path, frames_dir)
        image_folder = str(frames_dir)
        cleanup_frames_dir = True  # mark for deletion after upload
    else:
        image_folder = args.folder

    # If input is CVAT XML with bboxes, skip conversion
    skip_conversion = os.path.splitext(args.input)[1].lower() in [".xml"]
    if skip_conversion:
        print("BBox XML detected — skipping keypoint-to-bbox conversion.")
        output_for_import = args.input  # import XML directly
        import_format = "CVAT 1.1"
    else:
        print("Converting keypoints to bounding boxes...")
        with open(args.input, "r") as f:
            data = json.load(f)

        bbox_annotations = []
        for ann in data.get("annotations", []):
            kp = ann.get("keypoints")
            if not kp:
                continue
            bbox = keypoints_to_bboxes(kp, ann["image_id"], ann["category_id"], ann["id"])
            if bbox:
                bbox_annotations.append(bbox)

        coco_bbox = {
            "info": data.get("info", {}),
            "licenses": data.get("licenses", []),
            "images": data.get("images", []),
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

        output_for_import = args.output_json
        import_format = "COCO 1.0"

    # Create task, upload, import annotations
    try:
        with make_client(args.host, credentials=(args.username, args.password)) as client:
            image_paths = [
                os.path.join(image_folder, f)
                for f in os.listdir(image_folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            if not image_paths:
                raise RuntimeError(f"No images found in {image_folder}")

            task = client.tasks.create_from_data(
                spec={"name": args.task_name, "labels": create_bbox_label(args.label_name)},
                resources=image_paths,
                pbar=DeferredTqdmProgressReporter()
            )
            print(f"Task created with ID: {task.id}")

            task.import_annotations(
                format_name=import_format,
                filename=output_for_import,
                pbar=DeferredTqdmProgressReporter()
            )
            print("Bounding box annotations imported successfully.")
    finally:
        # Clean up extracted frames if we generated them
        if cleanup_frames_dir and frames_dir and frames_dir.exists():
            try:
                shutil.rmtree(frames_dir)
                print(f"Cleaned up extracted frames folder: {frames_dir}")
            except Exception as e:
                print(f"Warning: failed to remove frames folder {frames_dir}: {e}")

if __name__ == "__main__":
    main()






