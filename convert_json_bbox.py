# This program takes two files that are named metadata.py and train.json and produces
# a file named converted_bbox.json which is compatible with the CVAT bounding box import module.
import json
import importlib.util
import argparse
import os
import sys
from typing import Dict, Any
import pandas as pd
import cv2

# Loads the metadata.py data
def load_metadata(metadata_path: str) -> Dict[str, Any]:
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    spec = importlib.util.spec_from_file_location("metadata", metadata_path)
    metadata = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(metadata)
    except Exception as e:
        raise ImportError(f"Could not import metadata.py: {e}")
    return metadata.dataset_info

# Extract width/height from video file
def get_video_resolution(video_path: str) -> tuple[int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

# Creates a .json file with only the bounding box information (from COCO keypoints JSON)
def convert_to_coco_bbox_only(input_path: str, output_path: str, metadata_path: str):
    try:
        dataset_info = load_metadata(metadata_path)
        class_name = dataset_info["classes"][0]
    except Exception as e:
        print(f"Error loading metadata: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(input_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading '{input_path}': {e}", file=sys.stderr)
        sys.exit(1)

    images = data.get("images", [])
    annotations = data.get("annotations", [])

    coco_output = {
        "licenses": [{"id": 0, "name": "", "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": "COCO 1.0",
            "url": "",
            "version": "1.0",
            "year": ""
        },
        "categories": [
            {
                "id": 1,
                "name": class_name,
                "supercategory": "object"
            }
        ],
        "images": [],
        "annotations": []
    }

    next_ann_id = 0
    for ann in annotations:
        if "bbox" not in ann or "image_id" not in ann:
            continue

        bbox = ann["bbox"]
        image_id = ann["image_id"]

        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        area = ann.get("area", bbox[2] * bbox[3])

        coco_output["annotations"].append({
            "id": next_ann_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": bbox,
            "area": area,
            "iscrowd": ann.get("iscrowd", 0)
        })
        next_ann_id += 1

    try:
        with open(output_path, "w", encoding="utf-8") as f_out:
            json.dump(coco_output, f_out, ensure_ascii=False, indent=2)
        print(f"COCO 1.0 file generated: '{output_path}'")
    except Exception as e:
        print(f"Error writing to '{output_path}': {e}", file=sys.stderr)
        sys.exit(1)

# CSV to COCO 1.0 bbox conversion
def convert_csv_to_coco_bbox_only(input_csv: str, output_json: str, metadata_path: str, video_path: str):
    try:
        dataset_info = load_metadata(metadata_path)
        class_name = dataset_info["classes"][0]
    except Exception as e:
        print(f"Error loading metadata: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading CSV '{input_csv}': {e}", file=sys.stderr)
        sys.exit(1)

    try:
        width, height = get_video_resolution(video_path)
    except Exception as e:
        print(f"Error getting video resolution: {e}", file=sys.stderr)
        sys.exit(1)

    coco_output = {
        "licenses": [{"id": 0, "name": "", "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": "COCO 1.0",
            "url": "",
            "version": "1.0",
            "year": ""
        },
        "categories": [
            {
                "id": 1,
                "name": class_name,
                "supercategory": "object"
            }
        ],
        "images": [],
        "annotations": []
    }

    frame_to_image_id = {}
    image_id_counter = 0
    ann_id_counter = 0

    for _, row in df.iterrows():
        frame_id = int(row["frame_id"])
        if frame_id not in frame_to_image_id:
            frame_to_image_id[frame_id] = image_id_counter
            coco_output["images"].append({
                "id": image_id_counter,
                "file_name": f"{frame_id}.jpg",
                "width": width,
                "height": height
            })
            image_id_counter += 1

        image_id = frame_to_image_id[frame_id]
        cx, cy, w, h = row["cx"], row["cy"], row["w"], row["h"]
        x = cx - w / 2
        y = cy - h / 2

        coco_output["annotations"].append({
            "id": ann_id_counter,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0
        })
        ann_id_counter += 1

    try:
        with open(output_json, "w", encoding="utf-8") as f_out:
            json.dump(coco_output, f_out, ensure_ascii=False, indent=2)
        print(f"COCO 1.0 file generated: '{output_json}'")
    except Exception as e:
        print(f"Error writing to '{output_json}': {e}", file=sys.stderr)
        sys.exit(1)

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert input file (COCO JSON or CSV) to COCO 1.0 bounding box format.")
    parser.add_argument("input_file", help="Path to input file (.json or .csv)")
    parser.add_argument("output_json", help="Path to output COCO 1.0 bounding box JSON")
    parser.add_argument("metadata", help="Path to metadata.py file")
    parser.add_argument("--video", help="Path to video file (required for .csv input)", required=False)

    args = parser.parse_args()
    ext = os.path.splitext(args.input_file)[1].lower()

    if ext == ".json":
        convert_to_coco_bbox_only(args.input_file, args.output_json, args.metadata)
    elif ext == ".csv":
        if not args.video:
            print("Error: --video is required when input is a .csv file.", file=sys.stderr)
            sys.exit(1)
        convert_csv_to_coco_bbox_only(args.input_file, args.output_json, args.metadata, args.video)
    else:
        print("Error: Unsupported input file format. Only .json and .csv are supported.", file=sys.stderr)
        sys.exit(1)


