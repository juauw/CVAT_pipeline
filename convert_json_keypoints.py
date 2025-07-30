# This program takes a file named metadata.py and a .csv or.json file and produces
# a file named converted_keypoints.json which is compatible with the CVAT keypoints import module.
import json
import importlib.util
import argparse
import os
import sys
import csv
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, Any
import cv2

# Loads the metadata.py source file
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

# Builds the COCO header from the metadata.py file
def build_coco_header_from_metadata(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    keypoints = [info["name"] for idx, info in sorted(dataset_info["keypoint_info"].items())]
    name_to_index = {name: idx + 1 for idx, name in enumerate(keypoints)}
    skeleton = []
    for idx in sorted(dataset_info["skeleton_info"].keys()):
        link = dataset_info["skeleton_info"][idx]["link"]
        try:
            start = name_to_index[link[0]]
            end = name_to_index[link[1]]
            skeleton.append([start, end])
        except KeyError as e:
            raise ValueError(f"Keypoint missing in keypoint_info: {e}")
    category = {
        "id": 2,
        "name": dataset_info["classes"][0],
        "supercategory": "object",
        "keypoints": keypoints,
        "skeleton": skeleton,
    }
    return {
        "licenses": [{"id": 0, "name": "", "url": ""}],
        "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""},
        "categories": [category]
    }

# Reads .json annotations
def read_json_annotations(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

# Reads .csv annotations
def read_csv_annotations(csv_path, video_path):
    images, annotations = [], []
    next_ann_id = 0
    frame_to_image_id = {}
    image_id_counter = 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        keypoint_triplets = []
        i = 0
        while f"x{i}" in reader.fieldnames and f"y{i}" in reader.fieldnames and f"score{i}" in reader.fieldnames:
            keypoint_triplets.append((f"x{i}", f"y{i}", f"score{i}"))
            i += 1

        for row in reader:
            frame_id = int(row["frame_id"])
            instance_id = int(row["instance_id"])

            if frame_id not in frame_to_image_id:
                image_id = image_id_counter
                frame_to_image_id[frame_id] = image_id
                image_id_counter += 1
                images.append({"id": image_id, "file_name": f"{frame_id:04d}.jpg", "height": height, "width": width})
            else:
                image_id = frame_to_image_id[frame_id]

            keypoints = []
            for xk, yk, sk in keypoint_triplets:
                try:
                    x, y, score = float(row[xk]), float(row[yk]), float(row[sk])
                    v = 2 if score > 0.5 else 1
                    keypoints.extend([x, y, v])
                except:
                    keypoints.extend([0, 0, 0])

            valid_coords = [(keypoints[i], keypoints[i+1]) for i in range(0, len(keypoints), 3) if keypoints[i+2] != 0]
            bbox = [0, 0, 0, 0]
            if valid_coords:
                xs, ys = zip(*valid_coords)
                min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
                bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

            annotations.append({
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": 2,
                "keypoints": keypoints,
                "num_keypoints": sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] != 0),
                "iscrowd": 0,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "track_id": instance_id
            })
            next_ann_id += 1

    return {"images": images, "annotations": annotations}

# Converts .json annotations to .xml annotations
def convert_to_cvat_xml(annotations, metadata, output_file):
    label_name = metadata['classes'][0]
    tracks = defaultdict(list)
    for ann in annotations:
        tracks[ann['track_id']].append(ann)

    annotations_el = ET.Element("annotations")
    ET.SubElement(annotations_el, "version").text = "1.1"
    meta_el = ET.SubElement(annotations_el, "meta")
    task_el = ET.SubElement(meta_el, "task")
    labels_el = ET.SubElement(task_el, "labels")
    label_el = ET.SubElement(labels_el, "label")
    ET.SubElement(label_el, "name").text = label_name
    attrs_el = ET.SubElement(label_el, "attributes")
    for idx, kp in metadata["keypoint_info"].items():
        attr_el = ET.SubElement(attrs_el, "attribute")
        ET.SubElement(attr_el, "name").text = kp["name"]
        ET.SubElement(attr_el, "input_type").text = "number"

    for track_id, frames in tracks.items():
        track_el = ET.SubElement(annotations_el, "track", id=str(track_id), label=label_name, source="manual")
        for ann in sorted(frames, key=lambda x: x["image_id"]):
            points_str = ";".join(f"{ann['keypoints'][i]:.2f},{ann['keypoints'][i+1]:.2f}" for i in range(0, len(ann["keypoints"]), 3))
            ET.SubElement(track_el, "points", frame=str(ann["image_id"]), outside="0", occluded="0", keyframe="1", points=points_str)

    tree = ET.ElementTree(annotations_el)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"CVAT 1.1 XML file generated successfully: '{output_file}'")

def get_output_path(input_path: str, output_format: str) -> str:
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    return f"{base_name}_converted_keypoints.{output_format}"

# Main
def main():
    parser = argparse.ArgumentParser(description="Convert CSV/JSON annotations to COCO or CVAT format.")
    parser.add_argument("input_path", help="Path to input .csv or .json")
    parser.add_argument("video_path", help="Path to video file to obtain resolution")
    parser.add_argument("metadata", help="Path to metadata.py file")
    parser.add_argument("--format", choices=["json", "xml"], default="json", help="Output format (default: json)")
    args = parser.parse_args()

    dataset_info = load_metadata(args.metadata)
    header = build_coco_header_from_metadata(dataset_info)

    if args.input_path.endswith(".json"):
        original = read_json_annotations(args.input_path)
    elif args.input_path.endswith(".csv"):
        original = read_csv_annotations(args.input_path, args.video_path)
    else:
        raise ValueError("Unsupported input format. Use .json or .csv")

    output_path = "converted_keypoints.json"

    if args.format == "json":
        new_data = {
            "licenses": header["licenses"],
            "info": header["info"],
            "categories": header["categories"],
            "images": original.get("images", []),
            "annotations": original.get("annotations", []),
        }
        with open(output_path, "w", encoding="utf-8") as f_out:
            json.dump(new_data, f_out, ensure_ascii=False, indent=2)
        print(f"COCO Keypoints 1.0 file generated successfully: '{output_path}'")
    elif args.format == "xml":
        convert_to_cvat_xml(original["annotations"], dataset_info, output_path)

if __name__ == "__main__":
    main()

