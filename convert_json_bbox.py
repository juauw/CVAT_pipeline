# This program takes two files that are named metadata.py and .json or .csv file and produces
# a file named converted_bbox.json which is compatible with the CVAT bounding box import module.
import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict, namedtuple
from typing import Dict, Any, List, Tuple, Optional
from xml.etree.ElementTree import Element, SubElement, ElementTree

try:
    import cv2
except Exception:
    cv2 = None

# Metadata loader
def load_metadata(metadata_path: str) -> Dict[str, Any]:
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    spec = importlib.util.spec_from_file_location("metadata", metadata_path)
    metadata = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(metadata)
    except Exception as e:
        raise ImportError(f"Could not import metadata.py: {e}")
    if not hasattr(metadata, "dataset_info"):
        raise ValueError("metadata.py must define 'dataset_info'")
    return metadata.dataset_info


# Helpers
ImageInfo = namedtuple("ImageInfo", ["frame_id", "width", "height"])

def read_video_resolution(video_path: str) -> Tuple[int, int]:
    if cv2 is None:
        raise RuntimeError("OpenCV not available; cannot read video resolution.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def ensure_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return int(float(x))

def coco_keypoints_to_pairs(kpts: List[float], k: int) -> List[Tuple[float, float, float]]:
    """
    COCO keypoints: [x0, y0, v0, x1, y1, v1, ...]
    Return list of (x, y, v/score) length K.
    """
    out = []
    n = min(len(kpts) // 3, k)
    for i in range(n):
        x = float(kpts[3*i + 0])
        y = float(kpts[3*i + 1])
        v = float(kpts[3*i + 2])
        out.append((x, y, v))
    # If shorter, pad
    for _ in range(k - n):
        out.append((0.0, 0.0, 0.0))
    return out

def flatten_points_xy(pairs: List[Tuple[float, float, float]]) -> str:
    """
    Format: 'x0,y0;x1,y1;...'
    """
    return ";".join(f"{x:.3f},{y:.3f}" for (x, y, _v) in pairs)

# CSV reader
def parse_csv(
    csv_path: str,
    keypoint_count: int,
) -> Tuple[Dict[int, Dict[int, List[Tuple[float, float, float]]]], Dict[int, set], set]:
    """
    Returns:
      data[frame_id][instance_id] = list[(x,y,score)] length=keypoint_count
      classes_by_instance[instance_id] = set of class_ids seen (should be single value)
      frames_seen = set of frame_ids
    """
    data: Dict[int, Dict[int, List[Tuple[float, float, float]]]] = defaultdict(dict)
    classes_by_instance: Dict[int, set] = defaultdict(set)
    frames_seen = set()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")
        headers = reader.fieldnames or []
        # Expect x0,y0,score0 ... up to keypoint_count-1
        for row in reader:
            frame_id = ensure_int(row["frame_id"])
            class_id = ensure_int(row["class_id"])
            instance_id = ensure_int(row["instance_id"])
            frames_seen.add(frame_id)
            classes_by_instance[instance_id].add(class_id)

            pts: List[Tuple[float, float, float]] = []
            for i in range(keypoint_count):
                x = float(row.get(f"x{i}", 0) or 0)
                y = float(row.get(f"y{i}", 0) or 0)
                s = float(row.get(f"score{i}", 0) or 0)
                pts.append((x, y, s))
            data[frame_id][instance_id] = pts
    return data, classes_by_instance, frames_seen

# JSON reader
def parse_json(
    json_path: str,
    keypoint_count: int,
) -> Tuple[Dict[int, Dict[int, List[Tuple[float, float, float]]]], Dict[int, set], Dict[int, ImageInfo], set]:
    """
    Returns:
      data[frame_id][instance_id] = list[(x,y,score)] length=keypoint_count
      classes_by_instance[instance_id] = set of category_id seen
      images_info[frame_id] = ImageInfo(frame_id, width, height)
      frames_seen = set of frame_ids
    """
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Build image_id -> (frame_id, w, h)
    images = obj.get("images", [])
    id_to_frame: Dict[int, int] = {}
    images_info: Dict[int, ImageInfo] = {}
    for im in images:
        img_id = ensure_int(im["id"])
        if "frame_id" in im:
            frame_id = ensure_int(im["frame_id"])
        else:
            frame = str(im.get("file_name", ""))
            frame_id = ensure_int(im.get("frame_index", 0)) if "frame_index" in im else img_id
        w = ensure_int(im.get("width", 0))
        h = ensure_int(im.get("height", 0))
        id_to_frame[img_id] = frame_id
        images_info[frame_id] = ImageInfo(frame_id, w, h)

    # Build data
    data: Dict[int, Dict[int, List[Tuple[float, float, float]]]] = defaultdict(dict)
    classes_by_instance: Dict[int, set] = defaultdict(set)
    frames_seen = set()

    annotations = obj.get("annotations", [])
    for ann in annotations:
        img_id = ensure_int(ann["image_id"])
        frame_id = id_to_frame.get(img_id, img_id)
        frames_seen.add(frame_id)

        # track id: prefer ann.get('track_id') else 'instance_id' else 'id'
        if "track_id" in ann:
            instance_id = ensure_int(ann["track_id"])
        elif "instance_id" in ann:
            instance_id = ensure_int(ann["instance_id"])
        else:
            instance_id = ensure_int(ann.get("id", 0))

        cat_id = ensure_int(ann.get("category_id", 0))
        classes_by_instance[instance_id].add(cat_id)

        kpts = ann.get("keypoints", [])
        pts = coco_keypoints_to_pairs(kpts, keypoint_count)
        data[frame_id][instance_id] = pts

    return data, classes_by_instance, images_info, frames_seen


# XML writer (CVAT 1.1)
def build_cvat_xml(
    data: Dict[int, Dict[int, List[Tuple[float, float, float]]]],
    classes_by_instance: Dict[int, set],
    class_id_to_name: Dict[int, str],
    frames_sorted: List[int],
    width: Optional[int],
    height: Optional[int],
    task_name: str = "converted",
) -> ElementTree:
    """
    Build a minimal CVAT 1.1 XML with tracks of <points>.
    Tracks are keyed by instance_id; each frame gets a <points> with all keypoints.
    """
    root = Element("annotations")

    version = SubElement(root, "version")
    version.text = "1.1"

    meta = SubElement(root, "meta")
    task = SubElement(meta, "task")
    name_el = SubElement(task, "name")
    name_el.text = task_name

    if width is not None and height is not None:
        original_size = SubElement(task, "original_size")
        SubElement(original_size, "width").text = str(width)
        SubElement(original_size, "height").text = str(height)

    # Build tracks: one per instance_id
    # Map instance_id -> (label name, ordered list of (frame_id -> points))
    instance_ids = sorted({iid for frame in data.values() for iid in frame.keys()})
    for tidx, instance_id in enumerate(instance_ids):
        class_ids = sorted(classes_by_instance.get(instance_id, {0}))
        class_name = class_id_to_name.get(class_ids[0], class_id_to_name.get(0, "object"))

        track = SubElement(root, "track", id=str(instance_id), label=class_name)
        track.set("source", "manual")

        for frame_id in frames_sorted:
            frame_map = data.get(frame_id, {})
            if instance_id not in frame_map:
                continue
            pts = frame_map[instance_id]
            points_attr = flatten_points_xy(pts)

            SubElement(
                track,
                "points",
                frame=str(frame_id),
                outside="0",
                occluded="0",
                keyframe="1",
                points=points_attr,
            )

    return ElementTree(root)


# Main
def main():
    ap = argparse.ArgumentParser(description="Convert CSV/JSON keypoints to CVAT 1.1 XML.")
    ap.add_argument("input", help="Path to .csv or .json file")
    ap.add_argument("metadata", help="Path to metadata.py")
    ap.add_argument("--output", default="converted_bbox.xml", help="Output XML path")
    ap.add_argument("--video", help="Optional video path (used only for CSV to get width/height)")
    ap.add_argument("--task-name", default="converted", help="Name to embed in <meta>/<task>/<name>")
    args = ap.parse_args()

    dataset_info = load_metadata(args.metadata)

    # Keypoints order from metadata
    kp_pairs = sorted(
        [(int(k), v["name"]) for k, v in dataset_info["keypoint_info"].items()],
        key=lambda x: x[0]
    )
    keypoint_names = [name for (_i, name) in kp_pairs]
    keypoint_count = len(keypoint_names)

    # Classes from metadata
    classes = dataset_info.get("classes", ["object"])
    class_id_to_name = {i: c for i, c in enumerate(classes)}

    ext = os.path.splitext(args.input)[1].lower()
    if ext not in [".csv", ".json"]:
        raise ValueError("Unsupported input format. Use .json or .csv")

    width: Optional[int] = None
    height: Optional[int] = None

    if ext == ".csv":
        data, classes_by_instance, frames_seen = parse_csv(args.input, keypoint_count)
        frames_sorted = sorted(frames_seen)
        # width/height from video if provided
        if args.video:
            width, height = read_video_resolution(args.video)
    else:
        data, classes_by_instance, images_info, frames_seen = parse_json(args.input, keypoint_count)
        frames_sorted = sorted(frames_seen)
        # Try to pick a representative resolution from the first frame seen
        if frames_sorted:
            first = frames_sorted[0]
            info = images_info.get(first)
            if info:
                width, height = info.width, info.height

    tree = build_cvat_xml(
        data=data,
        classes_by_instance=classes_by_instance,
        class_id_to_name=class_id_to_name,
        frames_sorted=frames_sorted,
        width=width,
        height=height,
        task_name=args.task_name,
    )
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(f"Saved CVAT 1.1 XML to: {args.output}")

if __name__ == "__main__":
    main()


