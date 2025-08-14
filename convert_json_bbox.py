# This program takes two files that are named metadata.py and a .json or .csv file and produces
# a file named converted_bbox.xml which is compatible with the CVAT bounding box import module.
import argparse, csv, importlib.util, json, os
from collections import defaultdict, Counter
from typing import Dict, Any, Tuple, Optional
from xml.etree.ElementTree import Element, SubElement, ElementTree
import cv2

# Metadata loader
def load_metadata(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")
    spec = importlib.util.spec_from_file_location("metadata", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "dataset_info"):
        return {}  # be permissive
    return mod.dataset_info or {}

# Helpers
def read_video_resolution(video_path: str) -> Tuple[int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h

def to_int(x): 
    try: return int(x)
    except: return int(float(x))

def clamp_bbox(xtl, ytl, xbr, ybr, w=None, h=None):
    if w is None or h is None: 
        return xtl, ytl, xbr, ybr
    xtl = max(0.0, min(float(w), float(xtl)))
    ytl = max(0.0, min(float(h), float(ytl)))
    xbr = max(0.0, min(float(w), float(xbr)))
    ybr = max(0.0, min(float(h), float(ybr)))
    return xtl, ytl, xbr, ybr

# Reads csv file
def read_csv_boxes(csv_path: str, vid_wh: Tuple[Optional[int], Optional[int]]) -> Tuple[Dict[int, Dict[int, Tuple[float,float,float,float]]], Dict[int, Counter], Optional[int], Optional[int]]:
    boxes = defaultdict(dict)
    classes_by_instance: Dict[int, Counter] = defaultdict(Counter)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        needed = {"frame_id","class_id","instance_id","cx","cy","w","h"}
        if not needed.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV must contain columns: {sorted(needed)}")
        for row in reader:
            fr = to_int(row["frame_id"])
            cls = to_int(row["class_id"])
            inst = to_int(row["instance_id"])
            cx = float(row["cx"]); cy = float(row["cy"])
            w  = float(row["w"]);  h  = float(row["h"])
            xtl = cx - w/2.0; ytl = cy - h/2.0
            xbr = cx + w/2.0; ybr = cy + h/2.0
            xtl, ytl, xbr, ybr = clamp_bbox(xtl, ytl, xbr, ybr, *vid_wh)
            boxes[fr][inst] = (xtl, ytl, xbr, ybr)
            classes_by_instance[inst][cls] += 1
    return boxes, classes_by_instance, vid_wh[0], vid_wh[1]

# Reads json file
def read_json_boxes(json_path: str) -> Tuple[Dict[int, Dict[int, Tuple[float,float,float,float]]], Dict[int, Counter], Optional[int], Optional[int]]:
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Map image_id -> frame_id, width, height (prefer explicit)
    id2frame, frame_wh = {}, {}
    for im in obj.get("images", []):
        img_id = to_int(im.get("id", 0))
        if "frame_id" in im: frame_id = to_int(im["frame_id"])
        elif "frame_index" in im: frame_id = to_int(im["frame_index"])
        else: frame_id = img_id
        id2frame[img_id] = frame_id
        if "width" in im and "height" in im:
            frame_wh[frame_id] = (to_int(im["width"]), to_int(im["height"]))

    boxes = defaultdict(dict)
    classes_by_instance: Dict[int, Counter] = defaultdict(Counter)
    for ann in obj.get("annotations", []):
        img_id = to_int(ann["image_id"])
        fr = id2frame.get(img_id, img_id)
        inst = (ann.get("track_id") or ann.get("instance_id") or ann.get("id") or 0)
        inst = to_int(inst)
        cls = to_int(ann.get("category_id", 0))

        xtl=ytl=xbr=ybr=None
        if "bbox" in ann and isinstance(ann["bbox"], (list, tuple)) and len(ann["bbox"]) >= 4:
            x, y, w, h = [float(v) for v in ann["bbox"][:4]]
            xtl, ytl, xbr, ybr = x, y, x + w, y + h
        elif all(k in ann for k in ("cx","cy","w","h")):
            cx, cy, w, h = float(ann["cx"]), float(ann["cy"]), float(ann["w"]), float(ann["h"])
            xtl, ytl, xbr, ybr = cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0
        else:
            # try deriving from COCO keypoints if present (last resort)
            kpts = ann.get("keypoints")
            if kpts and len(kpts) >= 3:
                xs = [float(kpts[i]) for i in range(0, len(kpts), 3) if float(kpts[i+2]) > 0]
                ys = [float(kpts[i+1]) for i in range(0, len(kpts), 3) if float(kpts[i+2]) > 0]
                if xs and ys:
                    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
                    pad = 0.02 * max(x1 - x0, y1 - y0)
                    xtl, ytl, xbr, ybr = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        if xtl is None:  # nothing usable
            continue

        # Clamp when possible
        W = H = None
        if fr in frame_wh:
            W, H = frame_wh[fr]
        xtl, ytl, xbr, ybr = clamp_bbox(xtl, ytl, xbr, ybr, W, H)
        boxes[fr][inst] = (xtl, ytl, xbr, ybr)
        classes_by_instance[inst][cls] += 1

    # choose a representative resolution if available
    width = height = None
    if frame_wh:
        # take the most common WxH
        wh_counts = Counter(frame_wh.values())
        width, height = wh_counts.most_common(1)[0][0]
    return boxes, classes_by_instance, width, height

#  CVAT XML writer
def build_cvat_xml(
    boxes: Dict[int, Dict[int, Tuple[float,float,float,float]]],
    classes_by_instance: Dict[int, Counter],
    class_id_to_name: Dict[int, str],
    width: Optional[int],
    height: Optional[int],
    task_name: str
) -> ElementTree:

    root = Element("annotations")
    SubElement(root, "version").text = "1.1"

    meta = SubElement(root, "meta")
    task = SubElement(meta, "task")
    SubElement(task, "name").text = task_name
    if width is not None and height is not None:
        orig = SubElement(task, "original_size")
        SubElement(orig, "width").text = str(width)
        SubElement(orig, "height").text = str(height)

    labels = SubElement(task, "labels")
    for cid in sorted(class_id_to_name):
        lbl = SubElement(labels, "label")
        SubElement(lbl, "name").text = class_id_to_name[cid]

    # all frames + instances
    all_frames = sorted(boxes.keys())
    all_instances = sorted({iid for f in boxes.values() for iid in f.keys()})

    for inst in all_instances:
        # pick most frequent class for this instance
        cls_counter = classes_by_instance.get(inst, Counter({0:1}))
        cid, _ = cls_counter.most_common(1)[0]
        label = class_id_to_name.get(cid, class_id_to_name.get(0, "object"))

        track = SubElement(root, "track", id=str(inst), label=label)
        track.set("source", "manual")

        # write boxes for every frame where this instance appears
        for fr in all_frames:
            if inst not in boxes[fr]:
                continue
            xtl, ytl, xbr, ybr = boxes[fr][inst]
            SubElement(
                track, "box",
                frame=str(fr), outside="0", occluded="0", keyframe="1",
                xtl=f"{xtl:.3f}", ytl=f"{ytl:.3f}", xbr=f"{xbr:.3f}", ybr=f"{ybr:.3f}"
            )

    return ElementTree(root)

# Main
def main():
    ap = argparse.ArgumentParser(description="CSV/JSON → CVAT 1.1 bbox tracks")
    ap.add_argument("input", help="Path to .csv (cx,cy,w,h) or COCO .json")
    ap.add_argument("metadata", help="Path to metadata.py (for class names)")
    ap.add_argument("--output", default="converted_bbox.xml", help="Output XML")
    ap.add_argument("--video", help="Optional video path (for CSV width/height)")
    ap.add_argument("--task-name", default="converted", help="CVAT task name")
    args = ap.parse_args()

    info = load_metadata(args.metadata)
    classes = info.get("classes") or ["object"]
    class_id_to_name = {i: c for i, c in enumerate(classes)}

    ext = os.path.splitext(args.input)[1].lower()
    if ext not in (".csv", ".json"):
        raise ValueError("Input must be .csv or .json")

    width = height = None
    if ext == ".csv":
        vid_wh = (None, None)
        if args.video:
            vid_wh = read_video_resolution(args.video)
        boxes, cls_by_inst, width, height = read_csv_boxes(args.input, vid_wh)
    else:
        boxes, cls_by_inst, width, height = read_json_boxes(args.input)

    tree = build_cvat_xml(
        boxes=boxes,
        classes_by_instance=cls_by_inst,
        class_id_to_name=class_id_to_name,
        width=width, height=height,
        task_name=args.task_name,
    )
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(f"Saved CVAT 1.1 XML to: {args.output}")

if __name__ == "__main__":
    main()



