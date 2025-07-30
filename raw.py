# This script creates a raw input for the label creator in CVAT
# based on the metadata.py file.
import importlib.util
import json

def load_metadata(metadata_path: str):
    spec = importlib.util.spec_from_file_location("metadata", metadata_path)
    metadata = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metadata)
    return metadata.dataset_info

def generate_svg_string(dataset_info):
    keypoint_info = dataset_info["keypoint_info"]
    skeleton_info = dataset_info["skeleton_info"]

    coordinates = {
        "Snout": (23.984416961669922, 70.61769104003906),
        "Right Ear": (22.148021697998047, 60.7679443359375),
        "Right Leg": (32.832496643066406, 43.40567398071289),
        "Left Leg": (46.188087463378906, 55.25876235961914),
        "Left Ear": (34.50194549560547, 67.61268615722656),
        "Centroid": (34.0011100769043, 55.091819763183594),
        "Base of Tail": (47.18975830078125, 42.73789596557617),
        "Tailtag": (52.19810485839844, 36.56093215942383),
    }

    colors = [
        "#33ddff", "#fa3253", "#ffcc33", "#aaf0d1",
        "#34d1b7", "#ff6037", "#cc9933", "#733380"
    ]

    name_to_id = {kp["name"]: kp_id + 1 for kp_id, kp in keypoint_info.items()}
    id_to_name = {kp_id + 1: kp["name"] for kp_id, kp in keypoint_info.items()}

    svg_elements = ['<svg viewBox="0 0 100 100">']

    # Draw edges
    for skel in skeleton_info.values():
        a, b = skel["link"]
        x1, y1 = coordinates[a]
        x2, y2 = coordinates[b]
        svg_elements.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" '
            f'data-type="edge" data-node-from="{name_to_id[a]}" data-node-to="{name_to_id[b]}" '
            f'stroke-width="0.5"/>'
        )

    # Draw keypoints
    for i, (kp_id, kp) in enumerate(keypoint_info.items()):
        name = kp["name"]
        x, y = coordinates[name]
        color = colors[i]
        node_id = name_to_id[name]
        svg_elements.append(
            f'<circle r="0.75" cx="{x}" cy="{y}" data-type="element node" '
            f'data-element-id="{node_id}" data-node-id="{node_id}" '
            f'stroke="black" fill="{color}" stroke-width="0.1"/>'
        )

    # Embed description block
    desc_dict = {}
    for i, (kp_id, kp) in enumerate(keypoint_info.items()):
        node_id = str(kp_id + 1)
        desc_dict[node_id] = {
            "name": kp["name"],
            "color": colors[i],
            "type": "points",
            "attributes": [],
            "has_parent": True
        }
    svg_elements.append(
        f'<desc data-description-type="labels-specification">{json.dumps(desc_dict)}</desc>'
    )

    svg_elements.append('</svg>')
    return ''.join(svg_elements)

def convert_to_json_structure(dataset_info, svg_string):
    keypoint_info = dataset_info["keypoint_info"]
    colors = [
        "#33ddff", "#fa3253", "#ffcc33", "#aaf0d1",
        "#34d1b7", "#ff6037", "#cc9933", "#733380"
    ]
    label_id_start = 0
    entity_sublabels = []
    for idx, (kp_id, kp) in enumerate(sorted(keypoint_info.items())):
        sublabel = {
            "name": kp["name"],
            "attributes": [],
            "type": "points",
            "color": colors[idx],
            "id": label_id_start + 2 + idx
        }
        entity_sublabels.append(sublabel)

    mouse = {
        "name": "mouse",
        "id": label_id_start + 1,
        "color": "#c02020",
        "type": "skeleton",
        "sublabels": entity_sublabels,
        "svg": svg_string,
        "attributes": []
    }

    return [mouse]

def main():
    metadata_path = "metadata.py"
    output_json = "converted_metadata.json"

    dataset_info = load_metadata(metadata_path)
    svg_string = generate_svg_string(dataset_info)
    labels = convert_to_json_structure(dataset_info, svg_string)

    with open(output_json, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"Saved to {output_json} with inline SVG.")

if __name__ == "__main__":
    main()

