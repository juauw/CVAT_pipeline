# This script generates an SVG file from metadata.py that is ready to upload to CVAT
import importlib.util
import json
import math
import seaborn as sns

# Loads the metadata.py file
def load_metadata(metadata_path: str):
    spec = importlib.util.spec_from_file_location("metadata", metadata_path)
    metadata = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metadata)
    return metadata.dataset_info

def get_positions(keypoint_info):
    # Reference layout for mice skeleton
    reference_coords = {
        "Snout": (23.984416961669922, 70.61769104003906),
        "Right Ear": (22.148021697998047, 60.7679443359375),
        "Right Leg": (32.832496643066406, 43.40567398071289),
        "Left Leg": (46.188087463378906, 55.25876235961914),
        "Left Ear": (34.50194549560547, 67.61268615722656),
        "Centroid": (34.0011100769043, 55.091819763183594),
        "Base of Tail": (47.18975830078125, 42.73789596557617),
        "Tailtag": (52.19810485839844, 36.56093215942383),
    }

    keypoint_names = [kp["name"] for _, kp in sorted(keypoint_info.items())]
    if all(name in reference_coords for name in keypoint_names):
        return {name: reference_coords[name] for name in keypoint_names}, "reference"
    else:
        # Circular fallback layout
        n = len(keypoint_names)
        angle_step = 2 * math.pi / n
        radius = 35
        center_x, center_y = 50, 50
        return {
            name: (
                center_x + radius * math.cos(i * angle_step),
                center_y + radius * math.sin(i * angle_step)
            )
            for i, name in enumerate(keypoint_names)
        }, "fallback"

# Generates SVG
def generate_svg(dataset_info, output_file="skeleton.svg"):
    keypoint_info = dataset_info["keypoint_info"]
    skeleton_info = dataset_info["skeleton_info"]
    palette = sns.color_palette("hls", len(keypoint_info))

    name_to_id = {v["name"]: v["id"] + 1 for v in keypoint_info.values()}
    positions, layout_used = get_positions(keypoint_info)

    svg_lines = ['<svg viewBox="0 0 100 100">']
    desc = {}

    # Draw skeleton edges
    for skel in skeleton_info.values():
        a, b = skel["link"]
        if a not in positions or b not in positions:
            continue  # Skip invalid edges
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        svg_lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" '
            f'data-type="edge" data-node-from="{name_to_id[a]}" stroke-width="0.5" '
            f'data-node-to="{name_to_id[b]}"/>'
        )

    # Draw keypoints
    for idx, (kp_id, kp) in enumerate(sorted(keypoint_info.items())):
        name = kp["name"]
        node_id = kp_id + 1
        x, y = positions[name]
        color = '#{:02x}{:02x}{:02x}'.format(
            int(palette[idx][0] * 255),
            int(palette[idx][1] * 255),
            int(palette[idx][2] * 255)
        )
        svg_lines.append(
            f'<circle r="0.75" cx="{x}" cy="{y}" data-type="element node" '
            f'data-element-id="{node_id}" data-node-id="{node_id}" '
            f'stroke="black" fill="{color}" stroke-width="0.1"/>'
        )
        desc[str(node_id)] = {
            "name": name,
            "color": color,
            "type": "points",
            "attributes": [],
            "has_parent": True
        }

    svg_lines.append(
        f'<desc data-description-type="labels-specification">{json.dumps(desc)}</desc>'
    )
    svg_lines.append('</svg>')

    with open(output_file, "w") as f:
        f.write("\n".join(svg_lines))

    print(f"SVG saved to {output_file} using {layout_used} layout.")

def main():
    metadata_path = "metadata.py"
    dataset_info = load_metadata(metadata_path)
    generate_svg(dataset_info)

if __name__ == "__main__":
    main()

