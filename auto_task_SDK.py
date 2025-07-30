# This script creates a task in CVAT and automatically imports the images associated
# with the task. The annotations have to be uploaded manually.
import os
import cv2
import shutil
from pathlib import Path
from dotenv import load_dotenv
import argparse
from getpass import getpass
from cvat_sdk import make_client
from cvat_sdk.core.helpers import DeferredTqdmProgressReporter
import subprocess
import sys

# Configuration
def parse_args():
    parser = argparse.ArgumentParser(description="Create a CVAT task and upload images or video.")
    parser.add_argument("--env", default=".env", help="Path to .env file")
    parser.add_argument("--username", default="JUAUW", help="CVAT username")
    parser.add_argument("--password", default="Yetisb130^^", help="CVAT password")
    parser.add_argument("--task", default="CSV", help="Name for the new task")
    parser.add_argument("--folder", help="Folder with media to upload",
                        default=r"C:\\Users\\audet\\OneDrive\\Desktop\\ULaval\\Stage 2025\\Code")
    parser.add_argument("--keypoints", default=r"C:\\Users\\audet\\OneDrive\\Desktop\\ULaval\\Stage 2025\\Code\\kpts.csv",
                         help="Path to keypoints.csv or keypoints.json file")
    return parser.parse_args()

def load_config():
    return {
        "CVAT_HOST": os.getenv("CVAT_HOST", "http://localhost:8080"),
        "LABEL_NAME": os.getenv("LABEL_NAME", "mouse"),
    }

# Creates a temporary label
def create_label(label_name):
    return [
  {
    "name": label_name,
    "id": 2564,
    "color": "#c02020",
    "type": "skeleton",
    "sublabels": [],
    "svg": "<line x1=&quot;47.18975830078125&quot; y1=&quot;42.73789596557617&quot; x2=&quot;52.19810485839844&quot; y2=&quot;36.56093215942383&quot; data-type=&quot;edge&quot; data-node-from=&quot;7&quot; data-node-to=&quot;8&quot;></line>\n<line x1=&quot;32.832496643066406&quot; y1=&quot;43.40567398071289&quot; x2=&quot;47.18975830078125&quot; y2=&quot;42.73789596557617&quot; data-type=&quot;edge&quot; data-node-from=&quot;3&quot; data-node-to=&quot;7&quot;></line>\n<line x1=&quot;46.188087463378906&quot; y1=&quot;55.25876235961914&quot; x2=&quot;47.18975830078125&quot; y2=&quot;42.73789596557617&quot; data-type=&quot;edge&quot; data-node-from=&quot;4&quot; data-node-to=&quot;7&quot;></line>\n<line x1=&quot;34.0011100769043&quot; y1=&quot;55.091819763183594&quot; x2=&quot;46.188087463378906&quot; y2=&quot;55.25876235961914&quot; data-type=&quot;edge&quot; data-node-from=&quot;6&quot; data-node-to=&quot;4&quot;></line>\n<line x1=&quot;34.0011100769043&quot; y1=&quot;55.091819763183594&quot; x2=&quot;32.832496643066406&quot; y2=&quot;43.40567398071289&quot; data-type=&quot;edge&quot; data-node-from=&quot;6&quot; data-node-to=&quot;3&quot;></line>\n<line x1=&quot;34.50194549560547&quot; y1=&quot;67.61268615722656&quot; x2=&quot;34.0011100769043&quot; y2=&quot;55.091819763183594&quot; data-type=&quot;edge&quot; data-node-from=&quot;5&quot; data-node-to=&quot;6&quot;></line>\n<line x1=&quot;22.148021697998047&quot; y1=&quot;60.7679443359375&quot; x2=&quot;34.0011100769043&quot; y2=&quot;55.091819763183594&quot; data-type=&quot;edge&quot; data-node-from=&quot;2&quot; data-node-to=&quot;6&quot;></line>\n<line x1=&quot;23.984416961669922&quot; y1=&quot;70.61769104003906&quot; x2=&quot;34.50194549560547&quot; y2=&quot;67.61268615722656&quot; data-type=&quot;edge&quot; data-node-from=&quot;1&quot; data-node-to=&quot;5&quot;></line>\n<line x1=&quot;23.984416961669922&quot; y1=&quot;70.61769104003906&quot; x2=&quot;22.148021697998047&quot; y2=&quot;60.7679443359375&quot; data-type=&quot;edge&quot; data-node-from=&quot;1&quot; data-node-to=&quot;2&quot;></line>\n<circle r=&quot;0.75&quot; cx=&quot;23.984416961669922&quot; cy=&quot;70.61769104003906&quot; data-type=&quot;element node&quot; data-element-id=&quot;1&quot; data-node-id=&quot;1&quot; data-label-id=&quot;2565&quot;></circle>\n<circle r=&quot;0.75&quot; cx=&quot;22.148021697998047&quot; cy=&quot;60.7679443359375&quot; data-type=&quot;element node&quot; data-element-id=&quot;2&quot; data-node-id=&quot;2&quot; data-label-id=&quot;2566&quot;></circle>\n<circle r=&quot;0.75&quot; cx=&quot;32.832496643066406&quot; cy=&quot;43.40567398071289&quot; data-type=&quot;element node&quot; data-element-id=&quot;3&quot; data-node-id=&quot;3&quot; data-label-id=&quot;2567&quot;></circle>\n<circle r=&quot;0.75&quot; cx=&quot;46.188087463378906&quot; cy=&quot;55.25876235961914&quot; data-type=&quot;element node&quot; data-element-id=&quot;4&quot; data-node-id=&quot;4&quot; data-label-id=&quot;2568&quot;></circle>\n<circle r=&quot;0.75&quot; cx=&quot;34.50194549560547&quot; cy=&quot;67.61268615722656&quot; data-type=&quot;element node&quot; data-element-id=&quot;5&quot; data-node-id=&quot;5&quot; data-label-id=&quot;2569&quot;></circle>\n<circle r=&quot;0.75&quot; cx=&quot;34.0011100769043&quot; cy=&quot;55.091819763183594&quot; data-type=&quot;element node&quot; data-element-id=&quot;6&quot; data-node-id=&quot;6&quot; data-label-id=&quot;2570&quot;></circle>\n<circle r=&quot;0.75&quot; cx=&quot;47.18975830078125&quot; cy=&quot;42.73789596557617&quot; data-type=&quot;element node&quot; data-element-id=&quot;7&quot; data-node-id=&quot;7&quot; data-label-id=&quot;2571&quot;></circle>\n<circle r=&quot;0.75&quot; cx=&quot;52.19810485839844&quot; cy=&quot;36.56093215942383&quot; data-type=&quot;element node&quot; data-element-id=&quot;8&quot; data-node-id=&quot;8&quot; data-label-id=&quot;2572&quot;></circle>",
    "attributes": []
  }
]

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

    # Parse arguments first so we can pass relevant paths
    args = parse_args()
    load_dotenv(args.env)
    config = load_config()

    # Define paths
    input_csv_or_json = args.keypoints
    metadata_path = "metadata.py"
    output_path = "converted_keypoints.json"
    video_or_image_folder = None
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        video_candidates = list(Path(args.folder).glob(f"*{ext}"))
        if video_candidates:
            video_or_image_folder = str(video_candidates[0])
            break

    if not video_or_image_folder:
        print("No video file found in the specified folder for resolution extraction.")
        return

    # Run raw.py
    try:
        print("Running raw.py...")
        subprocess.run([sys.executable, "raw.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running raw.py: {e}")
        return

    # Run convert_json_keypoints.py
    try:
        print("Running convert_json_keypoints.py...")
        subprocess.run([
    sys.executable,
        "convert_json_keypoints.py",
        input_csv_or_json,
        video_or_image_folder,
        metadata_path
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running convert_json_keypoints.py: {e}")
        return

    # Proceed with CVAT task creation
    username = args.username or input("Enter your CVAT username: ")
    password = args.password or getpass("Enter your CVAT password: ")
    task_name = args.task or input("Enter your desired task name: ")
    media_folder = Path(args.folder)

    if not media_folder.exists() or not media_folder.is_dir():
        print(f"Media folder does not exist: {media_folder}")
        return

    image_exts = [".jpg", ".jpeg", ".png"]
    video_exts = [".mp4", ".avi", ".mov", ".mkv"]

    image_paths = sorted([p for p in media_folder.glob("*") if p.suffix.lower() in image_exts])
    video_paths = sorted([p for p in media_folder.glob("*") if p.suffix.lower() in video_exts])

    if not image_paths and not video_paths:
        print("No supported image or video files found in the specified folder.")
        return

    with make_client(config["CVAT_HOST"], credentials=(username, password)) as client:
        print("Connected to CVAT server.")
        labels = create_label(config["LABEL_NAME"])
        spec = {"name": task_name, "labels": labels}

        if video_paths:
            if len(video_paths) > 1:
                print("Multiple video files found. CVAT supports only one video file per task.")
                return

            video_file = video_paths[0]
            print(f"Extracting frames from video: {video_file.name}")
            temp_dir = media_folder / "temp_extracted_frames"
            temp_dir.mkdir(exist_ok=True)

            try:
                extracted_frames = video_to_frames(video_file, temp_dir)
                if not extracted_frames:
                    print("No frames extracted from the video.")
                    return

                print(f"Uploading {len(extracted_frames)} extracted frames.")
                task = client.tasks.create_from_data(
                    spec=spec,
                    resources=[str(p) for p in extracted_frames],
                    pbar=DeferredTqdmProgressReporter()
                )
                print(f"Task '{task_name}' created with ID: {task.id}")
            finally:
                shutil.rmtree(temp_dir)

        else:
            print(f"Uploading {len(image_paths)} image(s).")
            task = client.tasks.create_from_data(
                spec=spec,
                resources=[str(p) for p in image_paths],
                pbar=DeferredTqdmProgressReporter()
            )
            print(f"Task '{task_name}' created with ID: {task.id}")

if __name__ == "__main__":
    main()




