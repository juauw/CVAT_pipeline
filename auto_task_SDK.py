# This script creates a task in CVAT and automatically imports the images associated
# with the task. The annotations must be uploaded manually.
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
    parser.add_argument("--task", default="Keypoint annotation", help="Name for the new task")
    parser.add_argument("--folder", help="Folder with media to upload",
                        default=r"C:\\Users\\audet\\OneDrive\\Desktop\\ULaval\\Stage 2025\\CVAT_pipeline")
    parser.add_argument("--annotations", default=r"C:\\Users\\audet\\OneDrive\\Desktop\\ULaval\\Stage 2025\\CVAT_pipeline\\kpts.csv",
                         help="Path to annotations.csv or annotations.json file")
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
    "id": 1,
    "color": "#c02020",
    "type": "skeleton",
    "sublabels": [],
    "svg": "",
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

    # Run raw.py
    try:
        print("Running raw.py...")
        subprocess.run([sys.executable, "raw.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running raw.py: {e}")
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




