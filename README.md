# CVAT Annotation Automation Pipeline

This repository provides a semi-automated pipeline for preparing, importing, correcting, and merging keypoints and bounding box annotations in [CVAT (Computer Vision Annotation Tool)](https://cvat.org/). The pipeline is designed to handle annotations from CSV or JSON format and integrates with the CVAT Python SDK to automate task creation and.

## Overview

The pipeline supports the following workflows:

- Conversion of raw metadata and annotations
- Upload of annotations to CVAT via SDK (only bounding boxes)
- Manual correction (if needed) within CVAT
- Export and post-processing of corrected annotations
- Final merging of keypoints and bounding boxes into a single JSON

---

## Pipeline Components

### 1. Metadata Preparation

- **metadata.py**  
  Generates base metadata schema needed for CVAT import.

- **raw.py**  
  Uses metadata.py output to generate skeleton.SVG which is used to create the skeleton in the CVAT skeleton constructor

### 2. Annotation Conversion

- **convert_json_keypoints.py**  
  Converts keypoints.csv or keypoints.json into converted_keypoints.json.

- **convert_json_bbox.py**  
  Converts bbox.csv or bbox.json into converted_bbox.json.

### 3. CVAT Task Automation

- **auto_task_SDK.py** 
  Uses the CVAT Python SDK to:
  - Create a task
  - Upload images
  - Import converted_keypoints.json, and converted_bbox.json

- **auto_kpts_bbox.py**  
  Depending on the use case it either directly uploads bounding boxes into CVAT or converts keypoints into bounding boxes and then uploads them to CVAT.

### 4. Final Merge

- **merge_bbox_keypoints.py**  
  Merges corrected_keypoints.json and corrected_bbox.json into a unified combined.json file.
