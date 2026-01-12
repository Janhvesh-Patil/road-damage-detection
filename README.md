# Road Damage Detection using YOLOv8

This repository contains the training and inference pipeline developed for a road damage detection task as part of a hackathon challenge. The objective is to detect and classify different types of road surface damages from images using a deep learning–based object detection model.

---

## Problem Statement

Given an image of a road surface, the task is to detect the presence of road damage and classify it into one of the following categories:

- Longitudinal Crack  
- Transverse Crack  
- Alligator Crack  
- Other Corruption  
- Pothole  

Each test image requires exactly one prediction in YOLO format, including a confidence score.

---

## Model Architecture

The solution uses **YOLOv8n (Nano)** from the Ultralytics framework due to its balance between inference speed and detection performance under limited computational resources.

Key characteristics:
- One-stage object detector  
- Anchor-free detection  
- Optimized for real-time and low-resource environments  

---

## Dataset Structure

The dataset is expected to follow the standard YOLO directory format:

dataset/<br>
├── train/<br>
│ ├── images/<br>
│ └── labels/<br>
├── val/<br>
│ ├── images/<br>
│ └── labels/<br>


The dataset is **not included** in this repository and must be placed manually in the above structure before training.

---

## Data Configuration

The dataset configuration is defined in `data.yaml` using relative paths:

path: ./dataset
train: train/images
val: val/images

Class mapping:

- 0: Longitudinal crack  
- 1: Transverse crack  
- 2: Alligator crack  
- 3: Other corruption  
- 4: Pothole  

---

## Training

Training is performed using the `train.ipynb` notebook.

Final training configuration:
- Model: YOLOv8n  
- Image size: 512 × 512  
- Epochs: 15  
- Batch size: 16  
- Pretrained weights: Enabled  

### Training Environment

Due to GPU time limitations, training was carried out across **multiple cloud environments (Google Colab and Kaggle)**. While dataset loading paths varied between environments, the **model architecture, hyperparameters, and training strategy remained consistent**.

The provided `train.ipynb` reflects the **final consolidated training strategy**.

---

## Inference and Submission

Inference is performed using `inference.py`, which:

- Runs entirely on CPU  
- Processes one image at a time for stability  
- Ensures exactly **one prediction per image**  
- Outputs predictions in YOLO text format with confidence  
- Automatically generates `submission.zip`  

Fallback predictions are generated for images where no detection is produced, ensuring a valid submission format.

---

## Repository Structure

road-damage-detection/<br>
├── inference.py<br>
├── train.ipynb<br>
├── data.yaml<br>
├── requirements.txt<br>
├── README.md<br>
└── .gitignore<br>


---

## Dependencies

Install required packages using:

pip install -r requirements.txt

Dependencies:
- ultralytics  
- opencv-python  
- numpy  

---

## Notes

- Model weights (`.pt` files) and datasets are intentionally excluded.  
- All paths are relative for portability.  
- The codebase is designed to be reproducible and platform-agnostic.  

---

## Authors

- Kajal Jain
- Janhvesh Patil
