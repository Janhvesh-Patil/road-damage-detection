"""
Inference script for road damage detection using YOLOv8.
Generates one YOLO-format prediction per test image and packages results
into submission.zip for hackathon submission.
"""

# =====================================================
# WINDOWS + OPENCV STABILITY SETTINGS (DO NOT CHANGE)
# =====================================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# =====================================================
# STANDARD IMPORTS
# =====================================================
from ultralytics import YOLO
import shutil
import glob
import gc

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
TEST_IMAGES_DIR = os.getenv("TEST_IMAGES_DIR", "test")
CONF_THRESHOLD = 0.25
OUTPUT_ROOT = "submission"
LABEL_DIR = os.path.join(OUTPUT_ROOT, "labels")

# =====================================================
# CLEAN OUTPUT DIRECTORY
# =====================================================
if os.path.exists(OUTPUT_ROOT):
    shutil.rmtree(OUTPUT_ROOT)

os.makedirs(LABEL_DIR, exist_ok=True)

# =====================================================
# LOAD MODEL (CPU ONLY)
# =====================================================
model = YOLO(MODEL_PATH)

# =====================================================
# COLLECT TEST IMAGES
# =====================================================
image_paths = sorted(
    glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg")) +
    glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png")) +
    glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpeg"))
)

assert len(image_paths) == 6000, f"Expected 6000 images, found {len(image_paths)}"
print(f"Starting inference on {len(image_paths)} images...")

# =====================================================
# INFERENCE (ONE IMAGE AT A TIME – SAFE)
# =====================================================
for idx, img_path in enumerate(image_paths, 1):
    model.predict(
        source=img_path,
        conf=CONF_THRESHOLD,
        imgsz=512,
        save_txt=True,
        save_conf=True,
        project=OUTPUT_ROOT,
        name="",                # forces output to submission/labels/
        exist_ok=True,
        device="cpu",
        verbose=False
    )

    gc.collect()

    if idx % 250 == 0:
        print(f"Processed {idx}/6000 images")

print("Inference completed.")

# =====================================================
# POST-PROCESSING (ENSURE 1 FILE PER IMAGE)
# =====================================================
for img_path in image_paths:
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(LABEL_DIR, f"{img_name}.txt")

    # If YOLO produced no label file → create fallback
    if not os.path.exists(label_path):
        with open(label_path, "w") as f:
            f.write("0 0.5 0.5 0.1 0.1 0.01\n")
    else:
        with open(label_path, "r") as f:
            lines = f.readlines()

        best = max(lines, key=lambda x: float(x.strip().split()[-1]))

        with open(label_path, "w") as f:
            f.write(best)

# =====================================================
# FINAL VALIDATION
# =====================================================
label_files = glob.glob(os.path.join(LABEL_DIR, "*.txt"))
assert len(label_files) == 6000, f"Expected 6000 label files, found {len(label_files)}"

for lf in label_files:
    with open(lf, "r") as f:
        parts = f.readline().strip().split()
        assert len(parts) == 6, f"Bad format in {lf}"
        for v in parts[1:]:
            assert 0.0 <= float(v) <= 1.0, f"Out-of-range value in {lf}"

print("Submission format validated.")

# =====================================================
# ZIP SUBMISSION
# =====================================================
shutil.make_archive("submission", "zip", LABEL_DIR)
print("submission.zip created successfully.")
