import cv2
import argparse
import numpy as np
import torch
import os
import csv
from datetime import datetime

from openface.face_detection import FaceDetector
from openface.landmark_detection import LandmarkDetector
from head_pose import estimate_head_pose
from openface.Pytorch_Retinaface.utils.box_utils import decode, decode_landm
from openface.Pytorch_Retinaface.layers.functions.prior_box import PriorBox

# --- Parse CLI argument ---
parser = argparse.ArgumentParser(description="Head pose estimation on video frames")
parser.add_argument('--video', type=str, required=True, help='Path to input video file')
args = parser.parse_args()
video_path = args.video

# --- Create unique output folder ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = os.path.join("output", f"test_{timestamp}")
os.makedirs(output_folder, exist_ok=True)

# --- Prepare CSV file ---
csv_path = os.path.join(output_folder, "head_pose_results.csv")
csv_file = open(csv_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Pitch", "Yaw", "Roll"])

# --- Initialize detectors ---
face_detector = FaceDetector(model_path='../weights/Alignment_RetinaFace.pth', device='cuda')
landmark_detector = LandmarkDetector(model_path='../weights/Landmark_98.pkl', device='cuda', device_ids=[0])

# --- Open video file ---
cap = cv2.VideoCapture(video_path)
frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    frame_num += 1
    print(f"\nFrame {frame_num}")

    # Resize and preprocess
    resize = 1.0
    frame_resized = cv2.resize(frame, None, fx=resize, fy=resize)
    img = np.float32(frame_resized)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).to(face_detector.device)

    # Run model
    loc, conf, landms = face_detector.model(img)

    # Decode detections
    height, width, _ = frame_resized.shape
    priorbox = PriorBox(face_detector.cfg, image_size=(height, width))
    priors = priorbox.forward().to(face_detector.device)
    boxes = decode(loc.data.squeeze(0), priors, face_detector.cfg['variance'])
    landmarks = decode_landm(landms.data.squeeze(0), priors, face_detector.cfg['variance'])
    scores = conf.data.squeeze(0)[:, 1].cpu().numpy()

    boxes = boxes.cpu().numpy()
    landmarks = landmarks.cpu().numpy()

    # Filter by score threshold
    mask = scores > 0.1
    boxes = boxes[mask]
    landmarks = landmarks[mask]

    # Scale boxes AFTER filtering
    boxes[:, 0] *= width
    boxes[:, 1] *= height
    boxes[:, 2] *= width
    boxes[:, 3] *= height

    print(f"Valid boxes (after scaling): {boxes}")

    if len(boxes) > 0:
        # Pick largest face
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        biggest = np.argmax(areas)
        bbox = boxes[biggest].astype(int)

        if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
            print("Skipped invalid face bbox.")
            continue

        # Detect 98 landmarks
        bbox_with_score = np.array(list(bbox) + [1.0])  # [x1, y1, x2, y2, score]
        landmarks_98 = landmark_detector.detect_landmarks(frame, [bbox_with_score])

        if landmarks_98:
            pitch, yaw, roll = estimate_head_pose(landmarks_98[0], frame.shape)
            print(f"Head Pose — Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f}")

            # Save to CSV
            csv_writer.writerow([frame_num, f"{pitch:.2f}", f"{yaw:.2f}", f"{roll:.2f}"])
        else:
            print("No landmarks found.")
    else:
        print("No face detected.")

cap.release()
csv_file.close()
print(f"\n✅ Head pose data saved to: {csv_path}")
