import cv2
import argparse
import numpy as np
import torch
import os
from datetime import datetime

from openface.face_detection import FaceDetector
from openface.landmark_detection import LandmarkDetector
from head_pose import estimate_head_pose
from openface.Pytorch_Retinaface.utils.box_utils import decode, decode_landm
from openface.Pytorch_Retinaface.layers.functions.prior_box import PriorBox

# --- Parse CLI argument ---
parser = argparse.ArgumentParser(description="Head pose estimation and annotation on video frames")
parser.add_argument('--video', type=str, required=True, help='Path to input video file')
args = parser.parse_args()
video_path = args.video

# --- Initialize detectors ---
face_detector = FaceDetector(model_path='../weights/Alignment_RetinaFace.pth', device='cuda')
landmark_detector = LandmarkDetector(model_path='../weights/Landmark_98.pkl', device='cuda', device_ids=[0])

# --- Prepare output folder ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_name = os.path.splitext(os.path.basename(video_path))[0]
output_dir = f"output/{video_name}_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# --- Prepare video I/O ---
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = os.path.join(output_dir, f"{video_name}_annotated.mp4")
out_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1

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

    # Scale boxes
    boxes[:, 0] *= width
    boxes[:, 1] *= height
    boxes[:, 2] *= width
    boxes[:, 3] *= height

    if len(boxes) > 0:
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        biggest = np.argmax(areas)
        bbox = boxes[biggest].astype(int)

        if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
            print(f"Frame {frame_num}: Skipped invalid bbox.")
            out_writer.write(frame)
            continue

        bbox_with_score = np.array(list(bbox) + [1.0])
        landmarks_98 = landmark_detector.detect_landmarks(frame, [bbox_with_score])

        if landmarks_98:
            pitch, yaw, roll = estimate_head_pose(landmarks_98[0], frame.shape)
            print(f"Frame {frame_num} — Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f}")
            # Draw annotation
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"Pitch: {pitch:.1f}, Yaw: {yaw:.1f}, Roll: {roll:.1f}",
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2)
        else:
            print(f"Frame {frame_num}: No landmarks found.")
    else:
        print(f"Frame {frame_num}: No face detected.")

    out_writer.write(frame)

cap.release()
out_writer.release()
print(f"Annotated video saved to: {output_path}")
