import cv2
import argparse
import numpy as np
import torch
import os
import datetime

from openface.face_detection import FaceDetector
from openface.landmark_detection import LandmarkDetector
from head_pose import estimate_head_pose
from openface.Pytorch_Retinaface.utils.box_utils import decode, decode_landm
from openface.Pytorch_Retinaface.layers.functions.prior_box import PriorBox

# --- Parse CLI argument ---
parser = argparse.ArgumentParser(description="Annotate video with head pose estimation")
parser.add_argument('--video', type=str, required=True, help='Path to input video file')
args = parser.parse_args()
video_path = args.video

# --- Initialize detectors ---
face_detector = FaceDetector(model_path='../weights/Alignment_RetinaFace.pth', device='cuda')
landmark_detector = LandmarkDetector(model_path='../weights/Landmark_98.pkl', device='cuda', device_ids=[0])

# --- Prepare output folder ---
base_name = os.path.splitext(os.path.basename(video_path))[0]
date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = os.path.join('output', f'{base_name}_{date_str}')
os.makedirs(output_dir, exist_ok=True)

# --- Open video ---
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
output_path = os.path.join(output_dir, f'{base_name}_annotated.mp4')
writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    frame_resized = cv2.resize(frame, None, fx=1.0, fy=1.0)
    img = np.float32(frame_resized)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).to(face_detector.device)

    loc, conf, landms = face_detector.model(img)
    height, width, _ = frame_resized.shape
    priorbox = PriorBox(face_detector.cfg, image_size=(height, width))
    priors = priorbox.forward().to(face_detector.device)
    boxes = decode(loc.data.squeeze(0), priors, face_detector.cfg['variance'])
    scores = conf.data.squeeze(0)[:, 1].cpu().numpy()
    boxes = boxes.cpu().numpy()

    mask = scores > 0.1
    boxes = boxes[mask]
    boxes[:, 0] *= width
    boxes[:, 1] *= height
    boxes[:, 2] *= width
    boxes[:, 3] *= height

    if len(boxes) > 0:
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        biggest = np.argmax(areas)
        bbox = boxes[biggest].astype(int)

        if bbox[2] - bbox[0] > 0 and bbox[3] - bbox[1] > 0:
            bbox_with_score = np.array(list(bbox) + [1.0])
            landmarks_98 = landmark_detector.detect_landmarks(frame, [bbox_with_score])

            if landmarks_98:
                pitch, yaw, roll = estimate_head_pose(landmarks_98[0], frame.shape)
                text = f"Pitch: {pitch:.1f}, Yaw: {yaw:.1f}, Roll: {roll:.1f}"
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    writer.write(frame)

cap.release()
writer.release()
print(f"Annotated video saved to: {output_path}")
