import argparse
import os
import cv2
import torch
import csv
import time
from openface.face_detection import FaceDetector
from openface.landmark_detection import LandmarkDetector
from openface.multitask_model import MultitaskPredictor

def main(video_path, output_tsv, device):

    landmark_detector = LandmarkDetector('./weights/Landmark_98.pkl', device=device, device_ids=[0])
    face_detector = FaceDetector('./weights/Alignment_RetinaFace.pth', device=device)
    multitask_model = MultitaskPredictor('./weights/MTL_backbone.pth', device=device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = time.time()
        frame_path = f"{video_path}_frame_{frame_idx:05d}.jpg"

        # Write temporary frame
        cv2.imwrite(frame_path, frame)

        # Detect face
        cropped_face, dets = face_detector.get_face(frame_path)

        if cropped_face is not None and dets is not None:
            # Get landmarks
            landmarks = landmark_detector.detect_landmarks(frame, dets)

            # Predict multitask output
            emotion_logits, gaze_output, au_output = multitask_model.predict(cropped_face)
            emotion_idx = torch.argmax(emotion_logits, dim=1).item()

            results.append({
                'timestamp': timestamp,
                'image_path': frame_path,
                'face_id': 0,
                'face_detection': dets[0].tolist(),
                'landmarks': landmarks[0].tolist() if landmarks else None,
                'emotion': emotion_idx,
                'gaze_yaw': gaze_output[0][0].item(),
                'gaze_pitch': gaze_output[0][1].item(),
                'action_units': au_output.tolist()
            })

        frame_idx += 1

    cap.release()

    # Write output TSV
    with open(output_tsv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys(), delimiter='\t')
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"✅ Results saved to: {output_tsv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenFace3.0 on a video.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_tsv", type=str, default="results.tsv", help="Path to output TSV file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)")
    args = parser.parse_args()

    main(args.video_path, args.output_tsv, args.device)
