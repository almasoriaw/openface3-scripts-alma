import cv2
import argparse
import os
import pandas as pd

# Define thresholds for gaze-based focus detection
def is_focused(gaze_yaw, gaze_pitch, yaw_thresh=0.2, pitch_thresh=0.2):
    """
    Simple rule: Focused if gaze is roughly centered (i.e., facing screen).
    """
    return abs(gaze_yaw) < yaw_thresh and abs(gaze_pitch) < pitch_thresh

def annotate_video(video_path, tsv_path, output_path, yaw_thresh=0.2, pitch_thresh=0.2):
    df = pd.read_csv(tsv_path, sep='\t')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(df):
            break

        row = df.iloc[frame_idx]
        gaze_yaw = row['gaze_yaw']
        gaze_pitch = row['gaze_pitch']

        focused = is_focused(gaze_yaw, gaze_pitch, yaw_thresh, pitch_thresh)
        status_text = "Focused" if focused else "Distracted"
        color = (0, 255, 0) if focused else (0, 0, 255)

        cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Annotated video saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--tsv', required=True, help='Path to OpenFace TSV results')
    parser.add_argument('--output', default='annotated_output.mp4', help='Path to output annotated video')
    parser.add_argument('--yaw_thresh', type=float, default=0.2, help='Yaw threshold for focus detection')
    parser.add_argument('--pitch_thresh', type=float, default=0.2, help='Pitch threshold for focus detection')
    args = parser.parse_args()

    annotate_video(args.video, args.tsv, args.output, args.yaw_thresh, args.pitch_thresh)
