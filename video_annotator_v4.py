import cv2
import pandas as pd
import argparse
import ast
import os


def annotate_video(video_path, tsv_path, output_path, yaw_thresh=0.15, pitch_thresh=0.15):
    df = pd.read_csv(tsv_path, sep='\t')
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(df):
            break

        row = df.iloc[frame_idx]

        # Extract data
        yaw = float(row.get('gaze_yaw', 0))
        pitch = float(row.get('gaze_pitch', 0))
        gaze_focus = abs(yaw) < yaw_thresh and abs(pitch) < pitch_thresh

        # Action units
        au_data = ast.literal_eval(row['action_units'])[0] if 'action_units' in row and row['action_units'] else []
        au45 = au_data[44] if len(au_data) > 44 else 0.0

        # Draw focus status
        label = "Focused" if gaze_focus else "Unfocused"
        color = (0, 255, 0) if gaze_focus else (0, 0, 255)
        cv2.putText(frame, f"{label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Draw gaze values
        cv2.putText(frame, f"Yaw: {yaw:.2f}, Pitch: {pitch:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Draw blink if detected
        if au45 > 0.5:
            cv2.putText(frame, f"Blink detected (AU45={au45:.2f})", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Annotated video saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate video with focus and blink detection")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--tsv", type=str, required=True, help="Path to TSV results file")
    parser.add_argument("--output", type=str, required=True, help="Path to save annotated video")
    parser.add_argument("--yaw_thresh", type=float, default=0.15, help="Yaw threshold for focus detection")
    parser.add_argument("--pitch_thresh", type=float, default=0.15, help="Pitch threshold for focus detection")
    args = parser.parse_args()

    annotate_video(args.video, args.tsv, args.output, args.yaw_thresh, args.pitch_thresh)
