import cv2
import pandas as pd
import argparse


def draw_focus_state(frame, focused):
    label = "FOCUSED" if focused else "UNFOCUSED"
    color = (0, 255, 0) if focused else (0, 0, 255)
    cv2.putText(frame, f"Focus: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_emotion(frame, emotion):
    cv2.putText(frame, f"Emotion: {emotion}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


def draw_blink_status(frame, au_data):
    blink = au_data[0][0] > 0.5  # AU45
    label = "Yes" if blink else "No"
    cv2.putText(frame, f"Blink: {label}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)


def draw_mouth_status(frame, au_data):
    talking = au_data[0][5] > 0.2 or au_data[0][6] > 0.2 or au_data[0][7] > 0.2  # AU25/26/27
    label = "Talking" if talking else "Silent"
    cv2.putText(frame, f"Mouth AUs: {label}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)


def draw_head_motion(frame, curr_bbox, prev_bbox):
    if prev_bbox is None:
        return curr_bbox
    motion = sum([(a - b) ** 2 for a, b in zip(curr_bbox[:4], prev_bbox[:4])]) ** 0.5
    label = "Small" if motion < 5 else "Moving"
    cv2.putText(frame, f"Head Motion: {label}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
    return curr_bbox


def annotate_video(video_path, tsv_path, output_path, yaw_thresh=0.2, pitch_thresh=0.2):
    df = pd.read_csv(tsv_path, sep='\t')
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ))

    prev_bbox = None

    for idx, row in df.iterrows():
        ret, frame = cap.read()
        if not ret:
            break

        # Focus criteria
        focused = abs(row['gaze_yaw']) < yaw_thresh and abs(row['gaze_pitch']) < pitch_thresh

        # Annotations
        draw_focus_state(frame, focused)
        draw_emotion(frame, row['emotion'])
        draw_blink_status(frame, eval(row['action_units']))
        draw_mouth_status(frame, eval(row['action_units']))
        prev_bbox = draw_head_motion(frame, eval(row['face_detection']), prev_bbox)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Annotated video saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--tsv', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--yaw_thresh', type=float, default=0.2)
    parser.add_argument('--pitch_thresh', type=float, default=0.2)
    args = parser.parse_args()

    annotate_video(args.video, args.tsv, args.output, args.yaw_thresh, args.pitch_thresh)
