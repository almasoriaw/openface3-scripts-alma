import cv2
import pandas as pd
import ast

# Input files
video_path = "test_videos/alma_test1.mp4"
tsv_path = "output/alma_results.tsv"
output_path = "output/alma_annotated.mp4"

df = pd.read_csv(tsv_path, sep="\t")
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

for i, row in df.iterrows():
    ret, frame = cap.read()
    if not ret:
        break

    try:
        gaze_yaw = float(row['gaze_yaw'])
        gaze_pitch = float(row['gaze_pitch'])
        x, y = int(frame.shape[1] / 2), int(frame.shape[0] / 2)
        dx, dy = int(gaze_yaw * 100), int(-gaze_pitch * 100)
        cv2.arrowedLine(frame, (x, y), (x + dx, y + dy), (0, 255, 0), 2)
        cv2.putText(frame, f"Gaze: Yaw={gaze_yaw:.2f}, Pitch={gaze_pitch:.2f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Optional: show focus score based on thresholds
        focused = abs(gaze_yaw) < 0.1 and abs(gaze_pitch) < 0.1
        cv2.putText(frame, "Focused" if focused else "Distracted", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0) if focused else (0, 0, 255), 2)

    except Exception as e:
        print(f"Skipping frame {i} due to error: {e}")

    out.write(frame)

cap.release()
out.release()
print("Annotated video saved to", output_path)
