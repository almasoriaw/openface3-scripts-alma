import cv2
from openface.face_detection import FaceDetector
from openface.landmark_detection import LandmarkDetector
from head_pose import estimate_head_pose  # Make sure head_pose.py is in the same folder

# Load image
image_path = "input.jpg"  # Replace with your test image filename
image = cv2.imread(image_path)

# Initialize detectors
face_detector = FaceDetector(model_path='./weights/Alignment_RetinaFace.pth', device='cuda')
landmark_detector = LandmarkDetector(model_path='./weights/Landmark_98.pkl', device='cuda')

# Detect face
cropped_face, dets = face_detector.get_face(image_path)

if dets is not None and len(dets) > 0:
    landmarks = landmark_detector.detect_landmarks(image, dets)
    if landmarks:
        pitch, yaw, roll = estimate_head_pose(landmarks[0], image.shape)
        print(f"Head Pose:\n Pitch: {pitch:.2f}\n Yaw: {yaw:.2f}\n Roll: {roll:.2f}")
    else:
        print("No landmarks found.")
else:
    print("No face detected.")
