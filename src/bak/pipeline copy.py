import os
import cv2
from .video_io import open_video, create_writer
from .detector import YoloDetector

def run_pipeline(
    input_path="data/sample.mp4",
    output_path="outputs/out.mp4",
    resize_to=(640, 480),
):
    cap = open_video(input_path)
    os.makedirs("outputs", exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = create_writer(output_path, fps, resize_to)

    detector = YoloDetector()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, resize_to)
        annotated = detector.detect_and_draw(frame)
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"[pipeline] processed {frame_idx} frames")

    cap.release()
    writer.release()
    print(f"[pipeline] done → {output_path}")