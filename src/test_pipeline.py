from ultralytics import YOLO
import cv2
import os
from src.pipeline import run_pipeline
from src.pipeline_two_video import run_two_video_pipeline

print(">>> HELLO from src/test_pipeline.py (module top-level)")

def main():
    print(">>> main() is running")

    # 1. load model
    model = YOLO("yolov8n.pt")  # small, good for testing

    # 2. open a video
    cap = cv2.VideoCapture("data/sample.mp4")
    if not cap.isOpened():
        raise RuntimeError("Cannot open video. Put a video at data/sample.mp4")

    # 3. output video writer
    os.makedirs("outputs", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("outputs/out.mp4", fourcc, 30.0, (640, 480))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # resize to keep it simple
        frame = cv2.resize(frame, (640, 480))

        # 4. run detection
        results = model(frame, verbose=False)

        # 5. draw boxes
        annotated = results[0].plot()  # ultralytics returns annotated frame

        # 6. write frame
        out.write(annotated)

    cap.release()
    out.release()
    print("Done. Check outputs/out.mp4")

if __name__ == "__main__":
    # run_pipeline(
    #     input_path="data/macdavid.mp4",
    #     output_video_path="outputs/out.mp4",
    #     export_persons_dir="outputs/persons",
    #     save_annotated_video=True,
    # )
    run_two_video_pipeline(
        etalon_path="data/etalon.mp4",
        real_path="data/real.mp4",
        output_path="outputs/compare.mp4",
        resize_to=(640, 480),
        save_debug=True,
        also_draw_real=True,
    )
    #run_pipeline()
    #print(">>> __name__ == '__main__', calling main()")
    #main()