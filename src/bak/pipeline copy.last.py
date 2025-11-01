# src/pipeline.py

import os
import cv2
from .video_io import open_video
from .detector import YoloPosePersonDetector

CONF_THRES = 0.4          # Confidence threshold for poses
DEBUG_SAVE_EVERY = 50     # Save every Nth annotated frame for debugging


def run_pipeline(
    input_path="data/sample.mp4",
    output_video_path="outputs/out.mp4",
    resize_to=(640, 480),
    export_persons_dir="outputs/persons",
    save_annotated_video=True,
):
    # --- setup output dirs ---
    os.makedirs("outputs", exist_ok=True)
    os.makedirs(export_persons_dir, exist_ok=True)
    debug_dir = "outputs/debug_frames"
    os.makedirs(debug_dir, exist_ok=True)

    cap = open_video(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = resize_to

    writer = None
    if save_annotated_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    detector = YoloPosePersonDetector()

    frame_idx = 0
    crop_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (w, h))

        # 1️⃣ Run pose model
        result = detector.predict(frame)

        # 2️⃣ Filter by confidence
        persons = detector.get_person_boxes(result, conf_thres=CONF_THRES)

        # 3️⃣ Annotate only those persons
        if persons:
            keep_ids = [p["index"] for p in persons]
            annotated_frame = detector.annotate_frame(result, keep_indices=keep_ids)
        else:
            annotated_frame = frame.copy()

        # 4️⃣ Save debug annotated frame
        if frame_idx % DEBUG_SAVE_EVERY == 0:
            debug_path = os.path.join(debug_dir, f"debug_frame_{frame_idx:05d}.jpg")
            # optional overlay info
            overlay = annotated_frame.copy()
            cv2.putText(
                overlay,
                f"Frame {frame_idx} | Persons: {len(persons)}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(debug_path, overlay)

        # 5️⃣ Save crops of confident persons (from annotated frame)
        for p in persons:
            x1, y1, x2, y2 = [int(v) for v in p["xyxy"]]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            crop = annotated_frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            out_path = os.path.join(
                export_persons_dir,
                f"frame{frame_idx:06d}_person{crop_idx:03d}.jpg"
            )
            cv2.imwrite(out_path, crop)
            crop_idx += 1

        # 6️⃣ Write video frame
        if writer is not None:
            writer.write(annotated_frame)

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"[pipeline] frame {frame_idx}, kept={len(persons)}")

    # --- cleanup ---
    cap.release()
    if writer is not None:
        writer.release()

    print(f"[pipeline] done ✅ persons → {export_persons_dir}")
    print(f"[pipeline] debug frames → {debug_dir}")