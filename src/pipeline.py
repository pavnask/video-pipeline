# src/pipeline.py

import os
import cv2
import numpy as np
from .video_io import open_video
from .detector import YoloPosePersonDetector

CONF_THRES = 0.4          # confidence for accepting a detection
DEBUG_SAVE_EVERY = 50     # save every Nth annotated frame
ETALON_SHIFT = (15, 15)   # (dx, dy) to shift the etalon a bit, just to see the difference


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

    # this will hold our "ideal" / "etalon" pose as a full-size image
    etalon_canvas = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (w, h))

        # 1) run pose
        result = detector.predict(frame)

        # 2) detections above threshold
        persons = detector.get_person_boxes(result, conf_thres=CONF_THRES)

        # 3) draw ONLY these persons
        if persons:
            keep_ids = [p["index"] for p in persons]
            annotated_frame = detector.annotate_frame(result, keep_indices=keep_ids)
        else:
            annotated_frame = frame.copy()

        # 4) if we DON'T yet have an etalon, create it from THIS frame
        if etalon_canvas is None and persons:
            # build transparent-ish canvas
            etalon_canvas = np.zeros_like(annotated_frame)

            # take the same result, but draw it on EMPTY canvas
            # easiest: draw on a copy of annotated_frame, then paste into canvas
            # (we already have annotated_frame, so we can reuse it)
            etalon_person_frame = annotated_frame.copy()

            # OPTIONAL: shift it slightly so difference is visible
            dx, dy = ETALON_SHIFT
            # create empty canvas then paste shifted
            shifted = np.zeros_like(annotated_frame)
            # we just shift the whole frame for simplicity
            y1 = max(0, dy)
            y2 = min(h, h + dy)
            x1 = max(0, dx)
            x2 = min(w, w + dx)
            # source coords
            src_y1 = 0
            src_y2 = y2 - y1
            src_x1 = 0
            src_x2 = x2 - x1
            shifted[y1:y2, x1:x2] = etalon_person_frame[src_y1:src_y2, src_x1:src_x2]

            etalon_canvas = shifted  # store for future frames

        # 5) if we HAVE an etalon, overlay it at 50% on top of current frame
        if etalon_canvas is not None:
            # current annotated frame (real)
            # etalon (ideal) → we want both visible
            # result = 0.5 * real + 0.5 * ideal
            blended = cv2.addWeighted(annotated_frame, 1.0, etalon_canvas, 0.5, 0)
            annotated_frame = blended

        # 6) save crops (still from current annotated frame!)
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

        # 7) debug save
        if frame_idx % DEBUG_SAVE_EVERY == 0:
            debug_copy = annotated_frame.copy()
            cv2.putText(
                debug_copy,
                f"Frame {frame_idx} | persons={len(persons)}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(
                os.path.join(debug_dir, f"debug_frame_{frame_idx:05d}.jpg"),
                debug_copy,
            )

        # 8) write video
        if writer is not None:
            writer.write(annotated_frame)

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"[pipeline] frame {frame_idx}, kept={len(persons)}")

    cap.release()
    if writer is not None:
        writer.release()

    print("[pipeline] done ✅")