# src/pipeline_two_video.py
#
# Take pose from ETALON video and overlay (ghost) on REAL video.

import os
import cv2
import numpy as np

from .video_io import open_video
from .detector import YoloPosePersonDetector

CONF_THRES = 0.4
DEBUG_SAVE_EVERY = 50
ETALON_ALPHA = 0.5   # transparency of etalon overlay


def run_two_video_pipeline(
    etalon_path="data/etalon.mp4",
    real_path="data/real.mp4",
    output_path="outputs/compare.mp4",
    resize_to=(640, 480),
    save_debug=True,
    also_draw_real=True,
):
    os.makedirs("outputs", exist_ok=True)
    debug_dir = "outputs/debug_two_video"
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)

    cap_etalon = open_video(etalon_path)
    cap_real = open_video(real_path)

    w, h = resize_to
    fps_real = cap_real.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps_real, (w, h))

    detector = YoloPosePersonDetector()

    frame_idx = 0

    while True:
        ret_e, frame_e = cap_etalon.read()
        ret_r, frame_r = cap_real.read()

        # stop when REAL ends (you can change to min or max behaviour)
        if not ret_r or not ret_e:
            break

        frame_e = cv2.resize(frame_e, (w, h))
        frame_r = cv2.resize(frame_r, (w, h))

        # 1) run pose on ETALON
        result_e = detector.predict(frame_e)
        persons_e = detector.get_person_boxes(result_e, conf_thres=CONF_THRES)
        if persons_e:
            keep_e = [p["index"] for p in persons_e]
            etalon_annotated = detector.annotate_frame(result_e, keep_indices=keep_e)
        else:
            # no detection in etalon frame → just blank overlay
            etalon_annotated = np.zeros_like(frame_e)

        # 2) optionally run pose on REAL too (to see actual current pose)
        if also_draw_real:
            result_r = detector.predict(frame_r)
            persons_r = detector.get_person_boxes(result_r, conf_thres=CONF_THRES)
            if persons_r:
                keep_r = [p["index"] for p in persons_r]
                real_annotated = detector.annotate_frame(result_r, keep_indices=keep_r)
            else:
                real_annotated = frame_r.copy()
        else:
            real_annotated = frame_r.copy()

        # 3) overlay: real is base, etalon is ghost on top
        # you can swap the order if you want etalon to dominate
        blended = cv2.addWeighted(real_annotated, 1.0, etalon_annotated, ETALON_ALPHA, 0)

        # 4) (optional) add labels to know which is which
        cv2.putText(
            blended,
            "REAL",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            blended,
            "ETALON (ghost)",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # 5) write result
        writer.write(blended)

        # 6) debug save
        if save_debug and frame_idx % DEBUG_SAVE_EVERY == 0:
            debug_path = os.path.join(debug_dir, f"debug_{frame_idx:05d}.jpg")
            cv2.imwrite(debug_path, blended)

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"[two-video] frame {frame_idx}, etalon persons={len(persons_e) if persons_e else 0}")

    cap_etalon.release()
    cap_real.release()
    writer.release()

    print(f"[two-video] done ✅ → {output_path}")