# src/detector.py

from ultralytics import YOLO

class YoloPosePersonDetector:
    """
    Pose model wrapper: runs yolov8n-pose.pt (or bigger),
    can return boxes filtered by confidence, and can draw the skeletons.
    """
    def __init__(self, weights: str = "yolov8n-pose.pt", device=None):
        self.model = YOLO(weights)
        self.device = device

    def predict(self, frame):
        """Run pose model on a frame and return the first result."""
        results = self.model(frame, verbose=False, device=self.device)
        return results[0]

    def get_person_boxes(self, result, conf_thres: float = 0.0):
        """
        Pose results still have .boxes — we use them to crop persons.
        Returns: list of {index, xyxy, conf}
        """
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        people = []
        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            if conf < conf_thres:
                continue
            xyxy = boxes.xyxy[i].tolist()
            people.append(
                {
                    "index": i,
                    "xyxy": xyxy,
                    "conf": conf,
                }
            )
        return people

    def annotate_frame(self, result, keep_indices=None):
        """
        Draw pose (keypoints + lines) on the frame.
        If keep_indices is provided, draw only those.
        """
        if keep_indices is not None:
            # keep only selected detections
            result.boxes = result.boxes[keep_indices]
            if result.keypoints is not None:
                result.keypoints = result.keypoints[keep_indices]
        return result.plot()