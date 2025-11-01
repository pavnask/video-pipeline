from ultralytics import YOLO

class YoloDetector:
    def __init__(self, weights: str = "yolov8n.pt", device=None):
        self.model = YOLO(weights)
        self.device = device

    def detect_and_draw(self, frame):
        # you can also pass device here if you have GPU
        results = self.model(frame, verbose=False, device=self.device)
        return results[0].plot()