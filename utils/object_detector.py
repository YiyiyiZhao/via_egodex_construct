import sys
import cv2
from ultralytics import YOLOWorld
from pathlib import Path

# Import all classes from pre_trajectory
# utils/object_detector.py -> utils/ -> via_egodex_construct/ -> pre_trajectory/
sys.path.append(str(Path(__file__).parent.parent / "pre_trajectory"))
from yolo_world_classes import YOLO_WORLD_CLASSES


class ObjectDetector:
    def __init__(self, model_name="yolov8s-worldv2.pt", custom_classes=None):
        """
        Object detector using YOLO-World with custom vocabulary

        :param model_name: YOLO-World model
                          - yolov8s-worldv2.pt (small, fast)
                          - yolov8m-worldv2.pt (medium, balanced)
                          - yolov8l-worldv2.pt (large, accurate)
        :param custom_classes: List of class names, if None uses all 786 classes
        """
        print(f"Initializing YOLO-World detector with {model_name}...")
        self.model = YOLOWorld(model_name)

        # Set custom classes
        self.classes = custom_classes if custom_classes else YOLO_WORLD_CLASSES
        print(f"Setting {len(self.classes)} object classes...")
        self.model.set_classes(self.classes)
        print("✓ Object detector initialized")

    def detect(self, image, conf_thres=0.3):
        """
        Detect objects in image

        :param image: OpenCV image (numpy array, BGR)
        :param conf_thres: confidence threshold (0.3 recommended for YOLO-World)

        :return: [{'bbox':[x1,y1,x2,y2], 'confidence':0.9, 'class':0, 'name':'bottle'}, ...]
        """
        results = self.model(image, verbose=False)

        objects = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0].cpu().numpy())
                if conf < conf_thres:
                    continue

                cls = int(box.cls[0].cpu().numpy())
                name = self.classes[cls]  # Use our custom class names

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                objects.append({
                    "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                    "confidence": round(conf, 2),  # More precision for debugging
                    "class": cls,
                    "name": name
                })

        return objects
