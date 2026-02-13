from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import torch

class HandDetector:
    def __init__(self, model_path, cfg_path="./hand_detector.d2/faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml"):
        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)

        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.predictor = DefaultPredictor(cfg)

    def detect(self, image):
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()

        hands = []
        for box, score in zip(boxes, scores):
            hands.append({
                "bbox": [int(round(x)) for x in box.tolist()],
                "confidence": round(float(score), 1)
            })
        return hands
