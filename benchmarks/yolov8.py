from typing import Optional

from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO

from loader import MultiBundleLoader, Vec2
from benchmarks.detector import BoundingBox, LandingPadDetector

# The 'type: ignore[import-untyped]' annotation here tells mypy to ignore that
# these modules do not have type annotations.
from models.common import DetectMultiBackend  # type: ignore[import-untyped]
from utils.augmentations import letterbox  # type: ignore[import-untyped]
from utils.general import non_max_suppression, check_img_size  # type: ignore[import-untyped]
from utils.torch_utils import select_device  # type: ignore[import-untyped]



class Yolov8Detector(LandingPadDetector):

    def __init__(self, 
                 weights='benchmarks/yolov8_nano.pt',
                 device='cpu',
                 imgsz = 640,
                 half = False
                 ) -> None:

        self.model = YOLO(weights)
        self.device = device
        self.conf_thres = 0.25
        self.imgsz = imgsz
        self.half = half
        self.max_det = 1000
        self.iou_thres = 0.45
        super().__init__()

    def predict(self, image: Image.Image) -> Optional[BoundingBox]:
        """
        Runs the model on the supplied images

        Returns a list with dictionary items for each prediction
        """
        im0 = np.array(image)
        
        pred = self.model.predict(im0,
                                     conf=self.conf_thres,
                                     iou=self.iou_thres,
                                     imgsz=self.imgsz,
                                     half=self.half,
                                     device=self.device,
                                     max_det=self.max_det,
                                     )
        results = []
        for prediction in pred:
            if prediction.shape[0] == 0:
                continue

            x1, y1, x2, y2 = prediction[0, :4].tolist()
            results.append({
                'confidence': prediction[0, 4].item(),
                'x': min(x1, x2) - dw,
                'y': min(y1, y2) - dh,
                'w': abs(x2 - x1),
                'h': abs(y2 - y1)
            })

        if not results:
            return None

        # Return the max-confidence result
        max_confidence = max(results, key=lambda x: x['confidence'])
        x, y, w, h = max_confidence['x'], max_confidence['y'], max_confidence[
            'w'], max_confidence['h']
        return BoundingBox(Vec2(x, y), Vec2(w, h))
        
if __name__ == "__main__":
    import glob
    from benchmarks import benchmark
    
    detector = Yolov8Detector()
    bundles = MultiBundleLoader(glob.glob("tmp/labelled-bundles/*"))
    benchmark(detector, bundles)





        
