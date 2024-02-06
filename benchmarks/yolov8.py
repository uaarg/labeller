from typing import Optional

from PIL import Image
import numpy as np
from ultralytics import YOLO

from loader import MultiBundleLoader, Vec2
from benchmarks.detector import BoundingBox, LandingPadDetector

# The 'type: ignore[import-untyped]' annotation here tells mypy to ignore that
# these modules do not have type annotations.


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
        im = np.array(image)
        pred = self.model.predict(im,
                                     conf=self.conf_thres,
                                     iou=self.iou_thres,
                                     imgsz=self.imgsz,
                                     half=self.half,
                                     device=self.device,
                                     max_det=self.max_det,
                                     verbose=False)
        results = []
        for r in pred:
            results = r.boxes.xywh.tolist()
        if len(results) == 0: 
            return None
        else:
            results = results[0]
            x = results[0]
            y = results[1]
            w = results[0]
            h = results[0]
        return BoundingBox(Vec2(x, y), Vec2(w, h))

if __name__ == "__main__":
    import glob
    from benchmarks import benchmark
    detector = Yolov8Detector()
    bundles = MultiBundleLoader(glob.glob("tmp/labelled-bundles/*"))
    benchmark(detector, bundles)