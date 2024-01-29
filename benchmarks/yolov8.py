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

    def predict(self, source):

       
        print(results)
        
if __name__ == "__main__":
    import glob
    from benchmarks import benchmark
    
    detector = Yolov8Detector()
    bundles = MultiBundleLoader(glob.glob("tmp/labelled-bundles/*"))
    print(bundles.count)




        
