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



class Yolov8Detector():

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

    def predict(self, image: Image.Image) -> Optional[BoundingBox]:

        im0 = np.array(image)

        # Resize and transform im0 into one the model can read
        im, ratio, (dw, dh) = letterbox(im0,
                                        self.imgsz,
                                        stride=self.model.stride,
                                        auto=False,
                                        scaleup=False)  # padded resize
        im = im.transpose((2, 0, 1))  # HWC to CHW
        im = np.ascontiguousarray(im)  # contiguous

        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
        
        results = self.model.predict(im, 
                                     conf=self.conf_thres,
                                     iou=self.iou_thres,
                                     imgsz=self.imgsz,
                                     half=self.half,
                                     device=self.device,
                                     max_det=self.max_det,
                                     )
        print(results)
        
if __name__ == "__main__":
    import glob
    from benchmarks import benchmark
    
    detector = Yolov8Detector()
    bundles = MultiBundleLoader(glob.glob("tmp/labelled-bundles/*"))
    benchmark(detector, bundles)





        
